from datetime import timedelta
import os
import accelerate
import re
import torch

from omegaconf import DictConfig
from lm_eval.api.model import TemplateLM
from lm_eval.api.instance import Instance
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from tqdm import tqdm

from src.frame import Frame
from src.generation import generate, decode_final_frame, generate_, prefill
from src.utils import Timer, load_pretrained_model
from src.eval_model.sanitize import sanitize


class EvalModelBase(TemplateLM):
    """
    Mixin class for testing speed of a model.
    """

    def __init__(self, cfg: DictConfig, **kwargs):

        # setup facilities...
        accelerator_kwargs = accelerate.InitProcessGroupKwargs(
            timeout=timedelta(weeks=52)
        )
        self.accelerator = accelerate.Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.model = (
            load_pretrained_model(
                cfg,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation=cfg.attn_implementation,
            )
            .eval()
            .to(self.accelerator.device)
        )
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
            AutoTokenizer.from_pretrained(cfg.model.path, trust_remote_code=True)
        )

        # setup properties from LM
        self._rank = self.accelerator.local_process_index
        self._world_size = self.accelerator.num_processes

        # setup custom properties
        self.cfg = cfg
        self.throughput = None
        self.tps = None
        self.prefill_latency = None
        self.latency = None
        self.total_time = None
        self.recompute_ratio = None
        self.device = self.accelerator.device
        self.extra_gen_kwargs = kwargs.get("extra_gen_kwargs", {})

        if (drafter := self.extra_gen_kwargs.get("drafter")) is not None:
            if drafter == "self":
                self.extra_gen_kwargs["drafter"] = self.model
            else:
                self.extra_gen_kwargs["drafter"] = drafter.eval().to(
                    self.accelerator.device
                )

    def postprocess_code_humaneval(self, doc, code: str) -> str:
        return sanitize(
            doc["prompt"] + "\n" + code.split("```python\n", 1)[-1].split("```")[0],
            doc["entry_point"],
        )

    def postprocess_code_mbpp(self, doc, code: str) -> str:
        import re
        extracted_code = code
        match = re.search(r"```(?:python|py)?\s+([\s\S]*?)\s+```", code, flags=re.IGNORECASE)
        if match:
            extracted_code = match.group(1)
        else:
            for stop_word in ["\nExplanation:", "\nNote:", "\nHere is the code"]:
                if stop_word in extracted_code:
                    extracted_code = extracted_code.split(stop_word)[0]
        return sanitize(extracted_code)

    def tok_encode(self, string: str, **kwargs):
        return self.tokenizer(string, **kwargs).input_ids

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError

    def _loglikelihood_tokens(self, requests, **kwargs):
        raise NotImplementedError

    @property
    def eot_token_id(self) -> int:  # type: ignore
        try:
            return int(os.environ.get("EOT_TOKEN_ID"))  # type: ignore
        except TypeError:
            return self.tokenizer.eos_token_id  # type: ignore

    def generate_until(self, requests: list[Instance], disable_tqdm: bool = False):
        """Generate greedily until a stopping sequence

        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context, gen_kwargs).
            context: str
                Context string
            gen_kwargs: dict
                A dictionary of keyword arguments to pass to the generation function e.g. top_k, until, etc.
        :return: list[str]
            A list of model generated continuations.
            continuation: str
                The generated continuation.
        """
        prefill_latency = []
        out, throughput, tps = [], [], []
        recompute_ratio = []
        latency = []
        for instance in tqdm(
            requests,
            desc="Generating...",
            disable=disable_tqdm or not self.accelerator.is_main_process,
        ):
            context, until = instance.args  # type: ignore
            if self.cfg.generation.get("add_bos_token", False):
                context = self.tokenizer.bos_token + context
            dname = self.cfg.dataset.name
            if "mbpp" in dname or "humaneval" in dname:
                is_code_task = True
            else:
                is_code_task = False
            context = self.tokenizer(context, return_tensors="pt").input_ids
            try:

                with Timer("prefill") as ptimer:
                    cache, initial_frame, initial_delta, attention_mask = prefill(
                        self.model,
                        context,
                        **self.cfg.generation,
                        **self.extra_gen_kwargs,
                        tokenizer=self.tokenizer,
                    )
                pl = ptimer.elapsed_time_s
                prefill_latency.append(pl)
                with Timer("generate") as timer:
                    decode_record = generate_(
                        self.model,
                        cache,
                        initial_frame,
                        initial_delta,
                        attention_mask,
                        **self.cfg.generation,
                        **self.extra_gen_kwargs,
                        tokenizer=self.tokenizer,
                    )

                throughput.append(timer.token_per_second(decode_record, False))
                tps.append(timer.token_per_step(decode_record, False))
                latency.append(timer.elapsed_time_s + pl)
                recompute_ratio.append(decode_record.recompute_ratio)
                final_frame: Frame = decode_record[-1, 0]
                generated_answer = decode_final_frame(
                    self.tokenizer,
                    final_frame,
                    stop_words=None,
                    skip_special_tokens=True,
                )
                assert isinstance(
                    generated_answer, str
                ), f"Expected generated_answer to be a string, but got {type(generated_answer)}"

                if "humaneval" in dname:
                    generated_answer = self.postprocess_code_humaneval(
                        instance.doc, generated_answer
                    )
                elif "mbpp" in dname:
                    generated_answer = self.postprocess_code_mbpp(
                        instance.doc, generated_answer
                    )
                    
                out.append(generated_answer)

            except torch.cuda.OutOfMemoryError:
                out.append("[out-of-memory]")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        prefill_latency = self.accelerator.gather_for_metrics(prefill_latency)
        tps = self.accelerator.gather_for_metrics(tps)
        throughput = self.accelerator.gather_for_metrics(throughput)
        latency = self.accelerator.gather_for_metrics(latency)
        recompute_ratio = self.accelerator.gather_for_metrics(recompute_ratio)
        
        if self.accelerator.is_main_process:
            self.prefill_latency = sum(prefill_latency) / len(prefill_latency)
            self.tps = sum(tps) / len(tps)
            self.throughput = sum(throughput) / len(throughput)
            self.latency = sum(latency) / len(latency)
            self.total_time = Timer.get_cumulative_s("prefill") + Timer.get_cumulative_s("generate")
            self.recompute_ratio = sum(recompute_ratio) / len(recompute_ratio)

        return out

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def apply_chat_template(
        self, chat_history, add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )
        assert isinstance(chat_templated, str)
        return chat_templated
