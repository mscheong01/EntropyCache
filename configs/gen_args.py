from loguru import logger
from pydantic import BaseModel, Field, model_validator
from typing import Any, Literal
from hydra.core.hydra_config import HydraConfig


class GenerationArgs(BaseModel):
    gen_length: int = Field(ge=0)
    block_length: int = Field(ge=0)
    steps: int = Field(ge=0)

    alg: Literal["maskgit_plus", "entropy", "topk_margin"] = Field(
        default="maskgit_plus"
    )
    temperature: float | None = Field(default=None, ge=0)
    top_k: int | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    threshold: float | None = Field(default=None)
    debias: bool = Field(default=False)

    cache_args: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_constraints(self):
        if self.block_length > self.gen_length or self.steps > self.gen_length:
            raise ValueError(
                f"{self.block_length=} and {self.steps=} must be <= {self.gen_length=}"
            )
        num_blocks = self.gen_length // self.block_length
        if self.gen_length % self.block_length != 0 or self.steps % num_blocks != 0:
            raise ValueError(
                f"{self.gen_length=} must be divisible by {self.block_length=} and {self.steps=} must be divisible by number of blocks {num_blocks=}"
            )

        return self


def get_generation_args(task: str, model: str, cache: str | None = None):
    cache_args = {}
    alg = "maskgit_plus"
    block_length = None
    debias = False
    temperature, top_p, top_k = 0.0, None, None
    threshold = None

    match task:
        case (
            "gsm8k"
            | "mbpp"
            | "mbpp_instruct"   
            | "mbpp_plus"
            | "gpqa_main_generative_n_shot"
            | "mmlu_pro"
            | "mmlu_cot_fewshot"
            | "coqa"
            | "arc_challenge"
            | "drop"
            | "triviaqa"
        ):
            gen_length = 256
        case "humaneval"  | "humaneval_instruct" | "humaneval_plus" | "math-500" | "gsm8k_cot":
            gen_length = 512
        case task if "longbench" in task:
            gen_length = 512
        case task if ("mmlu" in task) and ("cot" in task):
            gen_length = 256
        case "thruthfulqa_gen":
            gen_length = 256
        case "bbh":
            gen_length = 1024
        case task if "gpqa" in task:
            gen_length = 256
        case _:
            logger.info(
                f"Unsupported task {task}, you should specify in {__file__}."
                " Using default gen_length=512."
            )
            gen_length = 512

    # set based on cache
    match cache:
        case "heat":
            sigma = 10.0
            cache_args = {
                "rollout_p": 0.1,
                "current_k": 32,
                "sigma": sigma,
            }
        case "dllm":
            kp, kr = 50, 4
            match task:
                case "gsm8k" | "gsm8k_cot":
                    match model:
                        case "llada-base":
                            kp, kr = 25, 5
                        case "llada-inst":
                            kp, kr = 50, 7
                        case "dream-base":
                            kp, kr = 100, 8
                        case "dream-inst":
                            kp, kr = 25, 2
                case "humaneval":
                    match model:
                        case "llada-base":
                            kp, kr = 50, 5
                        case "llada-inst":
                            kp, kr = 25, 5
                        case "dream-base":
                            kp, kr = 5, 1
                        case "dream-inst":
                            kp, kr = 50, 1
                case "math-500":
                    match model:
                        case "llada-base":
                            kp, kr = 50, 8
                        case "llada-inst":
                            kp, kr = 50, 1
                        case "dream-base":
                            kp, kr = 100, 4
                        case "dream-inst":
                            kp, kr = 50, 1
                case "mbpp":
                    match model:
                        case "llada-base":
                            kp, kr = 25, 4
                        case "llada-inst":
                            kp, kr = 100, 5
                        case "dream-base":
                            kp, kr = 25, 8
                        case "dream-inst":
                            kp, kr = 10, 8
                case "gpqa_main_generative_n_shot":
                    match model:
                        case "llada-base":
                            kp, kr = 100, 8
                        case "llada-inst":
                            kp, kr = 50, 6
                        case "dream-base":
                            kp, kr = 100, 8
                        case "dream-inst":
                            kp, kr = 10, 8 
                case "mmlu_pro":
                    match model:
                        case "llada-base":
                            kp, kr = 100, 6
                        case "llada-inst":
                            kp, kr = 50, 3
                        case "dream-base":
                            kp, kr = 25, 2
                        case "dream-inst":
                            kp, kr = 5, 1 
            cache_args = {"kp": kp, "kr": kr}

    # set based on model
    match model:
        case "dream-base" | "dream-inst":
            top_p = 0.9

    block_length = 32 if model.endswith("inst") else gen_length
    steps = gen_length

    return GenerationArgs(
        gen_length=gen_length,
        block_length=block_length,
        steps=steps,
        alg=alg,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        threshold=threshold,
        debias=debias,
        cache_args=cache_args,
    )
