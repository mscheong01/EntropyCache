
#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

MODEL_NAME="llada-inst"
# MODEL_NAME="dream-inst"
OUTPUT_ROOT_DIR="./outputs-eval-basic/${MODEL_NAME}"
DATASET_NAMES=("gsm8k" "math-500" "mbpp" "humaneval")
# DATASET_NAMES=("mbpp" "humaneval")


for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    CACHE_TYPE="entropy"
    OUTPUT_DIR="$OUTPUT_ROOT_DIR/$CACHE_TYPE"
    GEN_METHOD="sliding_window"

    HYDRA_FULL_ERROR=1 accelerate launch \
        --num_machines 1 \
        eval.py \
        dataset.name=$DATASET_NAME \
        hydra.run.dir=${OUTPUT_DIR}/$DATASET_NAME/$(date +"%Y-%m-%d_%H-%M-%S") \
        batch_size=1 \
        seed=1234 \
        cache=$CACHE_TYPE \
        cache.entropy_threshold=1.5 \
        cache.k_recent_keep=64 \
        generation=$GEN_METHOD \
        generation.window_size=32 \
        model=$MODEL_NAME \
        generation.threshold=0.9

    CACHE_TYPE="elasticcache"
    OUTPUT_DIR="$OUTPUT_ROOT_DIR/$CACHE_TYPE/"
    GEN_METHOD="sliding_window"
    HYDRA_FULL_ERROR=1 accelerate launch \
        --num_machines 1 \
        eval.py \
        dataset.name=$DATASET_NAME \
        hydra.run.dir=${OUTPUT_DIR}/$DATASET_NAME/$(date +"%Y-%m-%d_%H-%M-%S") \
        batch_size=1 \
        seed=1234 \
        cache=$CACHE_TYPE \
        generation=$GEN_METHOD \
        generation.window_size=32 \
        model=$MODEL_NAME \
        generation.threshold=0.9

    CACHE_TYPE="prefix"
    OUTPUT_DIR="$OUTPUT_ROOT_DIR/$CACHE_TYPE/"
    GEN_METHOD="sliding_window"
    HYDRA_FULL_ERROR=1 accelerate launch \
        --num_machines 1 \
        eval.py \
        dataset.name=$DATASET_NAME \
        hydra.run.dir=${OUTPUT_DIR}/$DATASET_NAME/$(date +"%Y-%m-%d_%H-%M-%S") \
        batch_size=1 \
        seed=1234 \
        cache=$CACHE_TYPE \
        cache.use_dual=true \
        generation=vanilla \
        generation.block_length=32 \
        model=$MODEL_NAME \
        generation.threshold=0.9

    CACHE_TYPE="d2cache"
    OUTPUT_DIR="$OUTPUT_ROOT_DIR/$CACHE_TYPE/"
    GEN_METHOD="sliding_window"
    HYDRA_FULL_ERROR=1 accelerate launch \
        --num_machines 1 \
        eval.py \
        dataset.name=$DATASET_NAME \
        hydra.run.dir=${OUTPUT_DIR}/$DATASET_NAME/$(date +"%Y-%m-%d_%H-%M-%S") \
        batch_size=1 \
        seed=1234 \
        cache=$CACHE_TYPE \
        generation=$GEN_METHOD \
        generation.window_size=32 \
        model=$MODEL_NAME \
        generation.threshold=0.9

    OUTPUT_DIR="$OUTPUT_ROOT_DIR/baseline/"
    GEN_METHOD="sliding_window"
    HYDRA_FULL_ERROR=1 accelerate launch \
        --num_machines 1 \
        eval.py \
        dataset.name=$DATASET_NAME \
        hydra.run.dir=${OUTPUT_DIR}/$DATASET_NAME/$(date +"%Y-%m-%d_%H-%M-%S") \
        batch_size=1 \
        seed=1234 \
        generation=$GEN_METHOD \
        generation.window_size=32 \
        model=$MODEL_NAME
done
