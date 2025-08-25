export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# Define parameters with comments
MODEL_TYPE="llava_ov"             # Model type: llava_ov or qwen2_5_vl
BASE_MODEL_PATH="/data/ycji/models/llava-onevision-qwen2-7b-ov-hf"  # Path to base model
DRAFT_MODEL_PATH="/data/ycji/models/llava-onevision-qwen2-7b-ov-hf"  # Path to draft model
DATA_PATH="/data/ycji/datasets/VideoDetailCaption"  # Path to dataset
TASK="VideoDetailCaption"         # Task type: VideoDetailCaption, MVBench, MVLU, LongVideoBench, MMBench
EVAL_NUM=1                        # Number of evaluation samples
MAX_NEW_TOKENS=256                # Maximum number of new tokens to generate
DATA_NUM=100                      # Number of data samples to load
DROP_RATE=0.9                     # Pruning rate
GPU_IDS="1,2,3"                   # GPU IDs to use

# A bigger frame number is suggested for better performance, according to the GPU memory and bandwidth.
# e.g. For Nvidia A100 GPUs, we suggest 128 and 64 for llava-ov 7b and 72b target model. For H200 GPUs, we suggest 256 and 128 for llava-ov 7b and 72b target model.
FRAME_NUM=256                       # Number of frames per video

# Run evaluation
CUDA_VISIBLE_DEVICES=$GPU_IDS python run.py \
    --model_type $MODEL_TYPE \
    --base_model_path $BASE_MODEL_PATH \
    --draft_model_path $DRAFT_MODEL_PATH \
    --data_path $DATA_PATH \
    --task $TASK \
    --frame_num $FRAME_NUM \
    --evaluation_num $EVAL_NUM \
    --max_new_tokens $MAX_NEW_TOKENS \
    --drop_rate $DROP_RATE \
    --data_num $DATA_NUM \
    --gpu_ids $GPU_IDS \
    --save_path "results/${MODEL_TYPE}_${TASK}_drop_rate_${DROP_RATE}" \
    --setting "standard" \
