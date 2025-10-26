export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# Define parameters with comments
MODEL_TYPE="llava_ov"             # Model type: llava_ov or qwen2_5_vl
BASE_MODEL_PATH="/data/ycji/models/llava-onevision-qwen2-72b-ov-hf"  # Path to base model
DRAFT_MODEL_PATH="/data/ycji/models/llava-onevision-qwen2-7b-ov-hf"  # Path to draft model

TASK="VideoDetailCaption"         # Task type: VideoDetailCaption, MVBench, MVLU, LongVideoBench, MMBench
DATA_PATH="/data/ycji/datasets/VideoDetailCaption"  # Path to dataset

EVAL_NUM=1                        # Number of evaluation samples
MAX_NEW_TOKENS=256                # Number of new tokens to generate
DATA_NUM=100                      # Number of data samples to load
DROP_RATE=0.9                     # Pruning ratio
GPU_IDS="1,2,3"                   # GPU IDs to use

# A larger number of frames is generally recommended, as permitted by your GPU memory capacity and bandwidth. Memory bottlenecks are typically triggered by long visual sequence. 
# Example: 
#   - On NVIDIA A100 GPUs, we recommend using 128 frames for the LLaVA-OV 7B target model and 64 frames for the 72B model. 
# Qwen2.5-VL currently does not support specifying input length directly. To control the input length, you will need to adjust the frame number accordingly.
# In some cases, current model doesn't support large frames as input. We are working on the implementation of SpecVLM for LLaVA-Video.
FRAME_NUM=160                     

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
