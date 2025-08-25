# SpecVLM
This repository contains the implementation of SpecVLM.


### Requirements
Install the required dependencies and other necessary dependencies:
```bash
pip install -r requirements.txt
```

Download the model checkpoints (if needed):
- For LLaVA-OneVision models: https://huggingface.co/llava-hf
- For Qwen2.5-VL models: https://huggingface.co/Qwen



## Usage
### Command Line Arguments
The main script (`main.py`) supports the following arguments:

--model_type          Model type (llava_ov or qwen2_5_vl)
--base_model_path     Path to the base model
--draft_model_path    Path to the draft model
--task                Evaluation task (VideoDetailCaption, MVBench, MVLU, LongVideoBench, MMBench)
--frame_num           Number of frames per video
--evaluation_num      Number of evaluation samples
--max_new_tokens      Maximum number of tokens to generate
--drop_rate           Pruning rate for token pruning
--data_num            Number of data samples to load
--save_path           Path to save results
--gpu_ids             GPU IDs to use

### Quick Evaluation
```bash
sh run.sh
```
