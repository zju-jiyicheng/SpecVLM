import os
import torch
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from tqdm import tqdm
import json
import argparse
from tree_choices.choices import mc_sim_7b_63, chain
from utils.utils import *
# from visualize import *




def run_eval(model_type, model, draft_model, data_video, task, frame_num, evaluation_num, max_new_tokens, drop_rate, video_token_id, save_path=None, data_path=None, processor=None):
    # Run evaluation
    model.eval()
    draft_model.eval()

    results = {}

    # Two stages
    results['ar_two_stage_decode'] = []

    results['sd_tree_two_stage_decode'] = []
    results['sd_tree_two_stage_accept_length'] = []

    results['specvlm_decode'] = []
    results['specvlm_accept_length'] = []

    for i in tqdm(range(evaluation_num)):
        data_instance = data_video[i]

        # AR two stage
        inputs = clip_input_video(processor, task, data_instance,frame_num = frame_num, model_type = model_type, data_path=data_path)
        if inputs == None:
            continue
        # print("judge:",inputs['input_ids'].shape[1])
        if model_type == 'qwen2_5_vl' and inputs['input_ids'].shape[1] > frame_num*130:
            continue
        output_ar = AR_generate_two_stage(inputs, model ,max_new_tokens=max_new_tokens, video_token_id=video_token_id)
        
        print("\n")
        print("-------Autoregressive Decoding-------")
        # print("Inference Time:", output_ar['inference_time'])
        print("Decoding Time:", output_ar['decoding_time'])
        output_text = processor.batch_decode(output_ar['output_ids'], skip_special_tokens=True)[0]
        print("Output:")
        print(output_text)
        print("\n")
        results['ar_two_stage_decode'].append(output_ar['decoding_time'])

        # SD tree two stage  
        inputs = clip_input_video(processor, task, data_instance,frame_num = frame_num, model_type = model_type, data_path=data_path)
        output_sd = SD_generate_two_stage(
                inputs,
                model,
                draft_model,
                processor,
                max_new_tokens=max_new_tokens,
                video_token_id=video_token_id,
                tree_choices=mc_sim_7b_63,
        )
        print("\n")
        print("-------Naive Speculative Decoding (with tree attn)-------")
        # print("Inference Time:", output_sd['inference_time'])
        print("Decoding Time:", output_sd['decoding_time'])
        print("Average Accept Length:", output_sd["mean_accept_length"].item())
        output_text = processor.batch_decode(output_sd['output_ids'], skip_special_tokens=True)[0]
        print("Output:")
        print(output_text)
        print("\n")
        results['sd_tree_two_stage_decode'].append(output_sd['decoding_time'])
        results['sd_tree_two_stage_accept_length'].append(output_sd["mean_accept_length"])

        # SpecVLM
        inputs = clip_input_video(processor, task, data_instance,frame_num = frame_num, model_type = model_type, data_path=data_path)
        output_specvlm = SD_generate_with_pruning(
                inputs,
                model,
                draft_model,
                processor,
                method = 'attention_plus_uniform',
                drop_rate = drop_rate,
                video_token_id = video_token_id,
                max_new_tokens=max_new_tokens,
                tree_choices=mc_sim_7b_63,
                percentage=0.5,
        )
        print("\n")
        print("-------SpecVLM-------")
        # print("Inference Time:", output_specvlm['inference_time'])
        print("Decoding Time:", output_specvlm['decoding_time'])
        print("Average Accept Length:", output_specvlm["mean_accept_length"].item())
        output_text = processor.batch_decode(output_specvlm['output_ids'], skip_special_tokens=True)[0]
        print("Output:")
        print(output_text)
        print("\n")
        results['specvlm_decode'].append(output_specvlm['decoding_time'])
        results['specvlm_accept_length'].append(output_specvlm["mean_accept_length"])

    
    # Save results
    if save_path is not None:
        
        # Compute mean
        print("\n")
        print("-------Average Results-------")

        # Two stage
        print("Autoregressive Decoding Time:", sum(results['ar_two_stage_decode'])/len(results['ar_two_stage_decode']))
        print("\n")

        print("Naive SD Decoding Time:", sum(results['sd_tree_two_stage_decode'])/len(results['sd_tree_two_stage_decode']))
        print("Naive SD Average Accept Length:", (sum(results['sd_tree_two_stage_accept_length'])/len(results['sd_tree_two_stage_accept_length'])).item())
        print("\n")

        print("SpecVLM:", sum(results['specvlm_decode'])/len(results['specvlm_decode']))
        print("SpecVLM Average Accept Length:", (sum(results['specvlm_accept_length'])/len(results['specvlm_accept_length'])).item())
        print("\n")
        
        print("-------End-------")

        metrics = {
        "Autoregressive Decoding Time": float(sum(results['ar_two_stage_decode'])/len(results['ar_two_stage_decode'])),
    
        "Naive SD": {
            "Decoding Time": float(sum(results['sd_tree_two_stage_decode'])/len(results['sd_tree_two_stage_decode'])),
            "Average Accept Length": float(sum(results['sd_tree_two_stage_accept_length'])/len(results['sd_tree_two_stage_accept_length']))
        },
        
        "specvlm": {
            "Decoding Time": float(sum(results['specvlm_decode'])/len(results['specvlm_decode'])),
            "Average Accept Length": float(sum(results['specvlm_accept_length'])/len(results['specvlm_accept_length']))
        }
        }
        
        # Write to jsonl file
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json_str = json.dumps(metrics, indent=4)
            f.write(json_str + '\n\n')


if __name__ == "__main__":
    # Create command line argument parser
    parser = argparse.ArgumentParser(description='Run video model evaluation')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='llava_ov', 
                        choices=['llava_ov', 'qwen2_5_vl'], 
                        help='Model type: llava_ov or qwen2_5_vl')
    parser.add_argument('--base_model_path', type=str, 
                        default=None,
                        help='Path to the base model')
    parser.add_argument('--draft_model_path', type=str, 
                        default=None,
                        help='Path to the draft model')

    parser.add_argument('--data_path', type=str,
                        default='/data',
                        help='Path to the data directory')
    
    # Evaluation parameters
    parser.add_argument('--task', type=str, default='VideoDetailCaption',
                        choices=['VideoDetailCaption', 'MVBench', 'MVLU', 'LongVideoBench', 'MMBench'],
                        help='Evaluation task type')
    parser.add_argument('--frame_num', type=int, default=8,
                        help='Number of frames per video')
    parser.add_argument('--evaluation_num', type=int, default=1,
                        help='Number of evaluation samples')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--drop_rate', type=float, default=0.9,
                        help='Pruning rate')
    parser.add_argument('--data_num', type=int, default=100,
                        help='Number of data samples to load')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save results. If not specified, a default path will be used')
    parser.add_argument('--gpu_ids', type=str, default="0,1,2,3,4,5",
                        help='GPU IDs to use')
    parser.add_argument('--setting', type=str, default='standard',
                        choices=['self', 'standard'],
                        help='Speculative Decoding setting')

    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Set GPU environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    # Import appropriate decoding functions based on model type
    if args.model_type == 'llava_ov':
        from decoding.tree_decoding import *
    else:
        from decoding.tree_decoding_qwen2_5 import *
    
    # Load models
    model, draft_model, processor, video_token_id = load_model(args.model_type, args.base_model_path, args.draft_model_path)
    #TODO: For 'self' setting, load draft model as base model
    
    # Load data
    data_video = load_data(args.task, args.data_num, args.data_path)
    
    # Set save path
    if args.save_path is None:
        save_path = f"results/{args.model_type}_{args.task}_{args.drop_rate}"
    else:
        save_path = args.save_path
    
    # Run evaluation
    run_eval(
        args.model_type,
        model=model, 
        draft_model=draft_model, 
        data_video=data_video, 
        task=args.task, 
        frame_num=args.frame_num, 
        evaluation_num=args.evaluation_num, 
        max_new_tokens=args.max_new_tokens,
        drop_rate=args.drop_rate,
        video_token_id=video_token_id,
        save_path=save_path,
        data_path=args.data_path,
        processor=processor,
    )