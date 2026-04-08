import os
import torch
from tqdm import tqdm
import json
import argparse
from tree_choices.choices import mc_sim_7b_63, chain
from utils.utils import *
# from visualize import *


def _to_python_scalar(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.flatten()[0].item()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value

def _normalize_generate_len(value):
    value = _to_python_scalar(value)
    return int(round(float(value)))


def _normalize_float(value):
    value = _to_python_scalar(value)
    return round(float(value), 2)


def _build_record(sample_index, method, output_dict, output_text, include_accept_length=False):
    record = {
        "sample_index": sample_index,
        "method": method,
        "decoding_time": _normalize_float(output_dict["decoding_time"]),
        "generate_len": _normalize_generate_len(output_dict["generate_len"]),
    }
    if include_accept_length:
        record["average_accept_length"] = _normalize_float(output_dict["mean_accept_length"])
    record["output"] = output_text
    return record


def _print_record(record):
    print(json.dumps(record, ensure_ascii=False, indent=2))


def _build_metric_record(method, decoding_times, generate_lens, accept_lengths=None):
    mean_decoding_time = sum(decoding_times) / len(decoding_times)
    mean_generate_len = sum(generate_lens) / len(generate_lens)
    avg_decoding_time = round(mean_decoding_time, 2)
    avg_generate_len = int(round(mean_generate_len))

    metric_record = {
        "method": method,
        "avg_generate_len": avg_generate_len,
        "avg_decoding_time": avg_decoding_time,
        "avg_tokens_per_second": round(mean_generate_len / mean_decoding_time, 2) if mean_decoding_time else None,
    }

    if accept_lengths:
        metric_record["avg_accept_length"] = round(sum(accept_lengths) / len(accept_lengths), 2)

    return metric_record




def run_eval(model_type, model, draft_model, data_video, task, frame_num, evaluation_num, max_new_tokens, drop_rate, video_token_id, save_path=None, data_path=None, processor=None):
    # Run evaluation
    model.eval()
    draft_model.eval()

    results = {
        "ar": {
            "decoding_times": [],
            "generate_lens": [],
        },
        "naive_sd": {
            "decoding_times": [],
            "generate_lens": [],
            "accept_lengths": [],
        },
        "specvlm": {
            "decoding_times": [],
            "generate_lens": [],
            "accept_lengths": [],
        },
    }
    sample_records = []

    for i in tqdm(range(evaluation_num)):
        data_instance = data_video[i]

        # AR two stage
        inputs = clip_input_video(processor, task, data_instance,frame_num = frame_num, model_type = model_type, data_path=data_path)
        if inputs == None:
            continue

        output_ar = AR_generate(inputs, model ,max_new_tokens=max_new_tokens, video_token_id=video_token_id, processor=processor)
        output_text = processor.batch_decode(output_ar['output_ids'], skip_special_tokens=True)[0]
        ar_record = _build_record(i, "ar", output_ar, output_text)
        print("\n-------Autoregressive-------")
        _print_record(ar_record)
        results["ar"]["decoding_times"].append(float(_to_python_scalar(output_ar["decoding_time"])))
        results["ar"]["generate_lens"].append(_normalize_generate_len(output_ar["generate_len"]))
        sample_records.append(ar_record)

        # SD tree two stage  
        inputs = clip_input_video(processor, task, data_instance,frame_num = frame_num, model_type = model_type, data_path=data_path)
        output_sd = SD_generate(
                inputs,
                model,
                draft_model,
                processor,
                max_new_tokens=max_new_tokens,
                video_token_id=video_token_id,
                tree_choices=mc_sim_7b_63,
        )
        output_text = processor.batch_decode(output_sd['output_ids'], skip_special_tokens=True)[0]
        sd_record = _build_record(i, "naive_sd", output_sd, output_text, include_accept_length=True)
        print("\n-------Naive SD-------")
        _print_record(sd_record)
        results["naive_sd"]["decoding_times"].append(float(_to_python_scalar(output_sd["decoding_time"])))
        results["naive_sd"]["generate_lens"].append(_normalize_generate_len(output_sd["generate_len"]))
        results["naive_sd"]["accept_lengths"].append(float(_to_python_scalar(output_sd["mean_accept_length"])))
        sample_records.append(sd_record)

        # SpecVLM
        inputs = clip_input_video(processor, task, data_instance,frame_num = frame_num, model_type = model_type, data_path=data_path)
        output_specvlm = SD_generate_with_pruning(
                inputs,
                model,
                draft_model,
                processor,
                method = 'specvlm',
                drop_rate = drop_rate,
                video_token_id = video_token_id,
                max_new_tokens=max_new_tokens,
                tree_choices=mc_sim_7b_63,
                percentage=0.4,
        )
        output_text = processor.batch_decode(output_specvlm['output_ids'], skip_special_tokens=True)[0]
        specvlm_record = _build_record(i, "specvlm", output_specvlm, output_text, include_accept_length=True)
        print("\n-------SpecVLM-------")
        _print_record(specvlm_record)
        results["specvlm"]["decoding_times"].append(float(_to_python_scalar(output_specvlm["decoding_time"])))
        results["specvlm"]["generate_lens"].append(_normalize_generate_len(output_specvlm["generate_len"]))
        results["specvlm"]["accept_lengths"].append(float(_to_python_scalar(output_specvlm["mean_accept_length"])))
        sample_records.append(specvlm_record)

    
    # Save results
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

        results_path = os.path.join(save_path, "results.jsonl")
        with open(results_path, 'w', encoding='utf-8') as f:
            for record in sample_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        metric_records = [
            _build_metric_record(
                "ar",
                results["ar"]["decoding_times"],
                results["ar"]["generate_lens"],
            ),
            _build_metric_record(
                "naive_sd",
                results["naive_sd"]["decoding_times"],
                results["naive_sd"]["generate_lens"],
                results["naive_sd"]["accept_lengths"],
            ),
            _build_metric_record(
                "specvlm",
                results["specvlm"]["decoding_times"],
                results["specvlm"]["generate_lens"],
                results["specvlm"]["accept_lengths"],
            ),
        ]

        print("\n-------metrics-------")
        for metric_record in metric_records:
            _print_record(metric_record)

        metrics_path = os.path.join(save_path, "metric.jsonl")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            for metric_record in metric_records:
                f.write(json.dumps(metric_record, ensure_ascii=False) + '\n')


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
                        help='Speculative Decoding setting') #TODO: For 'self' setting, load draft model as base model to save memory cost.

    
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
