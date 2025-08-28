import os
import numpy as np
import torch

from datasets import load_dataset, concatenate_datasets
import av

from transformers import AutoProcessor 
from models.modeling_llava_onevision_tree import LlavaOnevisionForConditionalGeneration

from models.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from models.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

# from visualize import *


def load_model(model_type, base_model_path, draft_model_path):
    if model_type == 'llava_ov':
        processor = AutoProcessor.from_pretrained(base_model_path, device_map="auto", torch_dtype=torch.float16)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            base_model_path, 
            device_map="auto", 
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        draft_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            draft_model_path, 
            device_map="auto", 
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        video_token_id = 151647
    elif model_type == 'qwen2_5_vl':
        processor = Qwen2_5_VLProcessor.from_pretrained(base_model_path, device_map="auto", torch_dtype=torch.float16)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path, 
            device_map="auto", 
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            attn_implementation = "sdpa",
        )
        draft_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            draft_model_path, 
            device_map="auto", 
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            attn_implementation = "sdpa",
        )
        video_token_id = 151656
    else:
        print("Not supported model type.")

    # video_token_id = model.config.video_token_id
    # print("video_token_id:",video_token_id)

    return model, draft_model, processor, video_token_id


def load_data(task, data_num, data_path):
    if task == "VideoDetailCaption":
        data_video = load_dataset(
                "/data/ycji/datasets/VideoDetailCaption",
                split="test",
                # cache_dir=cache_dir,
            ).shuffle(seed=42).select(range(data_num))
        
        def video_exists(example):
            video_path = os.path.join(video_dir, f"{example['video_name']}.mp4")
            return os.path.exists(video_path)

        video_dir = os.path.join(data_path, "Test_Videos")
        filtered_data = data_video.filter(video_exists)
        data_video = filtered_data
    elif task == 'MVBench':
        data_video_1 = load_dataset(
                "",
                'action_sequence',
                split="train",
                cache_dir=cache_dir,
            ).shuffle(seed=42).select(range(data_num))

        data_video_2 = load_dataset(
                "",
                'action_prediction',
                split="train",
                cache_dir=cache_dir,
            ).shuffle(seed=42).select(range(data_num))
        
        data_video = concatenate_datasets([data_video_1, data_video_2])
        data_video = data_video.shuffle(seed=42)

        def video_exists(example):
            video_path = os.path.join(video_dir, f"{example['video']}")
            return os.path.exists(video_path)
        
        video_dir = ""
        filtered_data = data_video.filter(video_exists)
        data_video = filtered_data
    elif task == 'MVLU':
        data_video = load_dataset(
                "",
                split="train",
                cache_dir=cache_dir,
            ).shuffle(seed=42).select(range(data_num))
    elif task == 'LongVideoBench':
        data_video = load_dataset(
                "",
                split="test",
                cache_dir=cache_dir,
            ).shuffle(seed=24).select(range(data_num))
    elif task == 'MMBench':
        data_video = load_dataset(
                "",
                split="train",
                cache_dir=cache_dir,
            ).shuffle(seed=42).select(range(data_num))
    elif task == 'COCO_caption':
        cache_dir = ''
        os.makedirs(cache_dir, exist_ok=True)
        data_video = load_dataset(
                "",
                split="test",
                cache_dir=cache_dir,
            ).shuffle(seed=42).select(range(100))
    else:
        data_video = None

    # print(data_video)
    return data_video



def read_video_pyav(container, indices=None):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)

    if indices is None:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
        print(f"INFO: {len(frames)} frames are decoded.")
        return np.stack(frames)
    else:
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)   
        print(f"INFO: {len(frames)} frames are decoded.")
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])
    


def convert_attention_to_score(attentions, input_ids, visual_token_id=151646, idx=None):
    # attentions: [num_layers, 1, num_heads, seq_len_q, seq_len_k]
    visual_token_mask = (input_ids == visual_token_id)
    visual_positions = torch.where(visual_token_mask[0])[0]
    n_image_tokens = len(visual_positions)
    print(f"INFO: Video Token Num: {n_image_tokens}")
    
    seq_len_k = attentions[0].shape[3]  
    device = attentions[0].device
    total_attention = torch.zeros(seq_len_k, device=device)
    
    num_layers = len(attentions)
    total_elements = num_layers
    
    for layer_attention in attentions:
        # layer_attention: [1, num_heads, seq_len_q, seq_len_k]
        layer_attention = layer_attention.mean(dim=1)  # [1, seq_len_q, seq_len_k]
        if idx != None:
            layer_attention = layer_attention[:,idx,:] #len(idx)>1
            layer_attention = layer_attention.mean(dim=1) 
        else:
            layer_attention = layer_attention.mean(dim=1)  # [1, seq_len_k]
        layer_attention = layer_attention.squeeze(0)  # [seq_len_k]
        total_attention += layer_attention
    
    avg_attention = total_attention / total_elements
    visual_attention = avg_attention[visual_positions]
    
    return visual_attention.tolist()





def drop_visual_tokens_random(inputs, drop_rate=0.5,visual_token_id=151647):
    visual_token_mask = (inputs['input_ids'] == visual_token_id) #LLaVA-OV:151647
    visual_positions = torch.where(visual_token_mask[0])[0]
    n_image_tokens = len(visual_positions)
    

    tokens_to_keep = int(n_image_tokens * (1 - drop_rate))
    
    perm = torch.randperm(n_image_tokens)
    keep_indices = perm[:tokens_to_keep]
    keep_indices, _ = torch.sort(keep_indices) #key
    keep_positions = visual_positions[keep_indices]
    
    non_visual_mask = ~visual_token_mask[0]
    final_mask = non_visual_mask.clone()
    final_mask[keep_positions] = True
    
    new_input_ids = inputs['input_ids'][:, final_mask]
    
    new_inputs = inputs
    new_inputs['input_ids'] = new_input_ids
    new_inputs['selected_indices'] = keep_indices
    new_inputs['attention_mask'] = None
    # print(keep_indices)
    
    return new_inputs


def drop_visual_tokens_window(inputs, drop_rate=0.5, visual_token_id=151647, drop_mode='front'):
    visual_token_mask = (inputs['input_ids'] == visual_token_id)
    visual_positions = torch.where(visual_token_mask[0])[0]
    n_image_tokens = len(visual_positions)
    
    tokens_to_keep = int(n_image_tokens * (1 - drop_rate))
    
    if drop_mode == 'front':
        keep_indices = torch.arange(tokens_to_keep)
    elif drop_mode == 'middle':
        start_idx = (n_image_tokens - tokens_to_keep) // 2
        keep_indices = torch.arange(start_idx, start_idx + tokens_to_keep)
    elif drop_mode == 'back':
        keep_indices = torch.arange(n_image_tokens - tokens_to_keep, n_image_tokens)
    else:
        raise ValueError("drop_mode must be one of: 'front', 'middle', 'back'")
    
    keep_positions = visual_positions[keep_indices]
    
    non_visual_mask = ~visual_token_mask[0]
    final_mask = non_visual_mask.clone()
    final_mask[keep_positions] = True
    
    new_input_ids = inputs['input_ids'][:, final_mask]
    
    new_inputs = inputs.copy()
    new_inputs['input_ids'] = new_input_ids
    new_inputs['selected_indices'] = keep_indices
    new_inputs['attention_mask'] = None
    
    return new_inputs



def drop_visual_tokens_uniform(inputs, drop_rate=0.5, visual_token_id=151647):
    visual_token_mask = (inputs['input_ids'] == visual_token_id)
    visual_positions = torch.where(visual_token_mask[0])[0]
    n_image_tokens = len(visual_positions)
    
    tokens_to_keep = int(n_image_tokens * (1 - drop_rate))
    
    # Calculate stride for uniform sampling
    if tokens_to_keep == 0:
        stride = n_image_tokens
    else:
        stride = n_image_tokens / tokens_to_keep
    # Generate indices at regular intervals
    keep_indices = torch.linspace(0, n_image_tokens-1, tokens_to_keep, dtype=torch.long)
    
    keep_positions = visual_positions[keep_indices]
    
    non_visual_mask = ~visual_token_mask[0]
    final_mask = non_visual_mask.clone()
    final_mask[keep_positions] = True
    
    new_input_ids = inputs['input_ids'][:, final_mask]
    
    new_inputs = inputs
    new_inputs['input_ids'] = new_input_ids
    new_inputs['selected_indices'] = keep_indices
    new_inputs['attention_mask'] = None
    
    return new_inputs




def drop_visual_tokens_by_attention(attentions, inputs, drop_rate=0.5, visual_token_id=151647,output_scores=False, reverse=False, idx=None):
    scores = convert_attention_to_score(attentions, inputs['input_ids'], visual_token_id, idx)
    
    visual_token_mask = (inputs['input_ids'] == visual_token_id)
    visual_positions = torch.where(visual_token_mask[0])[0]
    n_image_tokens = len(visual_positions)
    
    tokens_to_keep = int(n_image_tokens * (1 - drop_rate))
    
    scores_tensor = torch.tensor(scores)
    if reverse == True:
        _ , indices = torch.sort(scores_tensor, descending=False)
    else:
        _, indices = torch.sort(scores_tensor, descending=True)
    keep_indices = indices[:tokens_to_keep]
    keep_indices, _ = torch.sort(keep_indices)
    keep_positions = visual_positions[keep_indices]
    
    non_visual_mask = ~visual_token_mask[0]
    final_mask = non_visual_mask.clone()
    final_mask[keep_positions] = True
    
    new_input_ids = inputs['input_ids'][:, final_mask]
    
    new_inputs = inputs
    new_inputs['input_ids'] = new_input_ids
    new_inputs['selected_indices'] = keep_indices
    new_inputs['attention_mask'] = None
    
    if output_scores:
        return new_inputs, scores
    return new_inputs



def drop_visual_tokens_specvlm(attentions, inputs, drop_rate=0.5, visual_token_id=151647, output_scores=False, reverse=False, idx=None,threshold=None, percentage=None):
    scores = convert_attention_to_score(attentions, inputs['input_ids'], visual_token_id, idx)
    # print_scores_bar(scores)
    
    visual_token_mask = (inputs['input_ids'] == visual_token_id)
    visual_positions = torch.where(visual_token_mask[0])[0]
    n_image_tokens = len(visual_positions)
    
    # Calculate total tokens to keep
    tokens_to_keep = int(n_image_tokens * (1 - drop_rate))

    # Split between attention and uniform
    if threshold != None and percentage != None:
        print("Can't use threshold and percentage at the same time.")
    elif threshold != None:
        if (1 - drop_rate) < threshold:
            attention_tokens = tokens_to_keep
            uniform_tokens = 0
        else:
            attention_tokens = int(n_image_tokens * threshold)
            uniform_tokens = tokens_to_keep - attention_tokens
    elif percentage != None:
        attention_tokens = get_attention_token_from_percentage(scores, percentage)
        if attention_tokens > tokens_to_keep:
            uniform_tokens = 0
            attention_tokens = tokens_to_keep
        else:
            uniform_tokens = tokens_to_keep - attention_tokens
    # print("attention tokens:",attention_tokens)
        
    # Get attention-based indices
    scores_tensor = torch.tensor(scores)
    if reverse:
        _, attention_sorted = torch.sort(scores_tensor, descending=False)
    else:
        _, attention_sorted = torch.sort(scores_tensor, descending=True)
    attention_indices = attention_sorted[:attention_tokens]
    
    # Create mask of available positions for uniform sampling
    available_mask = torch.ones(n_image_tokens, dtype=torch.bool)
    available_mask[attention_indices] = False
    available_positions = torch.where(available_mask)[0]
    
    # Uniform sample from remaining positions
    if uniform_tokens != 0:
        stride = len(available_positions) / uniform_tokens
    else:
        stride = len(available_positions)
    if len(available_positions)-1 >=0 :
        uniform_positions = torch.linspace(0, len(available_positions)-1, uniform_tokens, dtype=torch.long)
        uniform_indices = available_positions[uniform_positions]
    else:
        uniform_indices=None
    
    # Combine and sort indices
    keep_indices = torch.cat([attention_indices, uniform_indices])
    keep_indices, _ = torch.sort(keep_indices)
    
    # Apply mask
    keep_positions = visual_positions[keep_indices]
    non_visual_mask = ~visual_token_mask[0]
    final_mask = non_visual_mask.clone()
    final_mask[keep_positions] = True
    
    new_input_ids = inputs['input_ids'][:, final_mask]
    
    new_inputs = inputs
    new_inputs['input_ids'] = new_input_ids
    new_inputs['selected_indices'] = keep_indices
    new_inputs['attention_mask'] = None
    
    if output_scores:
        return new_inputs, keep_indices
    return new_inputs

    


def get_last_video_idx(input_ids, video_token_id):
    #reverse order 
    last_video_idx = -1
    for i in range(len(input_ids)-1, -1, -1):
        if input_ids[i] == video_token_id:
            last_video_idx = i
            break
    return last_video_idx


def get_idx_from_attention(attentions, inputs, video_token_id):
    # Get attention scores for each visual token
    scores = convert_attention_to_score(attentions, inputs['input_ids'], video_token_id)
    
    # Convert scores list to tensor
    scores_tensor = torch.tensor(scores)
    
    # Calculate number of tokens to keep (50%)
    n_tokens = len(scores)
    tokens_to_keep = n_tokens // 2
    
    # Get indices of top 50% attention scores
    _, indices = torch.sort(scores_tensor, descending=True)
    top_indices = indices[:tokens_to_keep]
    
    # Sort indices to maintain original order
    top_indices, _ = torch.sort(top_indices)
    
    return top_indices



def get_idx_from_similarity(video_emb, text_emb, idx=None):
    # Remove batch dimension if present
    video_emb = video_emb.squeeze(0)  # [v_len, 3584]
    if idx != None:
        video_emb = video_emb[idx,:]
    text_emb = text_emb.squeeze(0)    # [t_len, 3584]
    
    # Calculate similarity
    similarity = torch.matmul(text_emb, video_emb.t())  # [t_len, v_len]
    
    # Get average similarity scores
    avg_scores = similarity.mean(dim=1)  # [t_len]
    
    # Get indices of top 50% scores
    k = len(avg_scores) // 2
    _, indices = torch.topk(avg_scores, k)
    
    # Sort indices to maintain original order
    indices = torch.sort(indices)[0]
    
    return indices



def calculate_attention_percentage(scores, threshold):
    # Convert to tensor if not already
    if not isinstance(scores, torch.Tensor):
        scores_tensor = torch.tensor(scores)
    else:
        scores_tensor = scores
    
    # Sort scores in descending order
    sorted_scores, _ = torch.sort(scores_tensor, descending=True)
    
    # Calculate number of tokens to consider for top percentage
    n_tokens = len(sorted_scores)
    top_k = max(1, int(n_tokens * threshold))  # At least 1 token
    
    # Sum of top scores
    top_sum = sorted_scores[:top_k].sum()
    
    # Total sum of all scores
    total_sum = scores_tensor.sum()
    
    # Calculate percentage
    if total_sum > 0:  # Avoid division by zero
        percentage = (top_sum / total_sum) * 100.0
    else:
        percentage = 0.0
    # print("Attention Percentage:",percentage)
    return percentage.item() if hasattr(percentage, 'item') else percentage


def get_attention_token_from_percentage(scores, threshold):
    # Convert to tensor if not already
    if not isinstance(scores, torch.Tensor):
        scores_tensor = torch.tensor(scores)
    else:
        scores_tensor = scores
    
    # Sort scores in descending order
    sorted_scores, _ = torch.sort(scores_tensor, descending=True)
    
    # Calculate total sum of all scores
    total_sum = scores_tensor.sum()
    
    # Target sum to reach
    target_sum = total_sum * threshold
    
    # Initialize variables
    current_sum = 0.0
    token_count = 0
    
    # Add tokens until reaching or exceeding the threshold
    for score in sorted_scores:
        current_sum += score
        token_count += 1
        
        if current_sum >= target_sum:
            break
    
    return token_count



def clip_input(processor, data_instance):
    image = data_instance["image"]
    conversation = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "Please provide a detailed description of the image, focusing on the main subjects, their actions, and the background scenes."},
          {"type": "image"},
        ],
    },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
    print("input id length:", inputs['input_ids'].shape[1])
    return inputs


def clip_input_video(processor, task, data_instance, frame_num=64, model_type='llava_ov',data_path=None):
    if model_type == 'llava_ov':
        if task == "VideoDetailCaption":
            video_path = os.path.join(data_path, "Test_Videos/")
            video_name = data_instance["video_name"]
            video_path = video_path + video_name + ".mp4"

            question = data_instance["question"]
            conversation = [
            {

                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": question},
                    ],
            },
            ]
            container = av.open(video_path)
            total_frames = container.streams.video[0].frames
            # print("Total frames:",total_frames)
            indices = np.arange(0, total_frames, total_frames / frame_num).astype(int)
            video = read_video_pyav(container, indices)
            
        elif task == "MVBench":
            video_path = data_path
            video_name = data_instance["video"]
            video_path = video_path + video_name

            question = "Please provide a detailed description of the video, focusing on the main subjects, their actions, and the background scenes."
            conversation = [
            {

                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": question},
                    ],
            },
            ]

            container = av.open(video_path)
            total_frames = container.streams.video[0].frames
            # print("Total frames:",total_frames)
            indices = np.arange(0, total_frames, total_frames / frame_num).astype(int)
            video = read_video_pyav(container, indices)

        elif task == 'LongVideoBench':
            video_path = data_path
            video_name = data_instance["video_path"]
            video_path = video_path + video_name

            question = "Please provide a detailed description of the video, focusing on the main subjects, their actions, and the background scenes."
            conversation = [
            {

                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": question},
                    ],
            },
            ]

            container = av.open(video_path)
            total_frames = container.streams.video[0].frames
            # print("Total frames:",total_frames)
            if total_frames == 0:
                return None
            indices = np.arange(0, total_frames, total_frames / frame_num).astype(int)
            video = read_video_pyav(container, indices)

        elif task == "MVLU":
            video_reader = data_instance['video']

            total_frames = len(video_reader)
            # print("Total frames:", total_frames)
            indices = np.linspace(0, total_frames - 1, frame_num, dtype=int)
            video = video_reader.get_batch(indices).asnumpy()

            question = "Please provide a detailed description of the video, focusing on the main subjects, their actions, and the background scenes."
            conversation = [
            {

                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": question},
                    ],
            },
            ]

        # display_frame_grid(video)
        # save_frames(video)
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(videos=list(video), text=prompt, return_tensors="pt").to("cuda")

    elif model_type == 'qwen2_5_vl':
        def calculate_fps_for_target_frames(container, target_frames):
            video_stream = container.streams.video[0]
            duration = container.duration / 1000000
            if duration <= 0:
                return 1.0 
            
            required_fps = target_frames / duration
            print(f"INFO: Duration: {duration:.2f}s, frame_num: {target_frames}, fps: {required_fps:.2f}")
            return required_fps

        if task == "VideoDetailCaption":
            video_path = os.path.join(data_path, "Test_Videos/")
            video_name = data_instance["video_name"]
            video_path = video_path + video_name + ".mp4"
            question = data_instance["question"]
        
        elif task == "MVBench":
            video_path = data_path
            video_name = data_instance["video"]
            video_path = video_path + video_name
            question = "Please provide a detailed description of the video, focusing on the main subjects, their actions, and the background scenes."
            
        elif task == 'LongVideoBench':
            video_path = data_path
            video_name = data_instance["video_path"]
            video_path = video_path + video_name
            question = "Please provide a detailed description of the video, focusing on the main subjects, their actions, and the background scenes."
            
        # elif task == "MVLU":
        #     video_reader = data_instance['video']
        #     total_frames = len(video_reader)
        #     print("Total frames:", total_frames)

        #     indices = np.linspace(0, total_frames - 1, frame_num, dtype=int)
        #     frames = video_reader.get_batch(indices).asnumpy()
        
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        # print("Total frames:", total_frames)
        
        if total_frames == 0:
            return None

        fps = calculate_fps_for_target_frames(container, frame_num)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{video_path}",
                        "max_pixels": 448*448,  
                        "fps": fps, 
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda")
    
    print("INFO: Input length:", inputs['input_ids'].shape[1])
    return inputs
