import os
import numpy as np
import torch

import matplotlib.pyplot as plt

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
    # cache_dir = '/data/ycji/cache/VideoDetailCaption'
    # os.makedirs(cache_dir, exist_ok=True)

    if task == "VideoDetailCaption":
        data_video = load_dataset(
                "/data/ycji/datasets/VideoDetailCaption",
                split="test",
                # cache_dir=cache_dir,
            ).shuffle(seed=42).select(range(data_num))
        
        def video_exists(example):
            video_path = os.path.join(video_dir, f"{example['video_name']}.mp4")
            return os.path.exists(video_path)

        # video_dir = "/data/ycji/datasets/VideoDetailCaption/Test_Videos"
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



def drop_visual_tokens_by_scores(scores, inputs, drop_rate=0.5, visual_token_id=151647):
    visual_token_mask = (inputs['input_ids'] == visual_token_id)
    visual_positions = torch.where(visual_token_mask[0])[0]
    n_image_tokens = len(visual_positions)
    
    tokens_to_keep = int(n_image_tokens * (1 - drop_rate))
    
    scores_tensor = torch.tensor(scores)
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



def drop_visual_tokens_attention_plus_uniform(attentions, inputs, drop_rate=0.5, visual_token_id=151647, output_scores=False, reverse=False, idx=None,threshold=None, percentage=None):
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
        if drop_rate > 0.75:
            percentage = 0.4
        else:
            percentage = 0.5
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





def drop_visual_tokens_frame(inputs, drop_rate, visual_token_id, tokens_per_frame=196):

    # Identify visual token positions in the input
    visual_token_mask = (inputs['input_ids'] == visual_token_id)
    visual_positions = torch.where(visual_token_mask[0])[0]
    n_image_tokens = len(visual_positions)
    
    # Calculate number of frames
    n_frames = n_image_tokens // tokens_per_frame
    
    # Calculate frames to keep based on (1-drop_rate)
    frames_to_keep = int(n_frames * (1 - drop_rate))
    frames_to_keep = max(1, frames_to_keep)  # Keep at least 1 frame
    
    # Uniformly select frames to keep
    frame_indices = torch.linspace(0, n_frames-1, frames_to_keep, dtype=torch.long)
    
    # Calculate tokens to keep per frame to achieve target total
    total_tokens_to_keep = int(n_image_tokens * (1 - drop_rate))
    
    # Collect indices of tokens to keep
    keep_indices = []
    for frame_idx in frame_indices:
        frame_start = frame_idx.item() * tokens_per_frame
        
        # Add to keep list
        keep_indices.extend([frame_start + idx for idx in range(tokens_per_frame)])
    
    # Ensure indices are within valid range
    keep_indices = [idx for idx in keep_indices if idx < n_image_tokens]
    
    # Sort indices
    keep_indices.sort()
    keep_indices = torch.tensor(keep_indices,dtype=torch.long)
    
    # Get positions to keep in the input
    keep_positions = visual_positions[keep_indices]
    
    # Create final mask for all tokens
    non_visual_mask = ~visual_token_mask[0]
    final_mask = non_visual_mask.clone()
    final_mask[keep_positions] = True
    
    # Create new inputs with filtered tokens
    new_input_ids = inputs['input_ids'][:, final_mask]
    
    # Update inputs dictionary
    new_inputs = inputs.copy()
    new_inputs['input_ids'] = new_input_ids
    new_inputs['selected_indices'] = keep_indices
    new_inputs['attention_mask'] = None
    
    return new_inputs




# def drop_visual_tokens_frame_plus_uniform(inputs, drop_rate, visual_token_id, tokens_per_frame=196):

#     # Identify visual token positions in the input
#     visual_token_mask = (inputs['input_ids'] == visual_token_id)
#     visual_positions = torch.where(visual_token_mask[0])[0]
#     n_image_tokens = len(visual_positions)
    
#     # Calculate number of frames
#     n_frames = n_image_tokens // tokens_per_frame
    
#     # Calculate frames to keep based on sqrt(1-drop_rate)
#     frames_to_keep = int(n_frames * (1 - drop_rate) ** 0.5)
#     frames_to_keep = max(1, frames_to_keep)  # Keep at least 1 frame
    
#     # Uniformly select frames to keep
#     frame_indices = torch.linspace(0, n_frames-1, frames_to_keep, dtype=torch.long)
    
#     # Calculate tokens to keep per frame to achieve target total
#     total_tokens_to_keep = int(n_image_tokens * (1 - drop_rate))
#     tokens_per_kept_frame = total_tokens_to_keep // frames_to_keep
    
#     # Collect indices of tokens to keep
#     keep_indices = []
#     for frame_idx in frame_indices:
#         frame_start = frame_idx.item() * tokens_per_frame
        
#         # Uniformly select tokens within the frame
#         token_indices = torch.linspace(0, tokens_per_frame-1, tokens_per_kept_frame, dtype=torch.long)
        
#         # Add to keep list
#         keep_indices.extend([frame_start + idx.item() for idx in token_indices])
    
#     # Ensure indices are within valid range
#     keep_indices = [idx for idx in keep_indices if idx < n_image_tokens]
    
#     # Sort indices
#     keep_indices.sort()
#     keep_indices = torch.tensor(keep_indices,dtype=torch.long)
    
#     # Get positions to keep in the input
#     keep_positions = visual_positions[keep_indices]
    
#     # Create final mask for all tokens
#     non_visual_mask = ~visual_token_mask[0]
#     final_mask = non_visual_mask.clone()
#     final_mask[keep_positions] = True
    
#     # Create new inputs with filtered tokens
#     new_input_ids = inputs['input_ids'][:, final_mask]
    
#     # Update inputs dictionary
#     new_inputs = inputs.copy()
#     new_inputs['input_ids'] = new_input_ids
#     new_inputs['selected_indices'] = keep_indices
#     new_inputs['attention_mask'] = None
    
#     return new_inputs




#Dycoke
def drop_visual_tokens_temporal(video_features, inputs, drop_rate=0.5, visual_token_id=151647, output_scores=False):
    # Get indices of tokens to keep based on temporal similarity
    video_indices = get_idx_from_video_features(video_features, drop_rate)
    
    # Identify visual token positions in the input
    visual_token_mask = (inputs['input_ids'] == visual_token_id)
    visual_positions = torch.where(visual_token_mask[0])[0]
    n_image_tokens = len(visual_positions)
    
    if len(video_indices) > n_image_tokens:
        video_indices = video_indices[:n_image_tokens]
    
    # Get the positions to keep in the input
    keep_indices = video_indices 
    keep_positions = visual_positions[keep_indices]
    
    # Create final mask for all tokens
    non_visual_mask = ~visual_token_mask[0]
    final_mask = non_visual_mask.clone()
    final_mask[keep_positions] = True
    
    # Create new inputs with filtered tokens
    new_input_ids = inputs['input_ids'][:, final_mask]
    
    # Update inputs dictionary
    new_inputs = inputs.copy()
    new_inputs['input_ids'] = new_input_ids
    new_inputs['selected_indices'] = keep_indices
    new_inputs['attention_mask'] = None
    
    return new_inputs




def get_idx_from_video_features(video_features, drop_rate, tokens_per_frame=196):
    # Check if number of tokens matches frames
    n_total_tokens = video_features.shape[0] - 1  # exclude last token
    assert n_total_tokens % tokens_per_frame == 0, "Total tokens must be divisible by tokens per frame"
    
    n_frames = n_total_tokens // tokens_per_frame
    
    # Normalize features for cosine similarity
    normalized_features = torch.nn.functional.normalize(video_features[:-1], dim=1)  # exclude last token
    
    keep_indices = []
    
    # Process frames in groups of 4
    for group_start in range(0, n_frames - n_frames % 4, 4):
        # Get indices for the 4 frames in current group
        frame1_start = group_start * tokens_per_frame
        frame2_start = (group_start + 1) * tokens_per_frame
        frame3_start = (group_start + 2) * tokens_per_frame
        frame4_start = (group_start + 3) * tokens_per_frame
        
        # Keep all tokens from first frame
        keep_indices.extend(range(frame1_start, frame1_start + tokens_per_frame))
        
        # Calculate number of tokens to keep for other frames
        tokens_to_keep = int(tokens_per_frame * (1 - drop_rate) * 0.75)  # 3/4 of (1-drop_rate)
        
        # Calculate cosine similarities of corresponding token positions
        # Frame 2 to Frame 1 (corresponding positions)
        sim_2_to_1 = torch.sum(
            normalized_features[frame2_start:frame2_start + tokens_per_frame] * 
            normalized_features[frame1_start:frame1_start + tokens_per_frame],
            dim=1
        )
        
        # Frame 4 to Frame 3 (corresponding positions)
        sim_4_to_3 = torch.sum(
            normalized_features[frame4_start:frame4_start + tokens_per_frame] * 
            normalized_features[frame3_start:frame3_start + tokens_per_frame],
            dim=1
        )
        
        # Frame 3 to Frame 1 (corresponding positions)
        sim_3_to_1 = torch.sum(
            normalized_features[frame3_start:frame3_start + tokens_per_frame] * 
            normalized_features[frame1_start:frame1_start + tokens_per_frame],
            dim=1
        )
        
        # Select tokens with lowest similarities for each frame
        # Frame 2
        _, frame2_indices = torch.topk(sim_2_to_1, tokens_to_keep, largest=False)
        keep_indices.extend([frame2_start + idx for idx in frame2_indices.tolist()])
        
        # Frame 3
        _, frame3_indices = torch.topk(sim_3_to_1, tokens_to_keep, largest=False)
        keep_indices.extend([frame3_start + idx for idx in frame3_indices.tolist()])
        
        # Frame 4
        _, frame4_indices = torch.topk(sim_4_to_3, tokens_to_keep, largest=False)
        keep_indices.extend([frame4_start + idx for idx in frame4_indices.tolist()])
    
    # Sort indices
    keep_indices.sort()
    
    return torch.tensor(keep_indices)




def drop_visual_tokens_frame_similarity(video_features, inputs, drop_rate, visual_token_id=151647):
    # Get indices of tokens to keep based on frame similarity
    video_indices = get_idx_from_frame_features(video_features, drop_rate)
    
    # Identify visual token positions in the input
    visual_token_mask = (inputs['input_ids'] == visual_token_id)
    visual_positions = torch.where(visual_token_mask[0])[0]
    n_image_tokens = len(visual_positions)
    
    if len(video_indices) > n_image_tokens:
        video_indices = video_indices[:n_image_tokens]
    
    # Get the positions to keep in the input
    keep_indices = video_indices 
    keep_positions = visual_positions[keep_indices]
    
    # Create final mask for all tokens
    non_visual_mask = ~visual_token_mask[0]
    final_mask = non_visual_mask.clone()
    final_mask[keep_positions] = True
    
    # Create new inputs with filtered tokens
    new_input_ids = inputs['input_ids'][:, final_mask]
    
    # Update inputs dictionary
    new_inputs = inputs.copy()
    new_inputs['input_ids'] = new_input_ids
    new_inputs['selected_indices'] = keep_indices
    new_inputs['attention_mask'] = None
    
    return new_inputs


def get_idx_from_frame_features(video_features, drop_rate, tokens_per_frame=196):

    n_total_tokens = video_features.shape[0] - 1 
    assert n_total_tokens % tokens_per_frame == 0, "Total tokens must be divisible by tokens_per_frame"
    
    n_frames = n_total_tokens // tokens_per_frame
    
    normalized_features = torch.nn.functional.normalize(video_features[:-1], dim=1)
    
    frame_similarities = []
    for i in range(n_frames - 1):
        frame1_start = i * tokens_per_frame
        frame2_start = (i + 1) * tokens_per_frame
        
        token_similarities = torch.sum(
            normalized_features[frame1_start:frame1_start + tokens_per_frame] * 
            normalized_features[frame2_start:frame2_start + tokens_per_frame],
            dim=1
        )
        
        avg_frame_similarity = torch.mean(token_similarities).item()
        frame_similarities.append((i, i+1, avg_frame_similarity))
    
    frame_similarities.sort(key=lambda x: x[2], reverse=True)
    
    frames_to_drop = int(n_frames * drop_rate)
    
    frames_to_remove = set()
    for _, frame2_idx, _ in frame_similarities[:frames_to_drop]:
        frames_to_remove.add(frame2_idx)
    
    keep_indices = []
    for frame_idx in range(n_frames):
        if frame_idx not in frames_to_remove:
            frame_start = frame_idx * tokens_per_frame
            keep_indices.extend(range(frame_start, frame_start + tokens_per_frame))

    keep_indices.sort()
    
    return torch.tensor(keep_indices)
    


def drop_visual_tokens_frame_attention(attentions, video_features, inputs, drop_rate,
                                    visual_token_id=151647):
    scores = convert_attention_to_score(attentions, inputs['input_ids'], visual_token_id) #List of length n_image_tokens

    # Get indices of tokens to keep based on temporal similarity and attention
    video_indices = get_idx_from_frame_features_and_attentions(scores, video_features, drop_rate)
    
    # Identify visual token positions in the input
    visual_token_mask = (inputs['input_ids'] == visual_token_id)
    visual_positions = torch.where(visual_token_mask[0])[0]
    n_image_tokens = len(visual_positions)
    
    if len(video_indices) > n_image_tokens:
        video_indices = video_indices[:n_image_tokens]
    
    # Get the positions to keep in the input
    keep_indices = video_indices 
    keep_positions = visual_positions[keep_indices]
    
    # Create final mask for all tokens
    non_visual_mask = ~visual_token_mask[0]
    final_mask = non_visual_mask.clone()
    final_mask[keep_positions] = True
    
    # Create new inputs with filtered tokens
    new_input_ids = inputs['input_ids'][:, final_mask]
    
    # Update inputs dictionary
    new_inputs = inputs.copy()
    new_inputs['input_ids'] = new_input_ids
    new_inputs['selected_indices'] = keep_indices
    new_inputs['attention_mask'] = None
    
    return new_inputs


def get_idx_from_frame_features_and_attentions(scores, video_features, drop_rate, tokens_per_frame=196):
    # Check if token count matches frame structure
    n_total_tokens = video_features.shape[0] - 1  # exclude last token
    assert n_total_tokens % tokens_per_frame == 0, "Total tokens must be divisible by tokens_per_frame"
    
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores)
    
    # Calculate number of frames and frames to keep
    n_frames = n_total_tokens // tokens_per_frame
    frames_to_keep = int(n_frames * (1 - drop_rate))
    
    # Calculate total score for each frame
    frame_scores = []
    for i in range(n_frames):
        frame_start = i * tokens_per_frame
        frame_end = frame_start + tokens_per_frame
        
        frame_score = torch.sum(scores[frame_start:frame_end]).item()
        frame_scores.append((i, frame_score))

    frame_scores.sort(key=lambda x: x[1], reverse=True)
    
    frames_to_keep = [frame_idx for frame_idx, _ in frame_scores[:frames_to_keep]]
    frames_to_keep.sort()  # Sort to maintain original order
    
    keep_indices = []
    for frame_idx in frames_to_keep:
        frame_start = frame_idx * tokens_per_frame
        keep_indices.extend(range(frame_start, frame_start + tokens_per_frame))
    
    return torch.tensor(keep_indices)






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
    print("Attention Percentage:",percentage)
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
            # video_path = "/data/ycji/datasets/VideoDetailCaption/Test_Videos/"
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

            # question = "Please answer the following question. After that, provide a detailed description of the video to support the answer, focusing on the main subjects, their actions, and the background scenes"
            # question = question + data_instance["question"]
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
            print(f"Duration: {duration:.2f}s, frame_num: {target_frames}, fps: {required_fps:.2f}")
            return required_fps

        if task == "VideoDetailCaption":
            video_path = ""
            video_name = data_instance["video_name"]
            video_path = video_path + video_name + ".mp4"
            question = data_instance["question"]
        
        elif task == "MVBench":
            video_path = ""
            video_name = data_instance["video"]
            video_path = video_path + video_name
            question = "Please provide a detailed description of the video, focusing on the main subjects, their actions, and the background scenes."
            
        elif task == 'LongVideoBench':
            video_path = ""
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







def clip_input_video_output(processor, task, data_instance, frame_num=64,model_type=None):
    if task == "VideoDetailCaption":
        video_path = ""
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
    elif task == "MVBench":
        video_path = ""
        video_name = data_instance["video"]
        video_path = video_path + video_name

        # question = "Please answer the following question. After that, provide a detailed description of the video to support the answer, focusing on the main subjects, their actions, and the background scenes"
        # question = question + data_instance["question"]
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

    # display_frame_grid(video)

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(videos=list(video), text=prompt, return_tensors="pt").to("cuda")
    
    print("input id length:", inputs['input_ids'].shape[1])
    return inputs, video