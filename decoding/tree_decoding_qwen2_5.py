# Codebase: https://github.com/SafeAILab/EAGLE

import copy
import json
import time

import torch
import os
from torch import nn

from kv_cache.kv_cache import initialize_past_key_values
from utils.utils_c import generate_tree_buffers_draft
from tree_choices.choices import mc_sim_7b_63
from utils.utils import *


DEPTH = 5
TOPK = 10
top_k = 10
DRAFT_TOTAL_TOKENS = 48


def reset_tree_mode(
        model,
):
    """
    Resets the tree settings to their initial state.

    This function ensures that after any operations involving,
    the model's the tree attention mask return to their default state.
    """
    model.model.tree_mask = None



def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))




def generate_tree_buffers(tree_choices, device="cuda"):
    sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
    tree_len = len(sorted_tree_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
    depth_counts = []
    prev_depth = 0
    for path in sorted_tree_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth

    tree_attn_mask = torch.eye(tree_len, tree_len)
    tree_attn_mask[:, 0] = 1
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            # retrieve ancestor position
            if len(cur_tree_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_tree_choice) - 1):
                ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
            tree_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]

    tree_indices = torch.zeros(tree_len, dtype=torch.long)
    p_indices = [0 for _ in range(tree_len - 1)]
    b_indices = [[] for _ in range(tree_len - 1)]
    tree_indices[0] = 0
    start = 0
    bias = 0
    for i in range(len(depth_counts)):
        inlayer_bias = 0
        b = []
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            cur_parent = cur_tree_choice[:-1]
            if j != 0:
                if cur_parent != parent:
                    bias += 1
                    inlayer_bias += 1
                    parent = cur_parent
                    b = []
            else:
                parent = cur_parent
            tree_indices[start + j + 1] = cur_tree_choice[-1] + TOPK * (i + bias) + 1
            p_indices[start + j] = inlayer_bias
            if len(b) > 0:
                b_indices[start + j] = copy.deepcopy(b)
            else:
                b_indices[start + j] = []
            b.append(cur_tree_choice[-1] + TOPK * (i + bias) + 1)
        start += depth_counts[i]

    p_indices = [-1] + p_indices
    tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
    start = 0
    for i in range(len(depth_counts)):
        tree_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_tree_choices)):
        cur_tree_choice = sorted_tree_choices[-i - 1]
        retrieve_indice = []
        if cur_tree_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_tree_choice)):
                retrieve_indice.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]))
                retrieve_paths.append(cur_tree_choice[:c + 1])
        retrieve_indices_nest.append(retrieve_indice)
    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
                                 dim=1)
    maxitem = retrieve_indices.max().item() + 5


    def custom_sort(lst):
        # sort_keys=[len(list)]
        sort_keys = []
        for i in range(len(lst)):
            sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
        return sort_keys

    retrieve_indices = retrieve_indices.tolist()
    retrieve_indices = sorted(retrieve_indices, key=custom_sort)
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)

    p_indices = torch.tensor(p_indices)
    p_indices_new = p_indices[retrieve_indices]
    p_indices_new = p_indices_new.tolist()

    b_indices = [[]] + b_indices
    b_indices_new = []
    for ib in range(retrieve_indices.shape[0]):
        iblist = []
        for jb in range(retrieve_indices.shape[1]):
            index = retrieve_indices[ib, jb]
            if index == -1:
                iblist.append([])
            else:
                b = b_indices[index]
                if len(b) > 0:
                    bt = []
                    for bi in b:
                        bt.append(torch.where(tree_indices == bi)[0].item())
                    iblist.append(torch.tensor(bt, device=device))
                else:
                    iblist.append(b)
        b_indices_new.append(iblist)

    # Aggregate the generated buffers into a dictionary
    tree_buffers = {
        "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": tree_indices,
        "tree_position_ids": tree_position_ids,
        "retrieve_indices": retrieve_indices,
    }

    # Move the tensors in the dictionary to the specified device
    tree_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v, device=device)
        for k, v in tree_buffers.items()
    }
    tree_buffers["p_indices"] = p_indices_new
    tree_buffers["b_indices"] = b_indices_new
    return tree_buffers



@torch.no_grad()
def tree_draft(input_ids, draft_model, draft_past_key_values,len_posi):
    # input_ids = input_ids[:, 1:]
    input_ids = input_ids.to(draft_model.device)
    ss_token,ss_prob,ss_op = [],[],[]
    # len_posi = total_input_len + sample_token
    draft_model.model.tree_mask=None

    #old kv cache and new token(s)
    outputs = draft_model(input_ids=input_ids, past_key_values=draft_past_key_values)
    last_headout = outputs.logits[0, -1:] #[input_len, vocab_size]

    for i in range(len(draft_model.tree_buffer['tree_indices'])):
        top=torch.topk(last_headout, top_k, dim=-1)
        topk_index,topk_prob = top.indices,top.values #[1,input_len,10]
        op=None
        ss_token.append(topk_index)
        # ss_prob.append(topk_prob)
        # ss_op.append(op)

        #flatten
        topk_index = topk_index.view(-1) #[input_len * 10]
        #Choose next input_ids
        select_index=topk_index[draft_model.tree_buffer['tree_indices'][i].to(topk_index.device)]
        input_ids=select_index[None,:] #[1,4]/[1,1]
        #Prepare next position_ids and attn_mask
        draft_model.model.tree_mask=draft_model.tree_buffer['attn_mask'][i]
        position_ids=len_posi+draft_model.tree_buffer["position_ids"][i]
        # outputs, past_key_values = draft_model(input_ids=input_ids, past_key_values=past_key_values,
        #                                     position_ids=position_ids,use_cache=True)
        outputs = draft_model(input_ids=input_ids, past_key_values=draft_past_key_values,position_ids=position_ids)
        len_posi += 1
        last_headout = outputs.logits[0] #[len_input, vocab_size]

    top = torch.topk(last_headout, top_k, dim=-1)
    topk_index, topk_prob = top.indices, top.values
    op=None

    ss_token.append(topk_index)
    ss_prob.append(topk_prob)
    ss_op.append(op)

    # return (torch.cat(ss_token),torch.cat(ss_prob),ss_op)
    return torch.cat(ss_token).view(-1)




def initialize_tree(inputs, model, draft_model, past_key_values, draft_past_key_values,
                              method=None, video_token_id=151656, drop_rate=None):
    #Find the last video_token
    last_video_idx = get_last_video_idx(inputs['input_ids'][0], video_token_id)
    text_input_ids = inputs['input_ids'][:,last_video_idx+1:].clone()
    text_attention_mask = inputs['attention_mask'][:,last_video_idx+1:].clone()

    inputs['input_ids'] = inputs['input_ids'][:, :last_video_idx+1]
    inputs['attention_mask'] = inputs['attention_mask'][:, :last_video_idx+1]


    #First stage of prefilling video tokens
    output = model(
        **inputs, past_key_values=past_key_values
    )

    #Second stage of prefilling text tokens
    output = model(
        input_ids=text_input_ids, 
        past_key_values=past_key_values, 
        output_attentions=True,
    )
    logits = output.logits
    sample_token = torch.argmax(logits[:, -1])
    sample_token = sample_token[None, None]

    #Prefill of Draft Model
    inputs['input_ids'] = torch.cat([inputs['input_ids'], text_input_ids], dim=1)
    inputs['attention_mask'] = torch.cat([inputs['attention_mask'], text_attention_mask], dim=1)
    output_draft = draft_model(
        **inputs, past_key_values=draft_past_key_values
    )
    return sample_token






def initialize_tree_with_pruning(inputs, model, draft_model, past_key_values, draft_past_key_values,
                              method=None, video_token_id=151656, drop_rate=None, idx=None, inputs_drop=None, threshold=None, percentage=None, similarity_threshold=0.95):
    #Find the last video_token
    last_video_idx = get_last_video_idx(inputs['input_ids'][0], video_token_id)
    text_input_ids = inputs['input_ids'][:,last_video_idx+1:].clone()
    text_attention_mask = inputs['attention_mask'][:,last_video_idx+1:].clone()

    input_ids = inputs['input_ids'].clone()
    inputs['input_ids'] = inputs['input_ids'][:, :last_video_idx+1]
    inputs['attention_mask'] = inputs['attention_mask'][:, :last_video_idx+1]

    #First stage of prefilling video tokens
    output1 = model(
        **inputs, past_key_values=past_key_values
    )
    # video_emb = output1.output_embeddings
    # video_features = output1.video_hidden_states

    #Second stage of prefilling text tokens
    output2 = model(
        input_ids=text_input_ids, 
        past_key_values=past_key_values, 
        output_attentions=True,
    )
    logits = output2.logits
    attentions = output2.attentions
    # text_emb = output2.output_embeddings
    sample_token = torch.argmax(logits[:, -1])
    sample_token = sample_token[None, None]

    #Prefill of Draft Model
    inputs['input_ids'] = torch.cat([inputs['input_ids'], text_input_ids], dim=1)

    scores = None
    #pruning draft kv cache
    # if method == 'random_two_stage':
    #     inputs_drop = drop_visual_tokens_random(inputs, drop_rate=drop_rate,
    #                                 visual_token_id=video_token_id)
    # elif method == 'uniform_two_stage':
    #     inputs_drop = drop_visual_tokens_uniform(inputs, drop_rate=drop_rate,
    #                                 visual_token_id=video_token_id)
    # elif method == 'attention_two_stage':
    #     inputs_drop,scores = drop_visual_tokens_by_attention(attentions, inputs, drop_rate=drop_rate,
    #                                 visual_token_id=video_token_id, output_scores=True, idx=idx)
    # elif method == 'attention_two_stage_reverse':
    #     inputs_drop = drop_visual_tokens_by_attention(attentions, inputs, drop_rate=drop_rate,
    #                                 visual_token_id=video_token_id, reverse=True, idx=idx)
    if method == 'specvlm':
        inputs_drop = drop_visual_tokens_specvlm(attentions, inputs, drop_rate=drop_rate,
                                    visual_token_id=video_token_id, idx=idx, threshold=threshold, percentage=percentage)
    else:
        print("Method not supported")

    draft_input_len = inputs_drop['input_ids'].shape[1]

    #Prefill of Draft Model
    output_draft = draft_model(
        **inputs_drop, past_key_values=draft_past_key_values
    )
    print("Target KV:",past_key_values[0][0].shape)
    print("Draft KV:",draft_past_key_values[0][0].shape)

    return sample_token, input_ids, draft_input_len, scores






def generate_candidates(tree_logits, tree_indices, retrieve_indices, sample_token, processor=None):
    sample_token = sample_token.to(tree_indices.device) #[1,1]

    candidates_logit = sample_token[0] #[1]

    candidates = torch.cat([candidates_logit, tree_logits.to(candidates_logit.device)], dim=-1)

    tree_candidates = candidates[tree_indices] #

    tree_candidates_ext = torch.cat(
        [tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device) - 1], dim=0) #[27]

    #Paths
    cart_candidates = tree_candidates_ext[retrieve_indices] #

    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0) #
    return cart_candidates, tree_candidates



def tree_decoding(
        model,
        tree_candidates,
        past_key_values,
        tree_position_ids,
        input_ids,
        retrieve_indices,
):
    position_ids = tree_position_ids + input_ids.shape[1]

    outputs = model.model(
        tree_candidates,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    tree_logits = model.lm_head(outputs[0]) #[1,26,152128]
    logits = tree_logits[0, retrieve_indices.to(tree_logits.device)] #[15,6,152128]
    return logits, outputs




def evaluate_posterior(
        logits: torch.Tensor,
        candidates: torch.Tensor,
):
    # Greedy decoding based on temperature value
    # Find the tokens that match the maximum logits for each position in the sequence
    posterior_mask = (
            candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
    ).int()
    candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
    accept_length = candidates_accept_length.max()
    # Choose the best candidate
    if accept_length == 0:
        # Default to the first candidate if none are accepted
        best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
    else:
        best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
    return best_candidate, accept_length, logits[best_candidate, accept_length]
    



@torch.no_grad()
def update_inference_inputs(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits,
        tree_logits,
        new_token,
        past_key_values_data_list,
        current_length_data,
        draft_model,
        draft_past_key_values,
        draft_past_key_values_data_list,
        draft_current_length_data,
        sample_p
):
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
            retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)
    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    #Updata Draft model past key values
    del_len = 1
    for i in draft_model.tree_buffer['tree_indices']:
        del_len += len(i)
    # print("del_len:",del_len)
    # del_len = 11 # 1+4+4+1+1
    for draft_past_key_values_data in draft_past_key_values_data_list:
        draft_past_key_values_data = draft_past_key_values_data[..., :-del_len, :]
    draft_current_length_data.fill_(prev_input_len)

    prob = sample_p
    token = torch.argmax(prob)
    token = token[None, None]
    len_posi = input_ids.shape[1] + 1
    tree_logits = tree_draft(input_ids=torch.cat([candidates[None, best_candidate, : accept_length + 1], token],dim=-1),
                              draft_model = draft_model, draft_past_key_values = draft_past_key_values, len_posi = len_posi)

    new_token += accept_length + 1

    return input_ids, tree_logits, new_token, None, token





@torch.no_grad()
def update_inference_inputs_with_pruning(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits,
        tree_logits,
        new_token,
        past_key_values_data_list,
        current_length_data,
        draft_model,
        draft_past_key_values,
        draft_past_key_values_data_list,
        draft_current_length_data,
        sample_p,
        draft_input_len,
):
    prev_input_len = input_ids.shape[1]
    prev_draft_input_len = draft_input_len
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
            retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)
    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    #Updata Draft model past key values
    del_len = 1
    for i in draft_model.tree_buffer['tree_indices']:
        del_len += len(i)
    # print("del_len:",del_len)
    # del_len = 11 # 1+4+4+1+1
    for draft_past_key_values_data in draft_past_key_values_data_list:
        draft_past_key_values_data = draft_past_key_values_data[..., :-del_len, :]
    draft_current_length_data.fill_(prev_draft_input_len)
    draft_input_len += accept_length + 1

    prob = sample_p
    token = torch.argmax(prob)
    token = token[None, None]
    len_posi = draft_input_len + 1
    # len_posi = input_ids.shape[1] + 1
    tree_logits = tree_draft(input_ids=torch.cat([candidates[None, best_candidate, : accept_length + 1], token],dim=-1),
                              draft_model = draft_model, draft_past_key_values = draft_past_key_values, len_posi = len_posi)

    new_token += accept_length + 1

    return input_ids, tree_logits, new_token, None, token, draft_input_len










@torch.no_grad()
def SD_generate(
        inputs,
        model,
        draft_model,
        processor,
        max_new_tokens=512,
        video_token_id=151656,
        log=False,
        tree_choices=mc_sim_7b_63,
    ):
        torch.cuda.synchronize()
        infer_start = time.time()

        #Tree Structure
        tree_buffers = generate_tree_buffers(
                tree_choices, device=model.model.layers[-1].self_attn.q_proj.weight.device
            )
        tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                model.lm_head.weight.device)
        model.tree_buffers = tree_buffers
        model.tree_choices = tree_choices

        #Draft Tree Structure
        draft_model.tree_buffer = generate_tree_buffers_draft(
            tree_choices, device=draft_model.model.layers[-1].self_attn.q_proj.weight.device)

        # Initialize the past key values
        (
                past_key_values,
                past_key_values_data,
                current_length_data,
        ) = initialize_past_key_values(model)
        model.model.past_key_values = past_key_values
        model.model.past_key_values_data = past_key_values_data
        model.model.current_length_data = current_length_data
        
        (
                draft_past_key_values,
                draft_past_key_values_data,
                draft_current_length_data,
        ) = initialize_past_key_values(draft_model)

        input_ids = inputs['input_ids']
        input_ids = input_ids.clone()
        input_len = input_ids.shape[1]

        #Init
        reset_tree_mode(model)
        reset_tree_mode(draft_model)

        #Prefill
        sample_token = initialize_tree(
            inputs, model, draft_model, past_key_values, draft_past_key_values,
            video_token_id=video_token_id
        )

        torch.cuda.synchronize()
        decode_start = time.time()
        
        #First Draft
        first_id = sample_token.to(inputs['input_ids'].device)
        len_posi = inputs['input_ids'].shape[1] + 1
        tree_logits = tree_draft(first_id, draft_model, draft_past_key_values, len_posi)
        model.model.tree_mask = tree_buffers["tree_attn_mask"]
        #tree_logits:[11,10]


        new_token = 0
        accept_length_total = []
        for step in range(max_new_tokens):
            candidates, tree_candidates = generate_candidates(
                tree_logits,
                tree_buffers["tree_indices"],
                tree_buffers["retrieve_indices"],
                sample_token,
                processor,
            )

            logits, outputs = tree_decoding(
                model,
                tree_candidates,
                past_key_values,
                tree_buffers["tree_position_ids"],
                input_ids,
                tree_buffers["retrieve_indices"],
                )
            
            best_candidate, accept_length, sample_p = evaluate_posterior(
                    logits, candidates
                )
            accept_length_total.append(accept_length)

            input_ids, tree_logits, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                tree_buffers["retrieve_indices"],
                logits,
                tree_logits,
                new_token,
                past_key_values_data,
                current_length_data,
                draft_model,
                draft_past_key_values,
                draft_past_key_values_data,
                draft_current_length_data,
                sample_p
            )

            # if processor.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            #     reset_tree_mode(model)
            #     reset_tree_mode(draft_model)
            #     torch.cuda.synchronize()
            #     end = time.time()
            #     return {
            #         'output_ids': input_ids,
            #         'inference_time': end - infer_start,
            #         'decoding_time': end - decode_start,
            #         'mean_accept_length': sum(accept_length_total) / len(accept_length_total),
            #     }

            # Currently, we mannually set the generation length for fair comparison.
            if new_token >= max_new_tokens:
                reset_tree_mode(model)
                reset_tree_mode(draft_model)
                torch.cuda.synchronize()
                end = time.time()
                return {
                    'output_ids': input_ids,
                    'inference_time': end - infer_start,
                    'decoding_time': end - decode_start,
                    'mean_accept_length': sum(accept_length_total) / len(accept_length_total),
                }



@torch.no_grad()
def SD_generate_with_pruning(
        inputs,
        model,
        draft_model,
        processor,
        method,
        drop_rate,
        video_token_id=151656,
        max_new_tokens=512,
        log=False,
        tree_choices=mc_sim_7b_63,
        idx=None,
        inputs_drop=None,
        threshold=None,
        percentage=None,
        similarity_threshold=0.95,
    ):
        torch.cuda.synchronize()
        infer_start = time.time()

        #Tree Structure
        tree_buffers = generate_tree_buffers(
                tree_choices, device=model.model.layers[-1].self_attn.q_proj.weight.device
            )
        tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                model.lm_head.weight.device)
        model.tree_buffers = tree_buffers
        model.tree_choices = tree_choices

        #Draft Tree Structure
        draft_model.tree_buffer = generate_tree_buffers_draft(
            tree_choices, device=draft_model.model.layers[-1].self_attn.q_proj.weight.device)

        # Initialize the past key values
        (
                past_key_values,
                past_key_values_data,
                current_length_data,
        ) = initialize_past_key_values(model)
        model.model.past_key_values = past_key_values
        model.model.past_key_values_data = past_key_values_data
        model.model.current_length_data = current_length_data
        
        (
                draft_past_key_values,
                draft_past_key_values_data,
                draft_current_length_data,
        ) = initialize_past_key_values(draft_model)


        #Init
        reset_tree_mode(model)
        reset_tree_mode(draft_model)

        scores = None
        sample_token, input_ids, draft_input_len, scores = initialize_tree_with_pruning(
            inputs, model, draft_model, past_key_values, draft_past_key_values,
            method, video_token_id, drop_rate, idx=idx, inputs_drop=inputs_drop,
            threshold=threshold, percentage=percentage,similarity_threshold=similarity_threshold,
        )


        input_ids = input_ids.clone()
        input_len = input_ids.shape[1]


        torch.cuda.synchronize()
        decode_start = time.time()
        #First Draft
        first_id = sample_token.to(input_ids.device)
        len_posi = draft_input_len + 1
        # len_posi = inputs['input_ids'].shape[1] + 1
        tree_logits = tree_draft(first_id, draft_model, draft_past_key_values, len_posi)
        model.model.tree_mask = tree_buffers["tree_attn_mask"]


        new_token = 0
        accept_length_total = []
        for step in range(max_new_tokens):
            candidates, tree_candidates = generate_candidates(
                tree_logits,
                tree_buffers["tree_indices"],
                tree_buffers["retrieve_indices"],
                sample_token,
                processor,
            )

            logits, outputs = tree_decoding(
                model,
                tree_candidates,
                past_key_values,
                tree_buffers["tree_position_ids"],
                input_ids,
                tree_buffers["retrieve_indices"],
                )
            
            best_candidate, accept_length, sample_p = evaluate_posterior(
                    logits, candidates
                )
            accept_length_total.append(accept_length)

            input_ids, tree_logits, new_token, hidden_state, sample_token, draft_input_len = update_inference_inputs_with_pruning(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                tree_buffers["retrieve_indices"],
                logits,
                tree_logits,
                new_token,
                past_key_values_data,
                current_length_data,
                draft_model,
                draft_past_key_values,
                draft_past_key_values_data,
                draft_current_length_data,
                sample_p,
                draft_input_len
            )

            # if processor.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            #     reset_tree_mode(model)
            #     reset_tree_mode(draft_model)
            #     torch.cuda.synchronize()
            #     end = time.time()
            #     return {
            #         'output_ids': input_ids,
            #         'inference_time': end - infer_start,
            #         'decoding_time': end - decode_start,
            #         'mean_accept_length': sum(accept_length_total) / len(accept_length_total),
            #     }

            # Currently, we mannually set the generation length for fair comparison.
            if new_token >= max_new_tokens:
                reset_tree_mode(model)
                reset_tree_mode(draft_model)
                torch.cuda.synchronize()
                end = time.time()
                return {
                    'output_ids': input_ids,
                    'inference_time': end - infer_start,
                    'decoding_time': end - decode_start,
                    'mean_accept_length': sum(accept_length_total) / len(accept_length_total),
                    'scores': scores,
                }
            


@torch.no_grad()
def AR_generate(inputs, model, max_new_tokens=100,video_token_id=151656):
    torch.cuda.synchronize()
    tic1 = time.time()

    (
        past_key_values,
        past_key_values_data,
        current_length_data,
    ) = initialize_past_key_values(model)
    model.model.past_key_values = past_key_values
    model.model.past_key_values_data = past_key_values_data
    model.model.current_length_data = current_length_data

    model.model.tree_mask = None 

    input_ids = inputs['input_ids']
    batch_size = input_ids.shape[0]
    
    with torch.no_grad():
        #Prefill the cache with the first token
        #Find the last video_token
        last_video_idx = get_last_video_idx(inputs['input_ids'][0], video_token_id)
        text_input_ids = inputs['input_ids'][:,last_video_idx+1:].clone()
        text_attention_mask = inputs['attention_mask'][:,last_video_idx+1:].clone()

        inputs['input_ids'] = inputs['input_ids'][:, :last_video_idx+1]
        inputs['attention_mask'] = inputs['attention_mask'][:, :last_video_idx+1]

        #First stage of prefilling video tokens
        output = model(
            **inputs, past_key_values=past_key_values
        )

        #Second stage of prefilling text tokens
        output = model(
            input_ids=text_input_ids, 
            past_key_values=past_key_values, 
            output_attentions=True,
        )
        logits = output.logits
        attentions = output.attentions
        # print(attentions)

        next_token = torch.argmax(logits[:, -1])
        next_token = next_token[None, None]

        inputs['input_ids'] = torch.cat([inputs['input_ids'], text_input_ids], dim=1)
        generated = torch.cat([inputs['input_ids'], next_token], dim=1)

        torch.cuda.synchronize()
        tic2 = time.time()
        
        for step in range(max_new_tokens - 1):
            new_inputs = {
                'input_ids': next_token,
                'past_key_values': past_key_values,
            }
            outputs = model(**new_inputs)
            
            next_token = torch.argmax(outputs.logits[:, -1:], dim=-1)
            generated = torch.cat([generated, next_token], dim=-1)

        torch.cuda.synchronize()
        toc = time.time()
        
        return {
        'output_ids':generated,
        'inference_time':toc - tic1,
        'decoding_time':toc - tic2,
    }



