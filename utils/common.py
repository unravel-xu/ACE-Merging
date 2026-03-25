import os
import re
import copy
import torch
import hashlib
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, GenerationConfig
from collections import OrderedDict
from huggingface_hub import hf_hub_download

import os
import hashlib

def safe_name(model_id: str, max_len: int = 100) -> str:
    if os.path.exists(model_id) or model_id.startswith((".", "/", "~")):
        parts = os.path.normpath(model_id).split(os.sep)
        model_type = None
        base = None
        if "checkpoints" in parts:
            idx = parts.index("checkpoints")
            if idx + 1 < len(parts):
                model_type = parts[idx + 1]
                base = parts[idx + 2]
        h = hashlib.md5(model_id.encode()).hexdigest()[:6]

        if model_type and base:
            name = f"{model_type}_{base}_{h}"
        else:
            name = f"{h}"

    else:
        name = model_id.replace("/", "_")  # 把 / 换成 _
        if len(name) > max_len:
            h = hashlib.md5(model_id.encode()).hexdigest()[:6]
            name = name[:max_len] + "_" + h

    return name


def construct_task_vectors_list(task_vector_dir, other_model_id_list, base_model_id):
    task_vectors_name = {}
    safe_base = safe_name(base_model_id)
    for model_id in other_model_id_list:
        safe_other = safe_name(model_id)
        fname = f"{safe_base}&{safe_other}.pth"
        task_vectors_name[model_id] = os.path.join(task_vector_dir, fname)
    return task_vectors_name

def get_missing_task_vectors(task_vectors):
    return {key: path for key, path in task_vectors.items() if not os.path.exists(path)}

def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict

def get_param_names_to_merge(input_param_names: list, exclude_param_names_regex: list):
    if not exclude_param_names_regex:
        return input_param_names
    
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any([re.match(exclude_pattern, param_name) for exclude_pattern in exclude_param_names_regex])
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge

def save_task_vector(task_vector_save_path, task_vector):
    os.makedirs(Path(task_vector_save_path).parent, exist_ok=True)
    if not os.path.exists(task_vector_save_path):
        torch.save(task_vector.state_dict(), task_vector_save_path)
    else:
        print("already exists!")

def adjust_shape(input_dim, param_tensor):
    shape = param_tensor.shape
    flag = False
    if shape[1] == input_dim:
        adjust_tensor = param_tensor
    elif shape[0] == input_dim:
        adjust_tensor = param_tensor.t()
        flag = True
    else:
        raise ValueError(f"Tensor shape {shape} cannot be reshaped to [n, {input_dim}]")
    return adjust_tensor, flag

def seeding(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def apply_to_model(merged_task_vector, base_model, merged_layers=[], scaling_coefficient = 1, mode='add'):
    """Apply merged task vector to the model's state_dict"""
    
    print(f"Mode: {mode}")
    sd = base_model.state_dict()
    for layer in sd.keys():
        if layer in merged_task_vector and layer not in merged_layers:
            print(f"Updating layer: {layer}")
            if mode == 'add':
                sd[layer] += merged_task_vector[layer] * scaling_coefficient
            elif mode == 'replace':
                sd[layer] = merged_task_vector[layer]
        elif layer in merged_layers:
            print(f"Skipping layer: {layer}")
        else:
            print(f"Keeping base model layer: {layer}")

    base_model.load_state_dict(sd)
    return base_model

def save_merge_model(model_type, model_id, model_save_dir, merge_model):
    os.makedirs(model_save_dir, exist_ok=True)
    if model_type == 'llm':
        gen_config = GenerationConfig.from_pretrained(model_id)
        gen_config.save_pretrained(model_save_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(model_save_dir)
        config = AutoConfig.from_pretrained(model_id)
        config.save_pretrained(model_save_dir)
        merge_model.save_pretrained(model_save_dir)
    if model_type == 'lm':
        merge_model.save_pretrained(model_save_dir)
    if model_type == 'vit':
        model_save_path = os.path.join(model_save_dir, 'model.pt')
        merge_model.save(model_save_path)

def close_ratio(layer_name, tensor1, tensor2):
    bins = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, float('inf')]
    diff = (tensor1 - tensor2).abs().flatten()
    total = diff.numel()
    results = []
    prev = bins[0]
    for threshold in bins[1:]:
        count = ((diff > prev) & (diff <= threshold)).sum().item()
        percent = 100 * count / total
        results.append(f"{count} ({percent:.2f}%)")
        prev = threshold
    row = f"| {layer_name} | " + " | ".join(results) + " |"
    print(row)
    return results