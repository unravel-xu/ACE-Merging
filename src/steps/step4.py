import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from src.vit.model import ImageEncoder
from src.train.model import SparseAutoEncoder
from src.merge.strategy import MergeStrategy
from utils.common import apply_to_model, save_merge_model, adjust_shape
from utils.task_vector import TaskVector

def run_step4(config):
    if not hasattr(config, "base_model"):
        if config.model_type == 'llm':
            config.base_model = AutoModelForCausalLM.from_pretrained(config.basic.model_id_list[0], device_map = config.device)
        elif config.model_type == 'lm':
            config.base_model = AutoModelForSequenceClassification.from_pretrained(config.basic.model_id_list[0], device_map = config.device)
        elif config.model_type == 'vit':
            config.base_model = ImageEncoder.load_from_hf_hub(config.basic.model_id_list[0], device_map = config.device)
        
    sae_model_list = []
    task_vector_list = [TaskVector.load(task_vector_path, config.device) for task_vector_path in config.runtime.task_vectors_path.values()]
    
    if config.with_sae:
        adjust_shape_layer_list = []
        all_sae_layers = []
        for i in range(len(config.sae_model_list_save_dir)):
            sae_model_save_dir = config.sae_model_list_save_dir[i]
            sae_model_path = os.path.join(sae_model_save_dir, 'sae_model.pth')
            sae_model_json_path = os.path.join(sae_model_save_dir, 'train.json')

            with open(sae_model_json_path, 'r', encoding='utf-8') as f:
                infos = json.load(f)[-1]

            apply_sae_layers = infos['layers'][-1]
            all_sae_layers.extend(apply_sae_layers)

            sae_model = SparseAutoEncoder.load(sae_model_path, config.device)
            sae_model.eval()
            sae_model_list.append({'model': sae_model, 'layers': apply_sae_layers})

            for layer in apply_sae_layers:
                param_list = []
                for task_vector in task_vector_list:
                    tensor = task_vector.task_vector_param_dict[layer]
                    if tensor.ndim == 2:
                        tensor, flag = adjust_shape(sae_model.input_dim, task_vector.task_vector_param_dict[layer])
                        if flag:
                            adjust_shape_layer_list.append(layer)
                    param_list.append(tensor.to(config.device))
                
                acts_topk, _ = sae_model.encoder(torch.stack(param_list, dim=0))

                for i, task_vector in enumerate(task_vector_list):
                    task_vector.task_vector_param_dict[layer] = acts_topk[i].detach().clone()
    
    merge_layers = task_vector_list[0].task_vector_param_dict.keys()
    # if config.with_sae:
    #     merge_layers = all_sae_layers
    ms = MergeStrategy(merge_layers, task_vector_list, config.device)

    if config.merge["method"] == "average":
        merged_task_vector = ms.average_merging()
    elif config.merge["method"] == "task_arithmetic":
        merged_task_vector = ms.task_arithmetic()
    elif config.merge["method"] == "ties":
        merged_task_vector = ms.ties_merging()
    elif config.merge["method"] == "shared_basis":
        merged_task_vector = ms.shared_basis()
    elif config.merge["method"] == "geo_proj":
        merged_task_vector = ms.geo_proj()
    elif config.merge["method"] == 'tsvd':
        merged_task_vector = ms.t_svd_merge()
    elif config.merge["method"] == 'isoc':
        merged_task_vector = ms.iso_c()
    elif config.merge["method"] == "emr":
        merged_task_vector = ms.emr_merging(config.merge["idx"])
    elif config.merge["method"] == "tsv":
        merged_task_vector = ms.tsv()
    elif config.merge["method"] == "tsvqr":
        merged_task_vector = ms.tsvqr()
    elif config.merge["method"] == "boost":
        merged_task_vector = ms.boost()
    elif config.merge["method"] == "boost_svd":
        merged_task_vector = ms.boost_svd()
    

    if sae_model_list:
        for i in range(len(sae_model_list)):
            for layer in sae_model_list[i]['layers']:
                sae_model = sae_model_list[i]['model']
                merged_task_vector[layer] = sae_model.decoder(merged_task_vector[layer])
                if layer in adjust_shape_layer_list:
                    merged_task_vector[layer] = merged_task_vector[layer].t()


    config.base_model = apply_to_model(merged_task_vector, [], config.base_model)
    
    model_save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), f'../../models/merged/{config.model_type}/{config.basic.task_id}/'))
    save_merge_model(config.model_type, config.basic.model_id_list[0], model_save_dir, config.base_model)
    print(f"{model_save_dir}")
    config.merged_model_save_path = model_save_dir
    config.save_config()
    return None