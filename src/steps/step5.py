import os
import json
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from src.vit.model import ImageEncoder
from src.train.model import SparseAutoEncoder
from src.merge.merge import MergeTaskVector
from utils.common import apply_to_model, save_merge_model
from utils.task_vector import TaskVector

def run_step4(config):
    if not hasattr(config, "base_model"):
        if config.model_type == 'llm':
            config.base_model = AutoModelForCausalLM.from_pretrained(config.basic.model_id_list[0], device_map = config.device)
        elif config.model_type == 'lm':
            config.base_model = AutoModelForSequenceClassification.from_pretrained(config.basic.model_id_list[0], device_map = config.device)
        elif config.model_type == 'vit':
            config.base_model = ImageEncoder.load_from_hf_hub(config.basic.model_id_list[0], device_map = config.device)
    merged_layers = []
    task_vector_list = [TaskVector.load(task_vector_path) for task_vector_path in config.runtime.task_vectors_path.values()]
    if config.with_sae:
        for i in range(len(config.sae_model_list_save_dir)):
            sae_model_save_dir = config.sae_model_list_save_dir[i]
            sae_model_path = os.path.join(sae_model_save_dir, 'sae_model.pth')
            sae_model_json_path = os.path.join(sae_model_save_dir, 'train.json')

            with open(sae_model_json_path, 'r', encoding='utf-8') as f:
                infos = json.load(f)[-1]

            merge_layers = infos["layers"][-1]

            config.train.aux_lambda = infos["train"]["aux_lambda"]
            config.train.warmup_epochs = infos["train"]["warmup_epochs"],
            config.train.recon_loss_threshold = infos["train"]["recon_loss_threshold"],

            sae_model = SparseAutoEncoder.load(sae_model_path, config.device)
            merge_task_vector = MergeTaskVector(
                base_model = config.base_model,
                task_vector_list = task_vector_list,
                merge_layers = merge_layers,
                sae_model= sae_model,
                device = config.device
            )

            if config.with_sparisfy:
                merged_task_vector = merge_task_vector.merge_task_vectors(config.merge, config.sparsify)
            else:
                merged_task_vector = merge_task_vector.merge_task_vectors(config.merge, None)
            
            config.base_model = apply_to_model(merged_task_vector, merged_layers, config.base_model)
            
            merged_layers.extend(merge_layers)
          
    merge_layers = [layer for layer in task_vector_list[0].state_dict()['task_vector'].keys() if layer not in merged_layers]

    merge_task_vector = MergeTaskVector(
        base_model = config.base_model,
        task_vector_list = task_vector_list,
        merge_layers = merge_layers,
        sae_model = None,
        device = config.device
    )
    
    if config.with_sparisfy:
        merged_task_vector = merge_task_vector.merge_task_vectors(config.merge, config.sparsify)
    else:
        merged_task_vector = merge_task_vector.merge_task_vectors(config.merge, None)

    config.base_model = apply_to_model(merged_task_vector, merged_layers, config.base_model)
    
    merged_layers.extend(merge_layers)

    model_save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), f'../../models/merged/{config.model_type}/{config.basic.task_id}/'))
    save_merge_model(config.model_type, config.basic.model_id_list[0], model_save_dir, config.base_model)
    print(f"{model_save_dir}")
    config.merged_model_save_path = model_save_dir
    config.save_config()
    return None