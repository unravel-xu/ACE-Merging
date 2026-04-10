import os
from utils.task_vector import TaskVector
from utils.common import apply_to_model, save_merge_model
from src.merge.strategy import MergeStrategy
from src.models.modeling import ImageEncoder
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification


def run_step3(config):
    if not hasattr(config, "base_model"):
        base_model_id = config.basic.model_id_list[0]
        print(base_model_id)
        if config.model_type == 'llm':
            config.base_model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map = config.device)
        elif config.model_type == 'lm':
            config.base_model = AutoModelForSequenceClassification.from_pretrained(base_model_id, device_map = config.device)
        elif config.model_type == 'vit':
            config.base_model = ImageEncoder.load(config.vit_type, base_model_id, config.device)

    task_vector_list = [TaskVector.load(task_vector_path, config.device) for task_vector_path in config.runtime.task_vectors_path.values()]
    merge_layers = task_vector_list[0].task_vector_param_dict.keys()
    ms = MergeStrategy(merge_layers, task_vector_list, config.device)

    if config.merge["method"] == "average":
        merged_task_vector = ms.average_merging()
    elif config.merge["method"] == "task_arithmetic":
        merged_task_vector = ms.task_arithmetic()
    elif config.merge["method"] == "ties":
        merged_task_vector = ms.ties_merging()
    elif config.merge["method"] == "wudi":
        merged_task_vector = ms.wudi_merging()
    elif config.merge["method"] == "cart":
        merged_task_vector = ms.cart()
    elif config.merge["method"] == 'pca':
        merged_task_vector = ms.pca_merging()
    elif config.merge["method"] == 'isoc':
        merged_task_vector = ms.iso_c()
    elif config.merge["method"] == 'isocts':
        merged_task_vector = ms.iso_cts()
    elif config.merge["method"] == "emr":
        merged_task_vector = ms.emr_merging(config.merge["idx"])
    elif config.merge["method"] == "tsvm":
        merged_task_vector = ms.tsvm()
    elif config.merge["method"] == "ace":
        merged_task_vector = ms.ace_merging()
    elif config.merge["method"] == "fr":
        merged_task_vector = ms.fr_merging()
    elif config.merge["method"] == "boost_svd":
        merged_task_vector = ms.boost_svd()
    elif config.merge["method"] == "rpca":
        merged_task_vector = ms.rpca()

    config.base_model = apply_to_model(merged_task_vector, config.base_model, scaling_coefficient=config.merge["scaling_coefficient"])
    model_save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), f'../../data/models/merged/{config.model_type}/{config.basic.task_id}/'))
    save_merge_model(config.model_type, config.basic.model_id_list[0], model_save_dir, config.base_model)
    print(f"{model_save_dir}")
    config.merged_model_save_path = model_save_dir
    config.save_config()
    return None