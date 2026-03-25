from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, GPT2ForSequenceClassification
from src.models.modeling import ImageEncoder
from utils.task_vector import TaskVector
from utils.common import save_task_vector

def run_step1(config):
    base_model_id = config.basic.model_id_list[0]
    if config.model_type == 'llm':
        config.base_model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map = config.device)
    elif config.model_type == 'lm':
        if config.lm_type == 'gpt2':
            config.base_model = GPT2ForSequenceClassification.from_pretrained(base_model_id, device_map = config.device)
        elif config.lm_type == 'bert':
            config.base_model = AutoModelForSequenceClassification.from_pretrained(base_model_id, device_map = config.device)
    elif config.model_type == 'vit':
        config.base_model = ImageEncoder.load(config.vit_type, base_model_id, config.device)

    for model_id in tqdm(config.runtime.missing_task_vectors.keys()):
        if config.model_type == 'llm':
            pt_model = AutoModelForCausalLM.from_pretrained(model_id, device_map = config.device)
            exclude_param_names_regex = [".*embed_tokens.*", "lm_head.weight"]
        elif config.model_type == 'lm':
            if config.lm_type == 'gpt2':
                pt_model = GPT2ForSequenceClassification.from_pretrained(model_id, device_map = config.device)
                exclude_param_names_regex = [".*score.*"]
            elif config.lm_type == 'bert':
                pt_model = AutoModelForSequenceClassification.from_pretrained(model_id, device_map = config.device)
                exclude_param_names_regex = [".*classifier.*", ".*bias.*", ".*LayerNorm.*", ".*embeddings.*"]
        elif config.model_type == 'vit':
            pt_model = ImageEncoder.load(config.vit_type, model_id, config.device)
            exclude_param_names_regex = None
        task_vector = TaskVector()
        task_vector.compute(config.base_model, base_model_id, pt_model, model_id, exclude_param_names_regex)
        task_vector_save_path = config.runtime.missing_task_vectors[model_id]
        save_task_vector(task_vector_save_path, task_vector)
    if config.with_sparisfy:
        return 2
    else:
        return 3