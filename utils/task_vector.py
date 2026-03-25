import torch
from utils.common import get_param_names_to_merge

class TaskVector:
    def __init__(self):
        self.task_vector_param_dict = {}
        self.change_param_ratio = {}
    
    def compute(self, base_model, base_model_name, pt_model, pt_model_name, exclude_param_names_regex):
        self.info = {
            "base_model" : base_model_name,
            "pt_model" : pt_model_name
        }
        base_param_dict = {param_name: param_value for param_name, param_value in base_model.named_parameters()}
        pt_param_dict = {param_name: param_value for param_name, param_value in pt_model.named_parameters()}
        param_names_to_merge = get_param_names_to_merge(input_param_names=list(base_param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
        if not param_names_to_merge:
            raise ValueError("No common parameters found between base and post-train model.")
        
        with torch.no_grad():
            for param_name in param_names_to_merge:
                self.task_vector_param_dict[param_name] = pt_param_dict[param_name] - base_param_dict[param_name]
                self.change_param_ratio[param_name] = 1 - (self.task_vector_param_dict[param_name] == 0).sum().item() / self.task_vector_param_dict[param_name].numel()                 
                         
    def state_dict(self):
        return {
            "task_vector": self.task_vector_param_dict,
            "change_param_ratio": self.change_param_ratio,
            "info": self.info
        }
        
    def load_state_dict(self, state_dict):
        self.task_vector_param_dict = state_dict["task_vector"]
        self.change_param_ratio = state_dict["change_param_ratio"]
        self.info = state_dict["info"]
        
    def get_rank_of_change_param_ratio(self):
        return sorted(self.change_param_ratio.items(), key=lambda e: e[1], reverse=True)
    
    @classmethod
    def load(cls, task_vector_save_path, device):
        task_vector = cls()
        task_vector.load_state_dict(torch.load(task_vector_save_path, weights_only=True, map_location=device))
        return task_vector