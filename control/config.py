import os
import json
from pathlib import Path
from datetime import datetime
from easydict import EasyDict

class ConfigManager:
    def __init__(self):
        self.with_sparisfy = False
        self.with_merge = True
        # self.model_type = 'llm'
        self.model_type = 'vit'
        self.vit_type = 'ViT-B-16'
        # self.vit_type = 'ViT-B-32'
        # self.vit_type = 'ViT-L-14'
        # self.model_type = 'lm'
        # self.lm_type = 'bert'
        # self.lm_type = 'gpt2'
        self.merge_layers = 'auto'
        self.device = "cuda:0"
        
        self.basic = EasyDict({
            'config_save_path': f'./data/configs/{self.model_type}',
            'task_vector_dir': './data/task_vectors/',
            'model_id_list': [
                # 'roberta-base',
                # './data/models/lm/checkpoints/roberta-base/cola/roberta-base_lr1e-05/',
                # './data/models/lm/checkpoints/roberta-base/sst2/roberta-base_lr1e-05/',
                # './data/models/lm/checkpoints/roberta-base/mrpc/roberta-base_lr1e-05/',
                # './data/models/lm/checkpoints/roberta-base/stsb/roberta-base_lr1e-05/',
                # './data/models/lm/checkpoints/roberta-base/qqp/roberta-base_lr1e-05/',
                # './data/models/lm/checkpoints/roberta-base/qnli/roberta-base_lr1e-05/',
                # './data/models/lm/checkpoints/roberta-base/mnli/roberta-base_lr1e-05/',
                # './data/models/lm/checkpoints/roberta-base/rte/roberta-base_lr1e-05/',

                # 'roberta-large',
                # './data/models/lm/checkpoints/roberta-large/cola/roberta-large_lr1e-05/',
                # './data/models/lm/checkpoints/roberta-large/sst2/roberta-large_lr1e-05/',
                # './data/models/lm/checkpoints/roberta-large/mrpc/roberta-large_lr1e-05/',
                # './data/models/lm/checkpoints/roberta-large/stsb/roberta-large_lr1e-05/',
                # './data/models/lm/checkpoints/roberta-large/qqp/roberta-large_lr1e-05/',
                # './data/models/lm/checkpoints/roberta-large/qnli/roberta-large_lr1e-05/',
                # './data/models/lm/checkpoints/roberta-large/mnli/roberta-large_lr1e-05/',
                # './data/models/lm/checkpoints/roberta-large/rte/roberta-large_lr1e-05/',

                # 'gpt2',
                # 'tanganke/gpt2_cola',
                # 'tanganke/gpt2_sst2',
                # 'tanganke/gpt2_mrpc',
                # 'tanganke/gpt2_qqp',
                # 'tanganke/gpt2_mnli',
                # 'tanganke/gpt2_qnli',
                # 'tanganke/gpt2_rte'

                f'./data/models/vit/checkpoints/{self.vit_type}/MNISTVal/nonlinear_zeroshot.pt',
                f'./data/models/vit/checkpoints/{self.vit_type}/MNISTVal/nonlinear_finetuned.pt',
                f'./data/models/vit/checkpoints/{self.vit_type}/CarsVal/nonlinear_finetuned.pt',
                f'./data/models/vit/checkpoints/{self.vit_type}/DTDVal/nonlinear_finetuned.pt',
                f'./data/models/vit/checkpoints/{self.vit_type}/EuroSATVal/nonlinear_finetuned.pt',
                f'./data/models/vit/checkpoints/{self.vit_type}/GTSRBVal/nonlinear_finetuned.pt',
                f'./data/models/vit/checkpoints/{self.vit_type}/RESISC45Val/nonlinear_finetuned.pt',
                f'./data/models/vit/checkpoints/{self.vit_type}/SUN397Val/nonlinear_finetuned.pt',
                f'./data/models/vit/checkpoints/{self.vit_type}/SVHNVal/nonlinear_finetuned.pt',

                # f'./data/models/vit/checkpoints/{self.vit_type}/PCAMVal/nonlinear_finetuned.pt',
                # f'./data/models/vit/checkpoints/{self.vit_type}/CIFAR100Val/nonlinear_finetuned.pt',
                # f'./data/models/vit/checkpoints/{self.vit_type}/STL10Val/nonlinear_finetuned.pt',
                # f'./data/models/vit/checkpoints/{self.vit_type}/OxfordIIITPetVal/nonlinear_finetuned.pt',
                # f'./data/models/vit/checkpoints/{self.vit_type}/Flowers102Val/nonlinear_finetuned.pt',
                # f'./data/models/vit/checkpoints/{self.vit_type}/FER2013Val/nonlinear_finetuned.pt',

                # f'./data/models/vit/checkpoints/{self.vit_type}/CIFAR10Val/nonlinear_finetuned.pt',
                # f'./data/models/vit/checkpoints/{self.vit_type}/Food101Val/nonlinear_finetuned.pt',
                # f'./data/models/vit/checkpoints/{self.vit_type}/RenderedSST2Val/nonlinear_finetuned.pt',
                # f'./data/models/vit/checkpoints/{self.vit_type}/EMNISTVal/nonlinear_finetuned.pt',
                # f'./data/models/vit/checkpoints/{self.vit_type}/FashionMNISTVal/nonlinear_finetuned.pt',
                # f'./data/models/vit/checkpoints/{self.vit_type}/KMNISTVal/nonlinear_finetuned.pt',

                # 'meta-llama/Llama-3.2-3B-Instruct',
                # 'MergeBench/Llama-3.2-3B-Instruct_multilingual',
                # 'MergeBench/Llama-3.2-3B-Instruct_math',
                # 'MergeBench/Llama-3.2-3B-Instruct_coding'
            ],
        })

        self.merge = EasyDict({
            # "method": "average",
            # "threshold": 1e-8
            # "method": "wudi",
            # "method": "task_arithmetic",
            # "method": "isoc",
            "method": "ace",
            # "method": "tsvm",
            # "method": "pca",
            # "method": "isocts",
            # "method": "cart",
            "scaling_coefficient": 1.0,
            # "method": "breadcrumbs",
            # "param_density": 0.9,
            # "param_value_mask_rate": 0.01,
            # "scaling_coefficient": 0.995
            # "scaling_coefficient": 0.3
        })

        self.sparsify = EasyDict({
            'method': 'rescaled_random',
            'place': ['front'],
            'density': 0.9,
            'rescale': True
        })

        self.runtime = EasyDict()

        os.makedirs(self.basic.config_save_path, exist_ok=True)
        self.basic.task_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        config_path = f'{self.basic.config_save_path}/{self.basic.task_id}.json'  
        self.basic.config_path = Path(config_path)

    def print_info(self):
        print(json.dumps(self.__dict__, indent=4, default=str))
    
    def save_config(self):
        exclude_keys = ['base_model', 'select_layers', 'datasets']
        def to_serializable(obj):
            if isinstance(obj, EasyDict):
                return {k: to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [to_serializable(i) for i in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj
        sanitized = {}
        if not self.with_sparisfy:
            exclude_keys.append('sparsify')
        for k, v in self.__dict__.items():
            if k in exclude_keys:
                continue
            sanitized[k] = to_serializable(v)
        with open(self.basic.config_path, "w", encoding='utf-8') as f:
            json.dump(sanitized, f, indent=4, ensure_ascii=False)