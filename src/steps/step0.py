from utils.common import construct_task_vectors_list, get_missing_task_vectors

def run_step0(config):
    task_vector_dir = config.basic.task_vector_dir
    base_model_id = config.basic.model_id_list[0]
    other_model_id_list = config.basic.model_id_list[1:]
    config.runtime.task_vectors_path = construct_task_vectors_list(task_vector_dir, other_model_id_list, base_model_id)
    config.runtime.missing_task_vectors = get_missing_task_vectors(config.runtime.task_vectors_path)
    if config.runtime.missing_task_vectors:
        for k in config.runtime.missing_task_vectors.keys():
            print(f'\t Missing task vectors for {k}')
        return 1
    else:
        print("[STEP0] All task vectors already exist.")
        if config.with_merge:
            return 3
        elif config.with_sparisfy:
            return 2
        else:
            return None