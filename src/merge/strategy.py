import time
import torch
from functools import wraps
import torch.nn.functional as F
import numpy as np
from collections import Counter

def track_stats(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        # 1. 清空显存缓存，准备统计起始显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            start_mem = torch.cuda.memory_allocated(self.device)
            torch.cuda.synchronize(self.device)
        
        start_time = time.perf_counter()

        # 执行核心合并算法
        result = method(self, *args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        
        end_time = time.perf_counter()

        # 3. 计算统计量
        duration = end_time - start_time
        if torch.cuda.is_available():
            # 峰值显存消耗 = 运行期间达到的最高显存 - 初始显存
            peak_mem = torch.cuda.max_memory_allocated(self.device) - start_mem
            peak_mem_mb = peak_mem / (1024 ** 2)
        else:
            peak_mem_mb = 0.0

        print(f"--- Stats for {method.__name__} ---")
        print(f"Execution Time: {duration:.4f} seconds")
        print(f"Peak VRAM Usage: {peak_mem_mb:.2f} MB")
        print("-" * 30)

        # 将统计结果也返回（可选，如果需要后续绘图）
        return result
    return wrapper

class MergeStrategy:

    def __init__(self, merge_layers, task_vector_list, device):
        self.merge_layers = merge_layers
        self.task_vector_list = task_vector_list
        self.device = device

    def average_merging(self):
        merged_task_vector = {}
        for layer in self.merge_layers:
            layer_param_list = []
            for task_vector in self.task_vector_list:
                layer_param_list.append(task_vector.task_vector_param_dict[layer])
            merged_task_vector[layer] = torch.stack(layer_param_list).mean(dim=0)
        return merged_task_vector

    def task_arithmetic(self):
        merged_task_vector = {}
        for layer in self.merge_layers:
            layer_param_list = []
            for task_vector in self.task_vector_list:
                layer_param_list.append(task_vector.task_vector_param_dict[layer])
            merged_task_vector[layer] = torch.stack(layer_param_list).sum(dim=0)
        return merged_task_vector

    def ties_merging(self, param_value_mask_rate=0.1):
        def mask_smallest_magnitude_param_values(flattened_models_to_merge_param, param_value_mask_rate):
            num_mask_params = int(flattened_models_to_merge_param.shape[1] * param_value_mask_rate)
            kth_values, _ = flattened_models_to_merge_param.abs().kthvalue(k=num_mask_params, dim=1, keepdim=True)
            mask = flattened_models_to_merge_param.abs() >= kth_values
            return flattened_models_to_merge_param * mask

        def get_param_signs(flattened_models_to_merge_param):
            param_signs = torch.sign(flattened_models_to_merge_param.sum(dim=0))
            majority_sign = torch.sign(param_signs.sum(dim=0))
            param_signs[param_signs == 0] = majority_sign
            return param_signs

        def disjoint_merge(flattened_models_to_merge_param, param_signs):
            param_to_preserve_mask = ((param_signs.unsqueeze(dim=0) > 0) & (flattened_models_to_merge_param > 0)) | ((param_signs.unsqueeze(dim=0) < 0) & (flattened_models_to_merge_param < 0))
            param_to_preserve = flattened_models_to_merge_param * param_to_preserve_mask
            num_models_param_preserved = (param_to_preserve != 0).sum(dim=0).float()
            merged_flattened_param = torch.sum(param_to_preserve, dim=0) / torch.clamp(num_models_param_preserved, min=1.0)
            return merged_flattened_param

        merged_task_vector = {}
        for layer in self.merge_layers:
            param_original_shape = self.task_vector_list[0].task_vector_param_dict[layer].shape
            flattened_models_to_merge_param = torch.vstack([task_vector.task_vector_param_dict[layer].flatten() for task_vector in self.task_vector_list])
            flattened_models_to_merge_param = mask_smallest_magnitude_param_values(flattened_models_to_merge_param=flattened_models_to_merge_param, param_value_mask_rate=param_value_mask_rate) 
            param_signs = get_param_signs(flattened_models_to_merge_param=flattened_models_to_merge_param)
            merged_flattened_param = disjoint_merge(flattened_models_to_merge_param=flattened_models_to_merge_param, param_signs=param_signs)
            merged_task_vector[layer] = merged_flattened_param.reshape(param_original_shape)
        return merged_task_vector

    # @track_stats
    def wudi_merging(self, iter_num=1000):
        merged_task_vector = {}
        for layer in self.merge_layers:
            original_device = self.device
            
            # 将所有task_vectors加载到GPU并堆叠
            task_vectors_list = [tv.task_vector_param_dict[layer].to(self.device) for tv in self.task_vector_list]
            task_vectors = torch.stack(task_vectors_list)
            
            num_tvs = len(task_vectors)
            
            # 初始化merged_task_vector
            # 使用.clone().detach()来创建一个新的、不带梯度的副本作为初始值
            initial_guess = torch.mean(task_vectors, dim=0) # 使用均值作为初始点通常比求和更稳定
            merged_param = torch.nn.Parameter(initial_guess)

            if task_vectors[0].ndim == 2 and "text_projection" not in layer:
                # 优化器只优化我们创建的这个Parameter
                optimizer = torch.optim.Adam([merged_param], lr=1e-4) # 学习率可以适当调高

                # 预先计算每个task_vector的F范数的平方 (这是一个标量)
                # 这等价于 ||vec(W_t)||_2^2
                l2_norms_sq = torch.sum(task_vectors.view(num_tvs, -1)**2, dim=-1)

                for i in range(iter_num):
                    # 计算干扰项
                    # 使用广播 (T, d, m) - (d, m) -> (T, d, m)
                    disturbing_vectors = merged_param - task_vectors
                    
                    # 计算Frobenius内积 <disturbing_vector, task_vector>_F
                    # (A*B).sum()
                    # 我们需要对每个任务分别计算
                    # disturb_vecs: (T, d, m), task_vecs: (T, d, m)
                    # product_elementwise: (T, d, m)
                    product_elementwise = disturbing_vectors * task_vectors
                    
                    # inner_products: (T,)
                    # 对每个任务的 d*m 个元素求和
                    inner_products = torch.sum(product_elementwise, dim=(1, 2))
                    
                    # 计算损失：sum_t ( <W_m-W_t, W_t>_F^2 / ||W_t||_F^2 )
                    # 这是一个标量
                    loss = torch.sum(torch.square(inner_products) / l2_norms_sq)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # 可选：打印loss观察收敛情况
                    if i % 50 == 0:
                        print(f"Layer {layer}, Iter {i}, Loss: {loss.item()}")

                # 优化结束后，将结果存入字典
                merged_task_vector[layer] = merged_param.detach()
            else: # 对于非二维张量，直接求平均
                merged_task_vector[layer] = torch.mean(task_vectors, dim=0)
            
            # 将最终结果移回原始设备
            merged_task_vector[layer] = merged_task_vector[layer].to(
                device=original_device, non_blocking=True
            )
        return merged_task_vector

    def iso_c(self):
        with torch.no_grad():
            merged_task_vector = {}
            for layer in self.merge_layers:
                task_vectors = [
                    tv.task_vector_param_dict[layer].to(self.device) 
                    for tv in self.task_vector_list
                ]

                merged_task_vector[layer] = sum(task_vectors) / len(task_vectors)

                if task_vectors[0].ndim == 2 and "text_projection" not in layer:
                    merged_task_vector[layer] *= len(task_vectors)
                    U, S, V = torch.linalg.svd(merged_task_vector[layer], full_matrices=False)
                    S_mean = torch.ones_like(S) * S.mean()

                    merged_task_vector[layer] = torch.linalg.multi_dot(
                        (
                            U,
                            torch.diag(S_mean),
                            V,
                        )
                    )
            return merged_task_vector
    
    # @track_stats
    def iso_cts(self, common_space_fraction=0.8):
        with torch.no_grad():
            merged_task_vector = {}
            for layer in self.merge_layers:
                task_vectors = [
                    tv.task_vector_param_dict[layer].to(self.device) 
                    for tv in self.task_vector_list
                ]
                shape_ = task_vectors[0].shape
                if task_vectors[0].ndim == 2 and "text_projection" not in layer:
                    print(f"Computing common space using sum for {layer}...")
                    combined_w = sum(task_vectors)
                    common_space_index_s = int(min(shape_) * common_space_fraction)
                    _task_specific_total_space_index_s = round((min(shape_) - common_space_index_s) / len(self.task_vector_list)) * len(self.task_vector_list)
                    common_space_index_s = min(shape_) - _task_specific_total_space_index_s

                    u, s, v = torch.linalg.svd(combined_w, full_matrices=False)
                    common_space_u = u[:, :common_space_index_s]
                    common_space_s = s[:common_space_index_s]
                    common_space_v = v[:common_space_index_s, :]

                    n_dims_per_task = int((min(shape_) - common_space_index_s) / len(self.task_vector_list))
                    for i, task_vector in enumerate(task_vectors):
                        w = task_vector.to(self.device)

                        w_ts = w - common_space_u @ common_space_u.T @ w
                        u_ts, s_ts, v_ts = torch.linalg.svd(w_ts, full_matrices=False)            
            
                        if i == 0:
                            combined_space_u = torch.zeros_like(u_ts, device=self.device)
                            combined_space_s = torch.zeros_like(s_ts, device=self.device)
                            combined_space_v = torch.zeros_like(v_ts, device=self.device)
                

                        combined_space_u[:, i * n_dims_per_task : (i + 1) * n_dims_per_task] = u_ts[:, :n_dims_per_task]
                        combined_space_s[i * n_dims_per_task : (i + 1) * n_dims_per_task] = s_ts[:n_dims_per_task]
                        combined_space_v[i * n_dims_per_task : (i + 1) * n_dims_per_task, :] = v_ts[:n_dims_per_task, :]

                    combined_space_u[:, len(self.task_vector_list) * n_dims_per_task : len(self.task_vector_list) * n_dims_per_task + common_space_index_s] = common_space_u
                    combined_space_s[len(self.task_vector_list) * n_dims_per_task : len(self.task_vector_list) * n_dims_per_task + common_space_index_s] = common_space_s
                    combined_space_v[len(self.task_vector_list) * n_dims_per_task : len(self.task_vector_list) * n_dims_per_task + common_space_index_s, :] = common_space_v
                    
                    ### Orthogonalize combined_space_u and combined_space_v ###
                    u_combined_space_u, s_combined_space_u, v_combined_space_u = torch.linalg.svd(combined_space_u, full_matrices=False)
                    u_combined_space_v, s_combined_space_v, v_combined_space_v = torch.linalg.svd(combined_space_v, full_matrices=False)
                    combined_space_u = u_combined_space_u @ v_combined_space_u
                    combined_space_v = u_combined_space_v @ v_combined_space_v

                    combined_space_s = torch.ones_like(combined_space_s) * combined_space_s.mean()
                
                    merged_task_vector[layer] = torch.linalg.multi_dot(
                        (
                            combined_space_u,
                            torch.diag(combined_space_s),
                            combined_space_v,
                        )
                    )

                else:
                    merged_task_vector[layer] = torch.stack(task_vectors, dim=0).mean(dim=0)

        return merged_task_vector

    # @track_stats
    def tsvm(self):
        sv_reduction = 1 / len(self.task_vector_list)
        print("Computing SVD...")
        with torch.no_grad():
            merged_task_vector = {}
            for layer in self.merge_layers:
                merged_task_vector[layer] = {}
                for i, task_vector in enumerate(self.task_vector_list):
                    vec = task_vector.task_vector_param_dict[layer]
                    if vec.ndim == 2 and "text_projection" not in layer:
                        u, s, v = torch.linalg.svd(vec, full_matrices=False)
                        if i == 0:
                            print(f"Computed SVD for {layer}...")
                            sum_u = torch.zeros_like(u, device=self.device)
                            sum_s = torch.zeros_like(s, device=self.device)
                            sum_v = torch.zeros_like(v, device=self.device)
                        reduced_index_s = int(s.shape[0] * sv_reduction)
                        # select only the first reduced_index_s columns of u and place them
                        sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                            :, :reduced_index_s
                        ]
                        sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                            :reduced_index_s
                        ]
                        # select only the first reduced_index_s rows of v and place them
                        sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                            :reduced_index_s, :
                        ]
                    else:
                        if i == 0:
                            merged_task_vector[layer] = vec.clone()
                        else:
                            merged_task_vector[layer] += (vec - merged_task_vector[layer]) / (i + 1)

                if len(task_vector.task_vector_param_dict[layer].shape) == 2 and "text_projection" not in layer:
                    u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                    u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

                    merged_task_vector[layer] = torch.linalg.multi_dot(
                        (
                            u_u,
                            v_u,
                            torch.diag(sum_s),
                            u_v,
                            v_v,
                        )
                    )
        return merged_task_vector

    def cart(self, rank_ratio: float = 0.08):
        with torch.no_grad():
            merged_task_vector = {}
            for layer in self.merge_layers:
                task_vectors = [
                    tv.task_vector_param_dict[layer].to(self.device) 
                    for tv in self.task_vector_list
                ]

                if task_vectors[0].ndim != 2 or "text_projection" in layer:
                    vec_avg = torch.mean(torch.stack(task_vectors), dim=0)
                    merged_task_vector[layer] = vec_avg
                    continue

                # 情况 2：权重矩阵
                # (1) 求平均向量
                vec_avg = torch.mean(torch.stack(task_vectors), dim=0)

                centered_vecs = [v - vec_avg for v in task_vectors]
                residual_stack = torch.stack(centered_vecs, dim=0)  # (num_tasks, m, n)

                # (3) 合并残差：先取平均
                residual_mean = torch.mean(residual_stack, dim=0)  # (m, n)

                # (4) SVD 截断低秩
                U, s, Vh = torch.linalg.svd(residual_mean, full_matrices=False)
                m, n = residual_mean.shape
                reduced_rank = max(1, int(min(m, n) * rank_ratio))
                U_r = U[:, :reduced_rank]
                s_r = s[:reduced_rank]
                Vh_r = Vh[:reduced_rank, :]

                lowrank_residual = U_r @ torch.diag(s_r) @ Vh_r  # (m, n)

                # (5) 组合 CART 合并结果
                merged_task_vector[layer] = vec_avg + lowrank_residual

            return merged_task_vector

    # @track_stats
    def ace_merging(self, eps = 1e-2, k_frac = 0.3):
        with torch.no_grad():
            merged_task_vector = {}

            for layer in self.merge_layers:
                task_vectors = [tv.task_vector_param_dict[layer] for tv in self.task_vector_list]

                is_1d_layer = any(vec.ndim != 2 or "text_projection" in layer for vec in task_vectors)
                if is_1d_layer:
                    merged_task_vector[layer] = torch.mean(torch.stack(task_vectors), dim=0)
                    continue

                T = len(task_vectors)
                _, in_dim = task_vectors[0].shape

                traces = torch.tensor([torch.trace(W_t.T @ W_t).item() for W_t in task_vectors])
                log_traces = torch.log(traces + 1e-12)

                gamma = torch.var(log_traces) / (torch.mean(log_traces).pow(2) + 1e-12)
                flag = gamma > 0.3
                avg_trace = traces.mean().item()

                Sigmas = []
                WSigma_sum = torch.zeros_like(task_vectors[0], device=self.device)
                Sigma_sum  = torch.zeros((in_dim, in_dim), device=self.device)

                for W_t in task_vectors:
                    W_t = W_t - W_t.mean(dim=0, keepdim=True)
                    Sigma_raw = W_t.T @ W_t
                    tr = torch.trace(Sigma_raw) + 1e-12

                    if flag:
                        Sigma_t = Sigma_raw / tr
                        eps_t = eps / tr
                    else:
                        Sigma_t = Sigma_raw
                        eps_t = eps

                    Sigmas.append(Sigma_t)
                    Sigma_t = Sigma_t + eps_t * torch.eye(in_dim, device=self.device)

                    WSigma_sum += W_t @ Sigma_t
                    Sigma_sum  += Sigma_t

                C_agg = torch.mean(sum(Sigmas), dim=0, keepdim=True)

                if flag:
                    C_agg = C_agg / (avg_trace + 1e-12)
                A = Sigma_sum + C_agg
                B = WSigma_sum

                try:
                    A_inv = torch.linalg.inv(A)
                except RuntimeError:
                    A_inv = torch.linalg.pinv(A)
                W_0 = B @ A_inv
                merging_vector = W_0
                if flag:
                    Sigma_mean = Sigma_sum / T
                    Delta_Res = torch.zeros_like(task_vectors[0], device=self.device)
                    for W_t, S_t in zip(task_vectors, Sigmas):
                        Delta_Res += W_t @ (S_t - Sigma_mean)

                    Delta_Fused = Delta_Res + W_0
                    U, S, Vh = torch.linalg.svd(Delta_Fused, full_matrices=False)

                    r = S.numel()
                    k = int(r * k_frac)
                    S_fused_k = S[:k]
                    sigma_iso = S_fused_k.mean()

                    U_k = U[:, :k]
                    V_k = Vh[:k, :].T
                    merging_vector += sigma_iso * (U_k @ V_k.T)
                merged_task_vector[layer] = merging_vector
            return merged_task_vector

    def emr_merging(self, idx):
        merged_task_vector = {}
        sum_param = {}
        n2p = []
        nums = len(self.task_vector_list)
        for m in range(nums):
            n2p_temp = self.task_vector_list[m].task_vector_param_dict
            n2p.append(n2p_temp)
            for n in n2p_temp:
                if n not in sum_param:
                    sum_param[n] = []
                sum_param[n].append(n2p_temp[n])
        sum_param = {k: torch.stack(v, 0).mean(0) for k, v in sum_param.items()}
        Vector_unified = {}
        masks = {}
        scales = torch.zeros(nums).to(self.device)
        with torch.no_grad():
            for n in sum_param:
                masks[n] = []
                flag = (sum_param[n] > 0) * 2 - 1
                param_max = torch.zeros_like(n2p[0][n])
                for m in range(nums):
                    param = self.task_vector_list[m].task_vector_param_dict[n]
                    mask = (param * flag) > 0
                    masks[n].append(mask)
                    param_abs = torch.abs(mask * param)
                    param_max = torch.where(param_abs > param_max, param_abs, param_max)
                    scales[m] += torch.mean(torch.abs(param))
                Vector_unified[n] = param_max * flag

            new_scales = torch.zeros(nums).to(self.device)
            for m in range(nums):
                for n in Vector_unified:
                    p = Vector_unified[n] * masks[n][m]
                    new_scales[m] += torch.mean(torch.abs(p))
            rescales = scales / new_scales

        for n in Vector_unified:
            merged_task_vector[n] = Vector_unified[n] * masks[n][idx] * rescales[idx]

        return merged_task_vector