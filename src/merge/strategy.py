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

    def pca_merging(self, low_freq_ratio=0.1): # 参数名我们暂时保留，但现在它的意义是"keep_ratio"
        """
        (新功能) 使用小波变换(DWT)进行幅度剪枝，以验证其能量集中性。
        """
        filtered_task_vector = {}
        task_vector_to_process = self.task_vector_list[0].task_vector_param_dict
        
        # 选择一个小波基函数，'haar'是最简单的，'db4' (Daubechies 4) 是一个常用的好选择
        wavelet = 'db4'

        for layer in self.merge_layers:
            param = task_vector_to_process[layer]
            
            # --- 我们主要处理2D的权重矩阵，因为这是小波变换最擅长的领域 ---
            if param.ndim == 2:
                # --- 步骤 1: 将PyTorch张量转为NumPy数组 ---
                param_np = param.cpu().numpy()

                # --- 步骤 2: 执行多层2D小波变换 ---
                # 'wavedec2' 会进行多层分解，并以一种方便的格式返回所有系数
                # level=None 会自动选择最大可能的分解层数
                coeffs_list = pywt.wavedec2(param_np, wavelet=wavelet, level=None)

                # --- 步骤 3: 将系数列表转换为单一的向量，以便进行幅度剪枝 ---
                # 'coeffs_to_array' 会将复杂的系数结构平铺成一个NumPy数组
                coeffs_arr, coeff_slices = pywt.coeffs_to_array(coeffs_list)
                
                # --- 步骤 4: 在小波域执行幅度剪枝 (与您之前的逻辑完全相同) ---
                flat_coeffs = coeffs_arr.flatten()
                num_to_keep = int(flat_coeffs.size * low_freq_ratio)

                if num_to_keep > 0:
                    # 找到阈值
                    threshold = np.partition(np.abs(flat_coeffs), -num_to_keep)[-num_to_keep]
                    
                    # 创建并应用掩码
                    mask = np.abs(coeffs_arr) >= threshold
                    pruned_coeffs_arr = coeffs_arr * mask
                else:
                    pruned_coeffs_arr = np.zeros_like(coeffs_arr)
                
                # --- 步骤 5: 将剪枝后的系数向量恢复成小波系数的列表结构 ---
                pruned_coeffs_list = pywt.array_to_coeffs(pruned_coeffs_arr, coeff_slices, output_format='wavedec2')
                
                # --- 步骤 6: 执行逆小波变换 ---
                pruned_param_np = pywt.waverec2(pruned_coeffs_list, wavelet=wavelet)
                
                # --- 步骤 7: 将结果转回PyTorch张量 ---
                # 需要确保形状与原始输入完全一致
                H, W = param.shape
                pruned_delta_theta = torch.from_numpy(pruned_param_np[:H, :W]).to(param.device, dtype=param.dtype)
                
                filtered_task_vector[layer] = pruned_delta_theta
            
            else:
                # 对于非2D张量，我们暂时不做处理，直接使用平均值（或保留原样）
                # 这是因为1D小波变换和2D在结构上差异较大，分开处理更清晰
                # 这里我们假设只分析2D层，其他层直接保留原样
                if self.task_vector_list:
                   filtered_task_vector[layer] = self.task_vector_list[0].task_vector_param_dict[layer]
                else:
                   filtered_task_vector[layer] = torch.zeros_like(param) # or some other default
        
        return filtered_task_vector

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


    # def wudi_merging(self, iter_num=300):
    #     merged_task_vector = {}
    #     for layer in self.merge_layers:
    #         original_device = self.device
    #         task_vectors = torch.stack([
    #             tv.task_vector_param_dict[layer].to(self.device) 
    #             for tv in self.task_vector_list
    #         ])
    #         num_tvs = len(task_vectors)
    #         merged_task_vector[layer] = torch.nn.Parameter(torch.sum(task_vectors, dim=0))

    #         if task_vectors[0].ndim == 2 and "text_projection" not in layer:
    #             optimizer = torch.optim.Adam([merged_task_vector[layer]], lr=1e-5, weight_decay=0)
    #             l2_norms = torch.square(
    #                 torch.norm(task_vectors.reshape(task_vectors.shape[0], -1), p=2, dim=-1)
    #             )
    #             for i in range(iter_num):
    #                 disturbing_vectors = merged_task_vector[layer].unsqueeze(0) - task_vectors
    #                 product = torch.matmul(disturbing_vectors, task_vectors.transpose(1, 2))
    #                 loss = torch.sum(
    #                     torch.square(product) / l2_norms.unsqueeze(-1).unsqueeze(-1)
    #                 )
    #                 optimizer.zero_grad()
    #                 loss.backward()
    #                 optimizer.step()
    #         else:
    #             merged_task_vector[layer] = merged_task_vector[layer] / num_tvs
    #         merged_task_vector[layer] = merged_task_vector[layer].to(
    #             device=original_device, non_blocking=True
    #         )
    #     return merged_task_vector
    @track_stats
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
    
    @track_stats
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

    @track_stats
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

    @track_stats
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


    # def compute_task_weights(self, task_vectors: list[torch.Tensor], 
    #                         normalization_method: str = 'softmax',
    #                         temperature: float = 1.0) -> torch.Tensor:
    #     """
    #     计算每个任务的综合重要性权重 w_t。

    #     Args:
    #         task_vectors: 任务向量列表。
    #         normalization_method: 'linear' 或 'softmax'。
    #         temperature: 用于 softmax 的温度参数。

    #     Returns:
    #         torch.Tensor: 包含每个任务最终权重的向量。
    #     """
    #     T = len(task_vectors)
    #     if T <= 1:
    #         return torch.tensor([1.0], device=task_vectors[0].device)

    #     # --- 预计算所有协方差代理 ---
    #     sigma_proxies = []
    #     spectral_entropies = []
    #     for W_t in task_vectors:
    #         # 计算谱熵
    #         try:
    #             U, S, Vh = torch.linalg.svd(W_t, full_matrices=False)
    #             if len(S) > 1:
    #                 p = S**2 / torch.sum(S**2)
    #                 entropy = -torch.sum(p * torch.log(p + 1e-12))
    #                 norm_entropy = entropy / torch.log(torch.tensor(len(S), dtype=torch.float32))
    #                 spectral_entropies.append(norm_entropy)
    #             else:
    #                 spectral_entropies.append(torch.tensor(0.0, device=W_t.device)) # 单一奇异值，熵为0
    #         except torch.linalg.LinAlgError:
    #             spectral_entropies.append(torch.tensor(1.0, device=W_t.device)) # SVD失败，视为最大熵/最低质量

    #         # 计算协方差代理
    #         W_t_centered = W_t - W_t.mean(dim=0, keepdim=True)
    #         S_t = W_t_centered.T @ W_t_centered
    #         trace = S_t.trace()
    #         if trace > 1e-12:
    #             sigma_proxies.append(S_t / trace)
    #         else:
    #             sigma_proxies.append(torch.zeros_like(S_t))

    #     # --- 计算 Quality 和 Uniqueness ---
    #     qualities = 1.0 - torch.stack(spectral_entropies)
    #     print(f'qualities: {qualities}')

    #     uniquenesses = torch.zeros(T, device=task_vectors[0].device)
    #     for i in range(T):
    #         total_dissimilarity = 0.0
    #         for j in range(T):
    #             if i == j: continue
    #             sim = F.cosine_similarity(sigma_proxies[i].flatten(), sigma_proxies[j].flatten(), dim=0)
    #             total_dissimilarity += (1.0 - sim)
    #         uniquenesses[i] = total_dissimilarity / (T - 1)

    #     print(f'uniquenesses: {uniquenesses}')
    #     # --- 计算原始权重 ---
    #     w_raw = qualities * uniquenesses

    #     # --- 归一化 ---
    #     if normalization_method == 'softmax':
    #         if temperature > 0:
    #             w_final = T * F.softmax(w_raw / temperature, dim=0)
    #         else: # T=0 -> argmax
    #             w_final = torch.zeros_like(w_raw)
    #             w_final[torch.argmax(w_raw)] = T
    #     else: # linear
    #         w_sum = torch.sum(w_raw)
    #         if w_sum > 1e-12:
    #             w_final = w_raw * (T / w_sum)
    #         else: # 如果所有权重都为0
    #             w_final = torch.ones(T, device=task_vectors[0].device)

    #     return w_final
    
    # def tsvqr(self):

    #     def compute_tsvm(task_vectors: list[torch.Tensor], ratio = 0.6) -> torch.Tensor:
    #         T = len(task_vectors)

    #         pooled_u = []
    #         pooled_s = []
    #         pooled_v = []
            
    #         # 步骤 1: SVD分解并收集最重要的部分
    #         for i, W_t in enumerate(task_vectors):

    #             u, s, v = torch.linalg.svd(W_t, full_matrices=False)

    #             # energies = s ** 2
    #             # total_energy = torch.sum(energies)
    #             # cumulative_energy = torch.cumsum(energies, dim=0)
    #             # E_cum = cumulative_energy / total_energy
    #             # D_cum = torch.cumsum(all_interference_scores[t], dim=0) / torch.sum(all_interference_scores[t])
    #             # balance = E_cum - D_cum
    #             # reduced_index_s = torch.argmax(balance).item() + 1
    #             reduced_index_s = int(1 / T * len(s))

    #             s_norm = s / s.sum()
    #             entropy = -(s_norm * torch.log(s_norm + 1e-12)).sum()
    #             er = torch.exp(entropy)
    #             s_reshaped = s[:reduced_index_s]
    #             truncated_energy = torch.sum(s_reshaped)
    #             # scaling_factor = total_energy / truncated_energy
    #             # # 找到第一个超过阈值的索引
    #             # # torch.searchsorted 效率很高
    #             # reduced_index_s = torch.searchsorted(cumulative_energy, total_energy * ratio, right=True).item()
    #             # print(f'ratio: {reduced_index_s / len(s)}, scaling: {scaling_factor}')

    #             # 根据 ratio 计算要保留的奇异值数量
    #             # reduced_index_s = int(s.shape[0] * ratio)

    #             # 收集最重要的部分
    #             pooled_u.append(u[:, :reduced_index_s])
    #             # pooled_s.append(s[:reduced_index_s] * scaling_vector)
    #             pooled_s.append(s_reshaped)
    #             pooled_v.append(v[:reduced_index_s, :]) # v 是 Vh, 所以是取行


    #         # 步骤 2: 将所有收集到的部分拼接成一个大的“全局池”
    #         sum_u = torch.cat(pooled_u, dim=1)
    #         sum_s = torch.cat(pooled_s, dim=0)
    #         sum_v = torch.cat(pooled_v, dim=0)

    #         # 步骤 3: 对全局池进行SVD，提取最终的共同结构
    #         u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
    #         u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)


    #         return torch.linalg.multi_dot(
    #             (
    #                 u_u,
    #                 v_u,
    #                 torch.diag(sum_s),
    #                 u_v,
    #                 v_v,
    #             )
    #         )

    #     with torch.no_grad():
    #         merged_task_vector = {}

    #         for layer in self.merge_layers:
    #             task_vectors = [tv.task_vector_param_dict[layer] for tv in self.task_vector_list]

    #             is_1d_layer = any(vec.ndim != 2 or "text_projection" in layer for vec in task_vectors)
    #             if is_1d_layer:
    #                 merged_task_vector[layer] = torch.mean(torch.stack(task_vectors), dim=0)
    #                 continue
                
    #             print(f"\n--- Processing Layer: {layer} ---")
    #             T = len(task_vectors)
    #             out_dim, in_dim = task_vectors[0].shape

    #             # print("  - Calculating functional (spectral) similarity...")
    #             # # 你可以选择一个合适的 k 值
    #             # functional_similarity = self.compute_spectral_similarity(task_vectors, k=32)

    #             # # 基于功能相似度设定 alpha
    #             # alpha = 1.0 - functional_similarity

    #             # print(f"  - Functional Similarity: {functional_similarity:.4f}")
    #             # print(f"  - Merge Strategy: alpha={alpha:.2f} ({(alpha*100):.0f}% Diverse + {(1-alpha)*100:.0f}% Common)")


    #             # ================= 步骤 3: 计算 W_diverse (原 tsvqr 逻辑) =================
    #             # 只有当 alpha > 0 时才需要计算
    #             W_diverse = None
    #             print("  - Calculating W_diverse (Conflict Resolution)...")
    #             # ... (这里是你原来的 tsvqr 阶段 1, 2, 3, 4, 5 的全部逻辑) ...
    #             # 阶段1: 预计算与SVD分解
    #             all_svd_results = []
    #             task_norms = []
    #             for W_t in task_vectors:
    #                 U, S, Vh = torch.linalg.svd(W_t, full_matrices=False)
    #                 all_svd_results.append((U, S, Vh))
    #                 task_norms.append(torch.norm(W_t, p='fro') + 1e-12)

    #             # 阶段2: 计算干扰分数
    #             all_interference_scores = []
    #             # ... (你的干扰分数计算代码) ...
    #             for t, (U, S, Vh) in enumerate(all_svd_results):
    #                 V = Vh.T
    #                 num_singular_values = len(S)
    #                 interference_scores_t = torch.zeros(num_singular_values, device=self.device)
    #                 for j in range(T):
    #                     if j == t: continue
    #                     projections = torch.diag(U.T @ task_vectors[j] @ V)
    #                     interference_scores_t += torch.abs(projections / task_norms[j])
    #                 interference_scores_t /= S
    #                 all_interference_scores.append(interference_scores_t)

    #             # 阶段3: 剪枝与重建
    #             reconstructed_task_vectors = []

    #             for t, (U, S, Vh) in enumerate(all_svd_results):
    #                 interference_scores = all_interference_scores[t]
    #                 q1, q3 = torch.quantile(interference_scores, torch.tensor([0.25, 0.75], device=self.device))
    #                 iqr = q3 - q1
    #                 pruning_threshold = q3 + 1.5 * iqr
    #                 prune_idx = torch.where(interference_scores > pruning_threshold)[0]
    #                 mask = torch.ones_like(S, device=self.device)
                        
    #                 # num_to_prune = int(len(S) * 0)
                    
    #                 # if num_to_prune > 0:
    #                 #     # 根据新的 pruning_scores 来找到要剪枝的索引
    #                 #     prune_idx = torch.topk(interference_scores, k=num_to_prune).indices
    #                 # else:
    #                 #     prune_idx = torch.tensor([], device=self.device, dtype=torch.long)
                        
    #                 # mask = torch.ones_like(S, device=self.device)
    #                 # if len(prune_idx) > 0: mask[prune_idx] = 0.0
    #                 S_pruned = S * mask
    #                 W_t_reconstructed = U @ torch.diag(S_pruned) @ Vh
    #                 reconstructed_task_vectors.append(W_t_reconstructed)

    #             # w_final = self.compute_task_weights(task_vectors)
    #             # print(w_final)

    #             # 阶段4 & 5: 合并得到 W_diverse
    #             Sigma_sum = torch.zeros((in_dim, in_dim), device=self.device)
    #             WSigma_sum = torch.zeros((out_dim, in_dim), device=self.device)

    #             C_t_list = [w.T @ w for w in task_vectors]
    #             # C_agg = torch.mean(torch.stack(C_t_list), dim=0)
    #             C_agg = torch.mean(sum(task_vectors), dim=0)
    #             # E_W = torch.mean(torch.stack(task_vectors, dim=0), dim=0)
    #             # print(E_W.shape)
    #             # C_agg = E_W.T @ E_W
    #             # U_sum, S_sum, V_sum = torch.linalg.svd(sum_W, full_matrices = False)
    #             # isoc = U_sum @ torch.diag(torch.ones_like(S_sum) * S_sum.mean()) @ V_sum
    #             # C_agg = isoc.T @ isoc
    #             # W_common = compute_tsvm(reconstructed_task_vectors)

    #             for t, W_t in enumerate(reconstructed_task_vectors):
    #                 U, S, Vh = all_svd_results[t]
    #                 V = Vh.T
    #                 energies = S ** 2
    #                 E_cum = torch.cumsum(energies, dim=0) / torch.sum(energies)
    #                 D_cum = torch.cumsum(all_interference_scores[t], dim=0) / torch.sum(all_interference_scores[t])
    #                 balance = E_cum - D_cum
    #                 r = torch.argmax(balance).item() + 1
    #                 noise_energy = torch.sum(S[r:] ** 2)
    #                 eps = (noise_energy / (in_dim - r + 1e-12)).item() if noise_energy > 0 else 1e-5

    #                 U_r, S_r, V_r = U[:, :r], S[:r], V[:, :r]

    #                 mu_vec = W_t.mean(dim=0)
    #                 temp = U_r @ torch.diag(S_r) @ V_r.T - mu_vec

    #                 # temp = W_t - mu_vec
    #                 Sigma_t = temp.T @ temp + eps * torch.eye(in_dim, device=self.device) + C_agg
    #                 WSigma_sum += W_t @ Sigma_t
    #                 Sigma_sum += Sigma_t
    #             try:
    #                 Sigma_sum_inv = torch.linalg.inv(Sigma_sum)
    #             except RuntimeError:
    #                 Sigma_sum_inv = torch.linalg.pinv(Sigma_sum)
    #             W_diverse = WSigma_sum @ Sigma_sum_inv

    #             merged_task_vector[layer] = W_diverse

    #         return merged_task_vector

    # 分块 SVD
    # def tsvqr(self):
    #     print("Computing SVD (top-k selection)...")
    #     merged_task_vector = {}
    #     with torch.no_grad():

    #         def topk_svd_merge(stacked_tensor):
    #             components = []
    #             num_tasks, m, n = stacked_tensor.shape
    #             for idx in range(num_tasks):
    #                 u, s, v = torch.linalg.svd(stacked_tensor[idx], full_matrices=False)
    #                 for i in range(len(s)):
    #                     components.append((s[i], u[:, i], v[i, :], idx))
    #             components.sort(key=lambda x: x[0], reverse=True)
                
    #             top_k = min(m, n)

    #             tensor_origins = [c[3] for c in components[:top_k]]
    #             print(f"Top {top_k} components origin distribution: {Counter(tensor_origins)}")

    #             U_all = torch.stack([components[i][1] for i in range(top_k)], dim=1)  # (m, top_k)
    #             V_all = torch.stack([components[i][2] for i in range(top_k)], dim=0)  # (top_k, n)
    #             S_top = torch.tensor([components[i][0] for i in range(top_k)],
    #                                 device=self.device, dtype=U_all.dtype)

    #             u_u, _, v_u = torch.linalg.svd(U_all, full_matrices=False)
    #             u_v, _, v_v = torch.linalg.svd(V_all, full_matrices=False)

    #             U_L = u_u @ v_u
    #             V_L = u_v @ v_v

    #             return U_L @ torch.diag(S_top) @ V_L

    #         for layer in self.merge_layers:
    #             vec_list = []
    #             for task_vector in self.task_vector_list:
    #                 vec = task_vector.task_vector_param_dict[layer]
    #                 if vec.ndim == 2 and "text_projection" not in layer:
    #                     vec_list.append(vec)
    #                 elif vec.ndim == 1:
    #                     if layer not in merged_task_vector:
    #                         merged_task_vector[layer] = vec.clone()
    #                     else:
    #                         merged_task_vector[layer] += (vec - merged_task_vector[layer]) / len(self.task_vector_list)

    #             if len(vec_list) == 0:
    #                 continue

    #             stacked_vec = torch.stack(vec_list, dim=0)  # (num_tasks, m, n)

    #             # 定义分块数量
    #             num_row_chunks = 2
    #             num_col_chunks = 1

    #             # 使用torch.chunk进行更鲁棒、更均匀的分块
    #             row_chunks = torch.chunk(stacked_vec, chunks=num_row_chunks, dim=1)
                
    #             merged_blocks_rows = []
    #             for row_block_stack in row_chunks:
    #                 col_chunks = torch.chunk(row_block_stack, chunks=num_col_chunks, dim=2)
    #                 merged_blocks_cols = []
    #                 for block in col_chunks:
    #                     merged_block = topk_svd_merge(block)
    #                     merged_blocks_cols.append(merged_block)
                    
    #                 # 拼接列块
    #                 merged_blocks_rows.append(torch.cat(merged_blocks_cols, dim=1))
                
    #             # 拼接行块
    #             merged_task_vector[layer] = torch.cat(merged_blocks_rows, dim=0)

    #         return merged_task_vector

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