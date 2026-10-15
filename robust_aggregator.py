import numpy as np
import torch
import time
import math 

# ===== RandEigen =====
def power_iteration(mat, iterations):
    dim = mat.shape[0]
    u = torch.randn((dim, 1), device='cuda')
    for _ in range(iterations):
        u = mat @ u / torch.linalg.norm(mat @ u) 
    eigenvalue = u.T @ mat @ u
    return eigenvalue, u

def randomized_agg_forced(data, dirty_worker, eps_poison, eps_jl=0.1, eps_pow=1e-3, device='cuda', seed=12):
    res, wm_workers_filtered, workers_filtered =  _randomized_agg(data, dirty_worker, eps_poison, eps_jl, eps_pow, 1, 10**-5, device, forced=True, seed=seed) # set threshold for convergence as 1*10**-5 (i.e. float point error)
    return res, wm_workers_filtered, workers_filtered

def _randomized_agg(data, dirty_worker, eps_poison, eps_jl, eps_pow, threshold=20, clean_eigen=10**-5, device='cuda', forced=False, seed=None):
    if seed:
        torch.manual_seed(seed)
    
    n = int(data.shape[0])
    data = data.to(device)

    d = int(math.prod(data[0].shape))
    data_flatten = data.reshape(n, d)
    data_mean = torch.mean(data_flatten, dim=0)
    data_sd = torch.std(data_flatten, dim=0)
    data_norm = (data_flatten - data_mean)/data_sd

    k = min(int(math.log(d)//eps_jl**2), d)
    
    A = torch.randn((d, k), device='cuda')
    A = A/(k**0.5)

    Y = data_flatten @ A # n times k
    Y = Y.to(device)
    power_iter_rounds = int(- math.log(4*k)/(2*math.log(1-eps_pow)))
    clean_eigen = clean_eigen * d/k
    old_eigenvalue = None

    # Filtering Metric ###
    wm_workers_filtered = 0
    workers_filtered = 0
    current_mask = torch.ones(n, dtype=torch.bool).to(device)
    # Filtering Metric ###

    if eps_poison <= 0.2: 
        stopping_criteria = (1-5*eps_poison)*n
    else: 
        stopping_criteria = (1-2*eps_poison)*n
        
    for iteration in range(max(int(2*eps_poison*n), 10)):
        Y_mean = torch.mean(Y, dim=0)
        Y = (Y - Y_mean)
        Y_cov = torch.cov(Y.T)
        Y_sq = Y_cov
            
        eigenvalue, eigenvector = power_iteration(Y_sq, power_iter_rounds)

        proj_Y = torch.abs(Y @ eigenvector )
        proj_Y = torch.flatten(proj_Y)
        if forced and old_eigenvalue and abs(old_eigenvalue - eigenvalue) < 10**-5: 
            # print('converge', flush=True)
            break

        if len(Y) <= stopping_criteria: 
            # print('new_criteria', flush=True)
            break 
        old_eigenvalue = eigenvalue
        
        uniform_rand = torch.rand(proj_Y.shape, device='cuda')
        kept_idx = uniform_rand > (proj_Y/torch.max(proj_Y))

        # Filtering Metric ###
        current_to_original = torch.where(current_mask)[0]  
        filtered_original_indices = current_to_original[~kept_idx]
        wm_workers_filtered += torch.sum(filtered_original_indices < dirty_worker).item()
        current_mask[current_to_original[~kept_idx]] = False

        false_count = (~kept_idx).sum().item()
        false_indices = (kept_idx == False).nonzero().squeeze()

        workers_filtered += false_count

        # print(f"Iteration {iteration}: Number of False in kept_idx: {false_count}, {false_indices}", flush=True)
        # Filtering Metric ###

        Y = Y[kept_idx]
        data = data[kept_idx]
    return torch.mean(data, dim=0), wm_workers_filtered, workers_filtered
    
def robust_aggregator(worker_updates, config): 

    unfiltered_updates = {}
    for name, gradients_list in worker_updates.items():
        flattened_gradients = [grad.flatten().to(torch.float32) for grad in gradients_list]
        flattened_tensor = torch.stack(flattened_gradients) 
    
        unfiltered_updates[name] = flattened_tensor

    # Randomized Aggregation 
    print("Randomized Aggregation!", flush=True)
    start_time = time.time()

    filter_grad = {}
    recall = {}
    precision = {}
    for name, p in unfiltered_updates.items():
        torch.cuda.empty_cache()
        if 'embed_out' not in name and 'embed_in' not in name:
            print(f"layer name: {name}, shape:{p.shape}", flush=True)
            filter_grad[name], wm_workers_filtered, workers_filtered = randomized_agg_forced(p, config.DIRTY_WORKER, (config.DIRTY_WORKER / config.N_WORKER), device="cuda")
            recall[name] = wm_workers_filtered / config.DIRTY_WORKER
            precision[name] = wm_workers_filtered / workers_filtered
            print(f"Corrupted Workers Filtered: {wm_workers_filtered}, Workers Filtered: {workers_filtered}", flush=True)
        else:
            # Split into 8 chunks along dimension 1
            print(f"Processing large parameter {name} in chunks, shape:{p.shape}", flush=True)
            chunk_num = 8
            chunks = torch.chunk(p, chunk_num, dim=1)      
            
            processed_chunks = []
            recall_val = 0
            precision_val = 0
            for chunk in chunks:
                torch.cuda.empty_cache()
                partial, wm_workers_filtered, workers_filtered = randomized_agg_forced(chunk, config.DIRTY_WORKER, (config.DIRTY_WORKER / config.N_WORKER), device="cuda")
                recall_val += wm_workers_filtered / config.DIRTY_WORKER
                precision_val += wm_workers_filtered / workers_filtered
                print(f"Corrupted Workers Filtered: {wm_workers_filtered}, Workers Filtered: {workers_filtered}", flush=True)
                processed_chunks.append(partial)

            filter_grad[name] = torch.cat(processed_chunks, dim=0)
            recall[name] = recall_val / chunk_num                                   # Takes Average here 
            precision[name] = precision_val / chunk_num

    end_time = time.time()
    print(f"Time taken (Compute Norm): {end_time - start_time:.4f} seconds", flush=True)

    return filter_grad, recall, precision
