import numpy as np
import pandas as pd
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
import numpy.linalg as linalg
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans
from torch.profiler import profile, record_function, ProfilerActivity
import skgstat as skg
import math
import tqdm
import sys
import itertools
import time
import torch
import random
torch.set_default_dtype(torch.float32)

############################## Size Key ##############################
# C: Number of gridded conditioning data points
# S: Number of points on the grid that need to be simulated 
# B: Number of points that are being processed in current batch
# K: Number of radial segments in NNS (default 8 for octant search)
# X: List of coordinates along the x axis of the full grid
# Y: List of coordinates alogn the y axis of the full grid 


def skrige_sgs(x, y, xy_cond, data_cond, cond_mask, index_map, xy_sim, sim_mask, vario_sim, num_nn, bb_size, num_gpus, multiplier):
    '''
    Launch function for gpu-accelerated Simple Kriging with multiprocessing and batch (vectorized) code

    Parameters:
    - xy_cond: tensor(C, 2), conditioning data locations
    - data_cond: tensor(C,), elevation values that coorespond to conditioning data locations
    - xy_sim: tensor(S, 2), locations to be simulated
    - vario_sim: tensor(S, 6), variogram params for each location to be simulated 
                                [azimuth, nugget, major_range, minor_range, sill, smooth]
    - num_nn: int, maximum number of nearest neigbors
    - radius: int, nearest neigbor search radius
    - num_gpus: int, number of available gpus (torch.cuda.device_count())
    - multiplier: int, multiplier to increase amount of processes and divide workload (# processes = num_gpus * multiplier)

    Returns:
    - sgs: tensor(C+S,), elevation value for each location in sorted grid
    '''
    preprocess_start = time.time()

    # Shuffle data to predict to create a random path
    index = torch.arange(len(xy_sim)) 
    shuffle = index[torch.randperm(len(xy_sim))]
    xy_sim_shuffled = xy_sim[shuffle]
    vario_sim_shuffled = vario_sim[shuffle]

    # full[C+S, 2]: Create a tensor that has all grid locations with the conditioning data at the beginning 
    full = torch.vstack((xy_cond, xy_sim_shuffled))

    # Create starting index that marks the beginning of the points to predict
    begin = len(xy_cond)

    # Determine how many points/grid-cells each process will predict 
    num_cells = len(xy_sim)
    processes = num_gpus * multiplier
    cells_per_process = num_cells//processes
    
    proc_id = [i for i in range(processes)]

    # Create list of index lists such that each list is for a single process
    # note sublist length is B in docstrings
    i_list = [[i for i in range(j*cells_per_process, (j+1)*cells_per_process)] for j in range(processes-1)]
    i_list.append([i for i in range((processes-1)*cells_per_process,num_cells)])
    
    # Create list of variogram parameter tensors that cooresponds to the simulation partition
    vario_list = [vario_sim_shuffled[j*cells_per_process: (j+1)*cells_per_process] for j in range(processes-1)]
    vario_list.append(vario_sim_shuffled[(processes-1)*cells_per_process: num_cells])

    
    # Gather the rest of the arguments to be sent to the function executed in parallel
    gpu_num = [i % num_gpus for i in range(processes)]
    args = zip(proc_id, i_list, vario_list, gpu_num, itertools.cycle([cond_mask]), itertools.cycle([sim_mask]),
               itertools.cycle([index_map]), itertools.cycle([x]), itertools.cycle([y]), itertools.cycle([shuffle]),
               itertools.cycle([full]), itertools.cycle([num_nn]), itertools.cycle([begin]), itertools.cycle([bb_size]))
    
    preprocess_end = time.time()
    
    print(f"Time to prepare data before parallel execution: {preprocess_end-preprocess_start}s")
    
    parallel_start = time.time()

    # use multiprocessing library to execute parallel_skring_weights function in parallel
    mp.set_start_method('spawn', force=True)
    with mp.Pool(processes=1) as pool:
        
        out = pool.starmap(parallel_skring_weights, tqdm.tqdm(args, total=len(proc_id)))
    
    # aggregate kriging weight data into a dictionary to look up data by index
    kr_dictionary = torch.zeros((3, num_cells, num_nn)).cuda(0)
    size_list = torch.zeros(num_cells).cuda(0)
    
    for cur_proc_id, kr_dict_subset, cur_size_list in out:
        
        begin_i = i_list[cur_proc_id][0]
        end_i = i_list[cur_proc_id][-1]+1

        kr_dictionary[:, begin_i:end_i, :] = kr_dict_subset.cuda(0)
        size_list[begin_i:end_i] = cur_size_list.cuda(0)
        
        
    parallel_end = time.time()
    
    print(f"Time for parallel kriging weight calculations: {parallel_end - parallel_start}")

    prediction_start = time.time()

    # Use kriging dictionary to sequentially predict elevation values for each unknown location
    sgs = pred_Z(kr_dictionary, size_list, full, data_cond, vario_sim_shuffled, 's')
    
    prediction_end = time.time()
    
    print(f"Time for serial prediction calculation: {prediction_end - prediction_start}")

    # Sort output to match up with original prediction grid
    sgs = sgs.cpu()
    sort = np.lexsort((sgs[:,0], -sgs[:,1]))
    sgs = sgs[sort]

    return sgs


def parallel_skring_weights(proc_id, i_list, batch_vario, gpu_id, cond_mask, sim_mask, index_map, x, y, shuffle, full, num_nn, begin, bb_size):
    '''
    Function with vectorized code to compute the (simple) Kriging weights for a batch of simulation locations

    Parameters:
    - proc_id: int, process number to corroborate results
    - i_list: list(B), index list to create a slice from full to get current batch locations
    - batch_vario: tensor(B, 6), variogram parameters that coorespond to current batch 
                                [azimuth, nugget, major_range, minor_range, sill, smooth]
    - gpu_id: int, number to indicate which device should be used to process current batch
    - full: tensor(C+S, 2), all grid locations with the conditioning data at the beginning 
    - radius: int, nearest neigbor search radius
    - num_nn: int, maximum number of nearest neigbors
    - num_gpus: int, number of available gpus (torch.cuda.device_count())
    - begin: int, index of full tensor that marks the start of locations to simulate

    Returns:
    - proc_id: int, process number to corroborate results
    - kr_dict_subset: tensor(3, B, num_nn), covariance array, NN inidicies, and kriging weights for batch locations
    - size_list: tensor(B,), Actual number of NN for each batch location (could be =< num_nn)
    '''
    # Assign the GPU
    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'
    
    # Memory management?
    torch.set_num_threads(1)
    torch.cuda.empty_cache()
    
    # Send data to GPU 
    full = full.cuda()
    batch_vario = batch_vario.cuda()
    cond_mask = cond_mask.cuda()
    index_map = index_map.cuda()
    x = x.cuda()
    y = y.cuda()
    
    # Create index list of grid cells to compute
    offset = torch.tensor([i+begin for i in i_list]).cuda()
    
    # Get the size of the search matrix 
    N = int(offset[-1])
    
    # Get number of simulation locations being processed
    B = len(i_list)

    # Create batch location tensor of grid cells to compute
    loc = full[offset]
    
    # Update conditioning and index map to reflect locations that were processed in prior batches
    new_cond_mask, new_index_map = update_masks(x, y, cond_mask, sim_mask, index_map, shuffle, begin, i_list)
    
    # Get reduced list of NN condidates by taking a square segment of the full grid 
    potential_nn, potential_indicies = reduce_nn_seach_candidates(x, y, full, B, new_cond_mask, new_index_map, begin+i_list[0], bb_size)
        
    # Perform sorting and initial preperation to collect nearest neighbors
    stack, indicies, bins, bin_indices, oct_count = preprocess_for_nn_search(potential_nn, potential_indicies, loc, num_nn)
    
    # Go through sorted tensor and collect the k nearest neighbors and their inidices
    batch_near, batch_indicies = nn_search_vectorized(stack, indicies, bin_indices, bins, num_nn, oct_count)

    # Create a rotation matrix for each batch simulation location wrt their variogram parameters
    batch_rot_mat = make_batch_rotation_matrix(batch_vario[:,0], batch_vario[:,2], batch_vario[:,3])
    
    # Load lookup tables to calculate matern covariance
    kv_lookup = torch.load("kv_lookup.pt", weights_only=False).cuda()
    gamma_lookup = torch.load("gamma_lookup.pt", weights_only=False).cuda()
    
    # Calcuate covariance matrix
    covariance_matrix = make_covariance_matrix(batch_near, batch_vario, batch_rot_mat, kv_lookup, gamma_lookup)

    # Calculate covariance between data and unknown
    covariance_array = make_covariance_array(
                    batch_near, loc.unsqueeze(1).repeat(1, num_nn, 1), 
                    batch_vario, batch_rot_mat, kv_lookup, gamma_lookup
                )

    # Reorder covariance matrix and array so NANs appear at the end
    sorted_cov_matrix, sorted_cov_array, sorted_indices = reorder(covariance_matrix,
                                                                    covariance_array,
                                                                    batch_indicies, num_nn)

    # Get size list of number of NN collected for future indexing
    size_list = torch.sum((~torch.isnan(sorted_cov_array)).int(), dim=1)

    # Solve system defined by batch covariance matrix and array to get kriging weights
    k_weights = solve_system(sorted_cov_matrix, sorted_cov_array, num_nn)
    
    # Store results in a batch "dictionary" tensor    
    kr_dict_subset = torch.stack((sorted_cov_array, sorted_indices, k_weights))
    
    return proc_id, kr_dict_subset, size_list


def update_masks(x, y, cond_mask, sim_mask, index_map, shuffle, begin, i_list):
    '''
    Updates conditioning data mask and index map to reflect all the locations that were processes in previous batches 

    Parameters:
    - x: tensor(X,), grid locations along the x-axis
    - y: tensor(Y,), grid locations along the y-axis
    - cond_mask: tensor(Y, X), binary mask where 1 denotes locations of original conditioning data 
    - sim_mask: tensor(Y, X), binary mask where 1 denotes locations that need to be simulated  
    - index_map: tensor(Y, X), mask of conditing data where non NAN values coorespond to the index location in full 
    - shuffle: tensor(S,), tensor used to shuffle simulation data locations to create random path
    - begin: int, starting index that marks the beginning of the points to simulate
    - i_list: list(B), index list to create a slice from full to get current batch locations

    Returns:
    - new_cond_mask: tensor(Y, X), binary mask where 1 denotes locations of original conditioning data as well as 
                                simulation locations processed in previous batches
    - new_index_map: tensor(Y, X), mask of conditing data and previously processed locations where non NAN values 
                                coorespond to the index location in full 
    '''

    # Create index map in same shape as original grid and gather simulation index coordinates
    ii, jj = torch.meshgrid(torch.arange(x.shape[0]), torch.arange(y.shape[0]), indexing='ij')
    ii_sim = ii[sim_mask]
    jj_sim = jj[sim_mask]
    ij_sim = torch.cat((ii_sim.unsqueeze(1), jj_sim.unsqueeze(1)), dim=1).to(torch.int32)
    
    # Shuffle simulation index coordates in same way the true coordinates were shuffled
    ij_sim_shuffled = ij_sim[shuffle]

    new_cond_mask = cond_mask.detach().clone()
    new_index_map = index_map.detach().clone()
    
    # Note first value in i list is the number of locations that were processed in prior batch(es)
    for i in range(i_list[0]):

        i_sim, j_sim = ij_sim_shuffled[i].tolist()

        # Add point and index to conditioning data
        new_cond_mask[i_sim, j_sim] = True
        new_index_map[i_sim, j_sim] = begin + i
        
    return new_cond_mask, new_index_map


def reduce_nn_seach_candidates(x, y, full, B, cond_mask, index_map, begin_batch, bb_size):
    '''
    Reduces search candidates for NNS by taking a square cut out of size (bb_size)x(bb_size) that surrounds each
    simulation location in the current batch and creates a list of potential NN that reside in the square 

    Parameters:
    - x: tensor(X,), grid locations along the x-axis
    - y: tensor(Y,), grid locations along the y-axis
    - full: tensor(C+S, 2), all grid locations with the conditioning data at the beginning 
    - B: int, batch size
    - cond_mask: tensor(Y, X), binary mask where 1 denotes locations of original conditioning data as well as 
                                simulation locations processed in previous batches
    - index_map: tensor(Y, X), mask of conditing data and previously processed locations where non NAN values 
                                coorespond to the index location in full 
    - begin_batch: int, starting index that marks the beginning of the points to simulate in the current batch
    - bb_size: int, size of square cut out (larger bb_size requires more memory in NNS)

    Returns:
    - potential_NN: tensor(B, bb_size^2, 2), list of potential NN locations for each location in current batch
    - potential_indicies: tensor(B, bb_size^2), list of potential NN indicies for each location in current batch
    '''
    
    xx, yy = torch.meshgrid(x,y, indexing='ij')
    
    # Mask out locations not part of conditioning data
    nan_cond_mask = torch.where(cond_mask, 1, float('nan'))
    xx_cond = xx * nan_cond_mask
    yy_cond = yy * nan_cond_mask
    
    potential_NN = torch.full((B, bb_size ** 2, 2), float('nan')).cuda()
    potential_indicies = torch.full((B, bb_size ** 2), float('nan')).cuda()
    
    vert_max = xx.shape[0]
    horiz_max = xx.shape[1]
    loop_iter = 0
    sim_offset = begin_batch
    
    for i in range(B):
        
        x_sim, y_sim = full[i+sim_offset].tolist()
        
        # Find grid index of current location being processed
        i_sim = int((x == x_sim).nonzero(as_tuple=False)[0])
        j_sim = int((y == y_sim).nonzero(as_tuple=False)[0])
        
        pad_size = int(bb_size / 2)
        num_potential_NN = 0
        
        while num_potential_NN < 8:

            # Create index bounds for bounding box centered on current location
            nb = i_sim - pad_size; sb = i_sim + pad_size
            wb = j_sim - pad_size; eb = j_sim + pad_size

            # Edge correction 
            if(nb < 0):
                sb = sb - nb
                nb = 0 
            elif(sb >= vert_max):
                nb = nb - (sb - vert_max - 1)
                sb = vert_max - 1
            elif(wb < 0):
                eb = eb - wb
                wb = 0
            elif(eb >= horiz_max):
                wb = wb - (eb - horiz_max - 1)
                eb = horiz_max - 1

            # Square slice to retrieve potential NN locations and indicies
            x_locs = xx_cond[nb:sb, wb:eb]
            y_locs = yy_cond[nb:sb, wb:eb]
            indicies = index_map[nb:sb, wb:eb]

            # Check to see how many potential NN exist withing bounding box
            num_potential_NN = torch.count_nonzero(~torch.isnan(x_locs.flatten()))

            # Increase pad size if while loop condition not satisfied 
            pad_size = pad_size * 2
            
        # Add point and index to conditioning data
        xx_cond[i_sim, j_sim] = x_sim
        yy_cond[i_sim, j_sim] = y_sim
        index_map[i_sim, j_sim] = i + sim_offset
        
        x_NN = x_locs[~torch.isnan(x_locs)]
        y_NN = y_locs[~torch.isnan(x_locs)]
        indicies_NN = indicies[~torch.isnan(x_locs)]

        # Correction if bounding box size was increased due to insuficient number of neighbors 
        if num_potential_NN > (bb_size ** 2):
            num_potential_NN = (bb_size ** 2)

        potential_NN[i, :num_potential_NN] = torch.cat((x_NN.unsqueeze(1), y_NN.unsqueeze(1)), dim=1)[:num_potential_NN]
        potential_indicies[i, :num_potential_NN] = indicies_NN[:num_potential_NN]
        
    return potential_NN, potential_indicies
    


def preprocess_for_nn_search(search_candidates, indicies, loc, num_nn):
    '''
    Prepare data for nearest-neighbor search with binning.

    Parameters:
    - search_candidates: tensor(B, bb_size^2, 2), potential NN locations wrt each batch location 
    - indicies: tensor(B, bb_size^2), list of potential NN indicies that is 1-1 with search_candidates
    - loc: tensor(B, 2), all the current batch locations
    - num_nn: int, maximum number of nearest neigbors

    Returns:
    - stack: (B, bb_size^2, 4), where each potential nearest neighbor has [x, y, distance, angle]
    - indices: tensor(B, bb_size^2), index of each potential nearest neighbor 
    - bins: list(K+1,), bin edges
    - bin_indices: tensor(B, bb_size^2), bin assignment for each point (1 to K)
    - oct_count: int, max number of closest neighbors to select from each bin
    '''
    B, N, _ = search_candidates.shape

    # Repeat loc to align shapes for distance computation
    locx = loc[:, 0].unsqueeze(1).repeat(1, N)
    locy = loc[:, 1].unsqueeze(1).repeat(1, N)

    # Extract x/y coordinates
    x_tensor = search_candidates[:, :, 0]
    y_tensor = search_candidates[:, :, 1]

    # Compute distance and angle from loc
    centered_x = x_tensor - locx
    centered_y = y_tensor - locy
    distances = torch.sqrt(centered_x**2 + centered_y**2)
    angles = torch.atan2(centered_y, centered_x)

    # Stack into (B, N, 4): x, y, dist, angle
    stack = torch.stack((x_tensor, y_tensor, distances, angles), dim=2)

    # Sort by distance
    sorted_dist_idxs = torch.argsort(stack[..., 2], dim=1)
    stack_idxs = torch.arange(B, device=stack.device).repeat_interleave(N).reshape(B, N)

    # Apply sorting
    stack = stack.gather(1, sorted_dist_idxs.unsqueeze(-1).expand(-1, -1, 4))
    indicies = indicies.gather(1, sorted_dist_idxs)

    # Define 8 bins over angle range
    bins = torch.tensor([
        -math.pi, -3*math.pi/4, -math.pi/2, -math.pi/4, 0,
         math.pi/4, math.pi/2,  3*math.pi/4, math.pi
    ], device=stack.device)

    # Bin index based on angle (angle at index 3)
    bin_indices = torch.bucketize(stack[..., 3], bins, right=False)

    # Octant point count
    oct_count = num_nn // 8

    return stack, indicies, bins, bin_indices, oct_count


def nn_search_vectorized(stack, indices, bin_indices, bins, num_nn, oct_count):
    '''
    Vectorized nearest-neighbor search with binning.

    Parameters:
    - stack: tensor(B, bb_size^2, 4), where each potential nearest neighbor has [x, y, distance, angle]
    - indices: tensor(B, bb_size^2), index of each potential nearest neighbor 
    - bin_indices: tensor(B, bb_size^2), bin assignment for each point (1 to K)
    - bins: list(K+1,), bin edges
    - num_nn: int, total points to select per batch (K * oct_count)
    - oct_count: int, max number of closest neighbors to select from each bin

    Returns:
    - batch_near: tensor(B, num_nn, 2), selected coordinates
    - index_list: tensor(B, num_nn), selected indices
    '''
    B, N, _ = stack.shape
    K = bins.shape[0] - 1  # number of bins

    # === Step 1: Create 3D bin mask ===
    # bin_mask[b, n, k] = 1 if point n in batch b is in bin k+1
    bin_mask = (bin_indices.unsqueeze(-1) == torch.arange(1, K+1, device=stack.device)).float()  # (B, N, K)
    
    # === Step 2: Apply mask to (x,y) and index values ===
    nan_mask = bin_mask.masked_fill(bin_mask == 0, float('nan'))  # Replace non-bin entries with NaN

    # Coordinates: (B, N, K, 2)
    masked_xy = stack[..., :2].unsqueeze(2) * nan_mask.unsqueeze(-1)
    
    # Indices: (B, N, K)
    masked_idx = indices.unsqueeze(2) * nan_mask

    # === Step 3: Count valid points per bin and clamp to oct_count ===
    is_valid = ~torch.isnan(masked_idx)
    bin_counts = is_valid.sum(dim=1)  # (B, K)
    clamped_counts = torch.clamp(bin_counts, max=oct_count)  # (B, K)

    # === Step 4: Sort distances inside bins ===
    masked_dist = stack[..., 2].unsqueeze(2) * nan_mask  # (B, N, K)
    sorted_dist, sorted_idx = torch.sort(masked_dist, dim=1)  # (B, N, K)
    topk_idx = sorted_idx[:, :oct_count, :]  # (B, oct_count, K)

    # === Step 5: Gather top-k points and indices ===
    b_idx = torch.arange(B, device=stack.device).view(B, 1, 1).expand(B, oct_count, K)
    k_idx = torch.arange(K, device=stack.device).view(1, 1, K).expand(B, oct_count, K)

    topk_xy = masked_xy[b_idx, topk_idx, k_idx, :]   # (B, oct_count, K, 2)
    topk_ids = masked_idx[b_idx, topk_idx, k_idx]    # (B, oct_count, K)

    # === Step 6: Mask out unused slots if bin has < oct_count points ===
    topk_range = torch.arange(oct_count, device=stack.device).view(1, -1, 1)
    valid_topk_mask = (topk_range < clamped_counts.unsqueeze(1)).float()  # (B, oct_count, K)

    topk_xy *= valid_topk_mask.unsqueeze(-1)
    topk_ids *= valid_topk_mask

    # === Step 7: Reshape results ===
    batch_near = topk_xy.permute(0, 2, 1, 3).reshape(B, num_nn, 2)
    index_list = topk_ids.permute(0, 2, 1).reshape(B, num_nn)

    return batch_near, index_list


def make_batch_rotation_matrix(azimuth, major_range, minor_range):
    '''
    Vectorized nearest-neighbor search with binning.

    Parameters:
    - azimuth: tensor(B,), azimuth for each batch simulation location
    - major_range: tensor(B,), major_range for each batch simulation location
    - minor_range: tensor(B,), minor_range for each batch simulation location

    Returns:
    - rotation_matrix: tensor(B, 2, 2), rotation_matrix for each batch simulation location
    '''
    theta = (azimuth / 180.0) * math.pi
    
    rotation_matrix = torch.zeros((len(theta), 2, 2), dtype=torch.float64).cuda()
    scaling_matrix = torch.zeros((len(theta), 2, 2), dtype=torch.float64).cuda()
    
    rotation_matrix[:,0,0] = torch.cos(theta)
    rotation_matrix[:,0,1] = -torch.sin(theta)
    rotation_matrix[:,1,0] = torch.sin(theta)
    rotation_matrix[:,1,1] = torch.cos(theta)
    
    scaling_matrix[:,0,0] = 1 / major_range
    scaling_matrix[:,1,1] = 1 / minor_range
    
    rotation_matrix = torch.matmul(rotation_matrix, scaling_matrix)
    
    return rotation_matrix


# def exponential_covariance(effective_lag, sill, nug):
    
#     return (sill - nug)*torch.exp(-3 * effective_lag)


def matern_covariance(effective_lag, sill, nug, s, kv_lookup, gamma_lookup):
    """
    Calculate matern covariance for covariance matrix or array

    Parameters:
    - effective_lag : tensor(B, num_nn, num_nn)|(B, num_nn), effective lag between nn and nn (matrix) or nn and data (array)
    - sill : tensor(B, 1, 1)|(B, 1), sill for each batch simulation location
    - nug : tensor(B, 1, 1)|(B, 1), nugget for each batch simulation location
    - s : tensor(B, 1, 1)|(B, 1), smoothness for each batch simulation location
    - kv_lookup : tensor(601, 100001), lookup table for scipy.special.kv for input [0:6,0.01][0:10,.0001]
    - gamma_lookup : tensor(601), lookup table for scipy.special.gamma for input [0:6,0.01]

    Returns:
    - c : tensor(B, num_nn, num_nn)|(B, num_nn), matern covariance matrix or array
    """
    
    scale = 0.45246434*torch.exp(-0.70449189*s)+1.7863836
    
    effective_lag[effective_lag==0.0] = 1e-8
    
    # Save original nan positions
    nan_mask = torch.where(~torch.isnan(effective_lag), 1, torch.nan)
    
    # Calculate index in lookup table that cooresonds to scipy function parameter
    v_index = (torch.round(s, decimals=2) * 100).int()
    z_temp = 2*scale*effective_lag*torch.sqrt(s)
    z_temp[torch.isnan(z_temp)] = 0
    z_index = (torch.round(z_temp, decimals=4) * 10_000).int()
    z_index = torch.where(z_index <= 10_000, z_index, 10_000)

    c = (sill-nug)*2/gamma_lookup[v_index]*torch.pow(scale*effective_lag*torch.sqrt(s), s)*kv_lookup[v_index, z_index]
    
    c = torch.where(torch.isnan(c), sill-nug, c)
    
    # Replace nan positions
    c = c * nan_mask 
    
    return c


def make_covariance_matrix(batch_near, batch_vario, batch_rot_mat, kv_lookup, gamma_lookup):
    """
    Make covariance matrix showing covariances between each pair of input coordinates

    Parameters:
    - batch_near : tensor(B, num_nn, 2), nn location for each batch simulation location
    - batch_vario : tensor(B, 6), variogram parameters for each batch simulation location
    - batch_rot_mat : tensor(B, 2, 2), rotation matrix for each batch simulation location
    - kv_lookup : tensor(601, 100001), lookup table for scipy.special.kv for input [0:6,0.01][0:10,.0001]
    - gamma_lookup : tensor(601), lookup table for scipy.special.gamma for input [0:6,0.01]

    Returns:
    - covariance_matrix : tensor(B, num_nn, num_nn), matricies of covariance between nn locations
    """
    
    # Unpack variogram parameters
    nug = batch_vario[:,1].view(-1,1,1)
    sill = batch_vario[:,4].view(-1,1,1)
    s = batch_vario[:,5].view(-1,1,1)
    
    batch_near = batch_near.to(dtype=torch.float64)
    mat = torch.matmul(batch_near, batch_rot_mat)
    effective_lag = torch.cdist(mat, mat, p=2)  # Compute pairwise distances
    
    covariance_matrix = matern_covariance(effective_lag, sill, nug, s, kv_lookup, gamma_lookup)

    return covariance_matrix


def make_covariance_array(batch_near, loc_repeated, batch_vario, rotation_matrix, kv_lookup, gamma_lookup):
    """
    Make covariance arrays showing covariances between each simulation location and their respective nn

    Parameters:
    - batch_near : tensor(B, num_nn, 2), nn location for each batch simulation location
    - loc_repeated : tensor(B, num_nn, 2), each batch simulation location repeated to match dims of batch_near
    - batch_vario : tensor(B, 6), variogram parameters for each batch simulation location
    - batch_rot_mat : tensor(B, 2, 2), rotation matrix for each batch simulation location
    - kv_lookup : tensor(601, 100001), lookup table for scipy.special.kv for input [0:6,0.01][0:10,.0001]
    - gamma_lookup : tensor(601), lookup table for scipy.special.gamma for input [0:6,0.01]

    Returns:
    - covariance_array : tensor(B, num_nn) arrays of covariance between nn and batch simulation locations
    """

    # Unpack variogram parameters
    nug = batch_vario[:,1].view(-1,1)
    sill = batch_vario[:,4].view(-1,1)
    s = batch_vario[:,5].view(-1,1)

    batch_near = batch_near.to(dtype=torch.float64)
    loc_repeated = loc_repeated.to(dtype=torch.float64)
    mat1 = torch.matmul(batch_near, rotation_matrix)
    mat2 = torch.matmul(loc_repeated, rotation_matrix)
    effective_lag = torch.sqrt(torch.sum((mat1 - mat2).pow(2), dim=2))
    
    covariance_array = matern_covariance(effective_lag, sill, nug, s, kv_lookup, gamma_lookup)

    return covariance_array


def reorder(covariance_matrix, covariance_array, batch_indicies, num_nn):
    """
    Make covariance arrays showing covariances between each simulation location and their respective nn

    Parameters:
    - covariance_matrix : tensor(B, num_nn, num_nn), matricies of covariance between nn locations
    - covariance_array : tensor(B, num_nn) arrays of covariance between nn and batch simulation locations
    - batch_indicies: tensor(B, num_nn), indices of nn for each batch simulation location
    - num_nn: int, maximum number of nearest neigbors

    Returns:
    - sorted_cov_matrix : tensor(B, num_nn, num_nn), sorted matricies of covariance between nn locations
    - sorted_cov_array : tensor(B, num_nn) sorted arrays of covariance between nn and batch simulation locations
    - sorted_batch_indicies: tensor(B, num_nn), sorted indices of nn for each batch simulation location
    """
    
    # Get mask to sort batch array and matrix so that nan are at the end
    nan_mask = torch.isnan(covariance_array)
    num_mask = ~torch.isnan(covariance_array)
    nan_indices = torch.nonzero(nan_mask)
    num_indices = torch.nonzero(num_mask)
    split_indices = torch.cat((num_indices, nan_indices))
    reorder_indices = split_indices[split_indices[:, 0].sort()[1]]
    reorder_indices = reorder_indices[:,1].reshape(covariance_array.shape)

    sorted_cov_array = covariance_array.gather(1, reorder_indices)
    sorted_batch_indicies = batch_indicies.gather(1, reorder_indices)

    reorder_rows = reorder_indices.unsqueeze(-1).repeat(1,1,num_nn)
    reoreder_cols = reorder_indices.unsqueeze(1).repeat(1,num_nn,1)

    sorted_cov_matrix = covariance_matrix.gather(1,reorder_rows).gather(2, reoreder_cols)
    
    return sorted_cov_matrix, sorted_cov_array, sorted_batch_indicies


def solve_system(sorted_cov_matrix, sorted_cov_array, num_nn):
    """
    Add identity values to NaN and solve system 

    Parameters:
    - sorted_cov_matrix : tensor(B, num_nn, num_nn), sorted matricies of covariance between nn locations
    - sorted_cov_array : tensor(B, num_nn), sorted arrays of covariance between nn and batch simulation locations
    - num_nn: int, maximum number of nearest neigbors

    Returns:
    - k_weights : tensor(B, num_nn), kriging weights 
    """
    

    # add padding to covariance array
    lstsq_cov_array = sorted_cov_array.nan_to_num(1)
    
    # add identity matrix pading to covariance matrix
    identity_matrix = torch.eye(num_nn, num_nn).unsqueeze(0).repeat(len(sorted_cov_array),1,1).cuda()
    lstsq_cov_matrix = torch.where(torch.isnan(sorted_cov_matrix), identity_matrix, sorted_cov_matrix)
    
    # solve
    k_weights = torch.linalg.lstsq(lstsq_cov_matrix, lstsq_cov_array).solution
    
    return k_weights


def pred_Z(kr_dictionary, size_list, full, cond_data, vario_sim, krig):
    """
    Final step in which the kriging mean and variance is calculated and an elevation value is sequentially determined by drawing from a normal distribution

    Parameters:
    - kr_dictionary : tensor(3, S, num_nn), batch dimension cooresponds to covariance array, indicies, and kriging weights, respectively
    - size_list : tensor(S), number of NN for each location to simulate
    - full : tensor(C+S, 2), all grid locations with the conditioning data at the beginning 
    - cond_data : tensor(C,), elevation values that coorespond to conditioning data locations
    - vario_sim : tensor(S, 6), variogram params for each location to be simulated [azimuth, nugget, major_range, minor_range, sill, smooth]
    - krig : char, 's' for simple or 'o' for ordinary kriging

    Returns:
    - full : tensor(C+S, 3), all grid locations with cooresponding elevation values
    """
    
    torch.cuda.set_device(0)
    
    torch.set_num_threads(1)
    torch.cuda.empty_cache()
                 
    z_mean = torch.mean(cond_data).cuda()
    z_lookup = torch.zeros(len(full)).cuda()
    z_lookup[:len(cond_data)] = cond_data
    
    vario_sim = vario_sim.cuda()
    
    for i in range(len(full) - len(cond_data)):
        
        covariance_array, indicies, weights = kr_dictionary[:,i,:int(size_list[i])]
        near_ele = torch.tensor([z_lookup[int(idx)] for idx in indicies]).cuda()
                                
        if krig == 'o':
            z_mean = torch.mean(near_ele)
            
        vario = vario_sim[i]
        
        # calculate kriging mean and variance
        est = z_mean + torch.dot(weights[:len(near_ele)].squeeze(), (near_ele - z_mean))
        var = torch.abs(vario[4] - torch.dot(weights[:len(near_ele)].squeeze(), covariance_array[:len(near_ele)]))
        
        z_lookup[len(cond_data) + i] = torch.normal(est,torch.sqrt(var))
    
    full = torch.column_stack((full.cuda(), z_lookup))
    
    return full

