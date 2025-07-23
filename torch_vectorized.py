import numpy as np
import pandas as pd
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
import numpy.linalg as linalg
from sklearn.preprocessing import QuantileTransformer
from torch.profiler import profile, record_function, ProfilerActivity
import skgstat as skg
import math
import itertools
import time
import torch

import sys
sys.path.append("../")

import gstatsim_torch as gst


def skrige_sgs_parallel(prediction_grid, torch_data, num_points, vario, radius, num_gpus, multiplier):
    
    preprocess_start = time.time()

    # Seperate data into known conditioning data and points to predict
    observed_coords = torch_data[:,:2].tolist()
    simulate_coords = [coord for coord in prediction_grid.tolist() if coord not in observed_coords]
    observed_coords = torch.tensor(observed_coords)
    simulate_coords = torch.tensor(simulate_coords)

    # Shuffle data to predict to create a random path
    index = torch.arange(len(simulate_coords)) 
    shuffle = index[torch.randperm(len(simulate_coords))]
    simulate_coords = simulate_coords[shuffle]

    # Create a tensor that has all grid locations with the conditioning data at the beginning
    full = torch.vstack((observed_coords, simulate_coords))

    # Unwrap variogram parameters and make a rotation matrix
    azimuth = vario[0]
    major_range = vario[2]
    minor_range = vario[3]
    rotation_matrix = gst.make_rotation_matrix(azimuth, major_range, minor_range, "cpu")

    # Create starting index that marks the beginning of the points to predict
    begin = len(observed_coords)

    # Determine how many points/grid-cells each process will predict 
    num_cells = len(simulate_coords)
    processes = num_gpus * multiplier
    cells_per_process = num_cells//processes
    
    proc_id = [i for i in range(processes)]

    # Create list of index lists such that each list is for a single process
    i_list = [[i for i in range(j*cells_per_process, (j+1)*cells_per_process)] for j in range(processes-1)]
    i_list.append([i for i in range((processes-1)*cells_per_process,num_cells)])
    
    # Gather the rest of the arguments to be sent to the function executed in parallel
    gpu_num = [i % num_gpus for i in range(processes)]
    args = zip(proc_id, i_list, gpu_num, itertools.cycle([full]), itertools.cycle([vario]), itertools.cycle([radius]),
               itertools.cycle([num_points]), itertools.cycle([begin]), itertools.cycle([rotation_matrix]))
    
    preprocess_end = time.time()
    
    print(f"Time to prepare data before parallel execution: {preprocess_end-preprocess_start}s")
    
    parallel_start = time.time()

    # use multiprocessing library to execute run_parallel_sgs_krig function in parallel
    mp.set_start_method('spawn', force=True)
    with mp.Pool(processes=num_gpus) as pool:
        
        out = pool.starmap(run_parallel_sgs_krig, args)
    
    # aggregate kriging weight data into a dictionary to look up data by index
    kr_dictionary = torch.zeros((3, num_cells, num_points)).cuda(0)
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
    sgs = pred_Z(kr_dictionary, size_list, full, torch_data[:,2], vario, 's')
    
    prediction_end = time.time()
    
    print(f"Time for serial prediction calculation: {prediction_end - prediction_start}")

    # Sort output to match up with original prediction grid
    sgs = sgs.cpu()
    sgs = sgs[np.lexsort((sgs[:,0], -sgs[:,1]))]

    return sgs[:,2]


def run_parallel_sgs_krig(proc_id, i_list, gpu_id, full, vario, radius, num_points, begin, rotation_matrix):
    
    # Assign the GPU
    torch.cuda.set_device(gpu_id)
    
    torch.set_num_threads(1)
    torch.cuda.empty_cache()
    
    # Send data to GPU 
    full = full.cuda()
    rotation_matrix = rotation_matrix.cuda()
    
    # Create index list of grid cells to compute
    offset = torch.tensor([i+begin for i in i_list]).cuda()
    
    # Get the size of the search matrix 
    N = int(offset[-1])

    # Create batch location tensor of grid cells to compute
    loc = full[offset]

    # Create batch location tensor of potential nearest neighbors wrt loc 
    search_candidates = torch.full((len(i_list), N, 2), float('nan')).cuda()
    for i, cur_offset in enumerate(offset):
        search_candidates[i, :cur_offset] = full[:cur_offset]
        
    # Add batch dimension to rotation matrix to be used for batch operations
    batch_rot_mat = rotation_matrix.unsqueeze(0).repeat(len(i_list),1,1)
        
    # Perform sorting and initial preperation to collect nearest neighbors
    stack, indicies, bins, bin_indices, oct_count = preprocess_for_nn_search(search_candidates, loc, radius, num_points)
    
    # Go through sorted tensor and collect the k nearest neighbors and their inidices
    batch_near, batch_indicies = nn_search_vectorized(stack, indicies, bin_indices, bins, num_points, oct_count)
    
    # Calcuate covariance matrix
    covariance_matrix = make_covariance_matrix(batch_near, vario, batch_rot_mat)

    # Calculate covariance between data and unknown
    covariance_array = make_covariance_array(batch_near, 
                    loc.unsqueeze(1).repeat(1, num_points, 1), 
                    vario, 
                    batch_rot_mat
                )

    # Reorder covariance matrix and array so NANs appear at the end
    sorted_cov_matrix, sorted_cov_array, sorted_indices = reorder(covariance_matrix,
                                                                    covariance_array,
                                                                    batch_indicies, num_points)

    # Get size list of number of NN collected for future indexing
    size_list = torch.sum((~torch.isnan(sorted_cov_array)).int(), dim=1)

    # Solve system defined by batch covariance matrix and array to get kriging weights
    k_weights = solve_system(sorted_cov_matrix, sorted_cov_array, num_points)
    
    # Store results in a batch "dictionary" tensor    
    kr_dict_subset = torch.stack((sorted_cov_array, sorted_indices, k_weights))
    
    return proc_id, kr_dict_subset, size_list


# NNS FUNCTIONS 
def preprocess_for_nn_search(search_candidates, loc, radius, num_points):
    """
    Preprocess input search_candidates and loc 
    input search_candidates, loc, radius, num_points
    return:
        stack
        indices,
        bins
        bin_indices
        oct_count
    """
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

    # Create index tensor (B, N)
    indices = torch.arange(N, device=stack.device).unsqueeze(0).repeat(B, 1).float()

    # Mask out points beyond the radius
    mask = torch.where(distances < radius, 1.0, float('nan'))
    stack = stack * mask.unsqueeze(2).repeat(1, 1, 4)
    indices = indices * mask

    # Sort by distance
    sorted_dist_idxs = torch.argsort(stack[..., 2], dim=1)
    stack_idxs = torch.arange(B, device=stack.device).repeat_interleave(N).reshape(B, N)

    # Apply sorting
    stack = stack.gather(1, sorted_dist_idxs.unsqueeze(-1).expand(-1, -1, 4))
    indices = indices.gather(1, sorted_dist_idxs)

    # Define 8 bins over angle range
    bins = torch.tensor([
        -math.pi, -3*math.pi/4, -math.pi/2, -math.pi/4, 0,
         math.pi/4, math.pi/2,  3*math.pi/4, math.pi
    ], device=stack.device)

    # Bin index based on angle (angle at index 3)
    bin_indices = torch.bucketize(stack[..., 3], bins, right=False)

    # Octant point count
    oct_count = num_points // 8

    return stack, indices, bins, bin_indices, oct_count


def nn_search_vectorized(stack, indices, bin_indices, bins, num_points, oct_count):
    '''
    Vectorized nearest-neighbor search with binning.

    Parameters:
    - stack: (B, N, 3), where each point has [x, y, distance]
    - indices: (B, N), index of each point
    - bin_indices: (B, N), bin assignment for each point (1 to K)
    - bins: (K+1,), bin edges
    - num_points: total points to select per batch (K * oct_count)
    - oct_count: max number of closest neighbors to select from each bin

    Returns:
    - smallest: (B, num_points, 2), selected coordinates
    - index_list: (B, num_points), selected indices
    - vec_time: elapsed time for execution
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
    smallest = topk_xy.permute(0, 2, 1, 3).reshape(B, num_points, 2)
    index_list = topk_ids.permute(0, 2, 1).reshape(B, num_points)

    return smallest, index_list

def make_covariance_matrix(smallest, vario, rotation_matrix):
    """
    Make covariance matrix showing covariances between each pair of input coordinates

    Parameters
    ----------
        smallest : (B, num_points, 2)
        vario : list of variogram parameters [azimuth, nugget, major_range, minor_range, sill, vtype]
        rotation_matrix : (2,2) matrix used to perform coordinate transformations

    Returns
    -------
        covariance_matrix : (B, num_points, num_points) matrix of covariance between n points
    """
    
    nug = vario[1]
    sill = vario[4]
    vtype = vario[5]
    
    smallest = smallest.to(dtype=torch.float64)
    mat = torch.matmul(smallest, rotation_matrix)
    effective_lag = torch.cdist(mat, mat, p=2)  # Compute pairwise distances
    covariance_matrix = gst.Covariance.covar(effective_lag, sill, nug, vtype)

    return covariance_matrix


def make_covariance_array(coord1, coord2, vario, rotation_matrix):
    """
    Make covariance array showing covariances between each data points and grid cell of interest

    Parameters
    ----------
        coord1 : numpy.ndarray
            coordinates of n data points
        coord2 : numpy.ndarray
            coordinates of grid cell of interest (i.e. grid cell being simulated) that is repeated n times
        vario : list
            list of variogram parameters [azimuth, nugget, major_range, minor_range, sill, vtype]
            azimuth, nugget, major_range, minor_range, and sill can be int or float type
            vtype is a string that can be either 'Exponential', 'Spherical', or 'Gaussian'
        rotation_matrix - rotation matrix used to perform coordinate transformations

    Returns
    -------
        covariance_array : numpy.ndarray
            nx1 array of covariance between n points and grid cell of interest
    """

    nug = vario[1]
    sill = vario[4]
    vtype = vario[5]
    coord1 = coord1.to(dtype=torch.float64)
    coord2 = coord2.to(dtype=torch.float64)
    mat1 = torch.matmul(coord1, rotation_matrix)
    mat2 = torch.matmul(coord2, rotation_matrix)
    effective_lag = torch.sqrt(torch.sum((mat1 - mat2).pow(2), dim=2))
    
    #Using the Matern Variogram 
    
    covariance_array = gst.Covariance.covar(effective_lag, sill, nug, vtype)

    return covariance_array


def reorder(covariance_matrix, covariance_array, index_vec, num_points):
    
    # Get mask to sort batch array and matrix so that nan are at the end
    nan_mask = torch.isnan(covariance_array)
    num_mask = ~torch.isnan(covariance_array)
    nan_indices = torch.nonzero(nan_mask)
    num_indices = torch.nonzero(num_mask)
    split_indices = torch.cat((num_indices, nan_indices))
    reorder_indices = split_indices[split_indices[:, 0].sort()[1]]
    reorder_indices = reorder_indices[:,1].reshape(covariance_array.shape)

    sorted_cov_array = covariance_array.gather(1, reorder_indices)
    sorted_index_vec = index_vec.gather(1, reorder_indices)

    reorder_rows = reorder_indices.unsqueeze(-1).repeat(1,1,num_points)
    reoreder_cols = reorder_indices.unsqueeze(1).repeat(1,num_points,1)

    sorted_cov_matrix = covariance_matrix.gather(1,reorder_rows).gather(2, reoreder_cols)
    
    return sorted_cov_matrix, sorted_cov_array, sorted_index_vec


def solve_system(sorted_cov_matrix, sorted_cov_array, num_points):
    
    # prepare for least squares by converting nan elements to identity

    lstsq_cov_array = sorted_cov_array.nan_to_num(1)

    identity_matrix = torch.eye(num_points, num_points).unsqueeze(0).repeat(len(sorted_cov_array),1,1).cuda()
    lstsq_cov_matrix = torch.where(torch.isnan(sorted_cov_matrix), identity_matrix, sorted_cov_matrix)
    
    k_weights = torch.linalg.lstsq(lstsq_cov_matrix, lstsq_cov_array).solution
    
    return k_weights


def pred_Z(kr_dictionary, size_list, full, df, vario, krig):
    
    torch.cuda.set_device(0)
    
    torch.set_num_threads(1)
    torch.cuda.empty_cache()
                 
    z_mean = torch.mean(df).cuda()
    z_lookup = torch.zeros(len(full)).cuda()
    z_lookup[:len(df)] = df
    
    for i in range(len(full) - len(df)):
        
        covariance_array, indicies, weights = kr_dictionary[:,i,:int(size_list[i])]
        near_ele = torch.tensor([z_lookup[int(idx)] for idx in indicies]).cuda()
                                
        if krig == 'o':
            z_mean = torch.mean(near_ele)
        
        # calculate kriging mean and variance
        est = z_mean + torch.dot(weights[:len(near_ele)].squeeze(), (near_ele - z_mean))
        var = torch.abs(vario[4] - torch.dot(weights[:len(near_ele)].squeeze(), covariance_array[:len(near_ele)]))
        
        z_lookup[len(df) + i] = torch.normal(est,torch.sqrt(var))
    
    full = torch.column_stack((full.cuda(), z_lookup))
    
    return full


def prepare_data():
    
    df_bed = pd.read_csv('demos/data/greenland_test_data.csv')

    # remove erroneously high values due to bad bed picks
    df_bed = df_bed[df_bed["Bed"] <= 700]  
    
    # grid data to 100 m resolution and remove coordinates with NaNs
    res = 1000
    df_grid, torch_data, rows, cols = gst.Gridding.grid_data(df_bed, 'X', 'Y', 'Bed', res)
    df_grid = df_grid[df_grid["Z"].isnull() == False]
    torch_data = torch_data[torch_data[:,2].isnan() == False]
    df_grid = df_grid.rename(columns = {"Z": "Bed"})

    # normal score transformation
    data = df_grid['Bed'].values.reshape(-1,1)
    nst_trans = QuantileTransformer(n_quantiles=500, output_distribution="normal").fit(data)
    df_grid['Nbed'] = nst_trans.transform(data)

    # compute experimental (isotropic) variogram
    coords = df_grid[['X','Y']].values
    values = df_grid['Nbed']
    torch_data[:,2] = torch.tensor(df_grid['Nbed'].to_numpy())

    maxlag = 50000             # maximum range distance
    n_lags = 70                # num of bins

    V1 = skg.Variogram(coords, values, bin_func='even', n_lags=n_lags, 
                       maxlag=maxlag, normalize=False)

    # use exponential variogram model
    V1.model = 'exponential'
    V1.parameters
    
    # define coordinate grid
    xmin = torch.min(torch_data[:,0]); xmax = torch.max(torch_data[:,0])     # min and max x values
    ymin = torch.min(torch_data[:,1]); ymax = torch.max(torch_data[:,1])     # min and max y values

    Pred_grid_xy = gst.Gridding.prediction_grid(xmin, xmax, ymin, ymax, res)
    
    # set variogram parameters
    azimuth = 0
    nugget = V1.parameters[2]

    # the major and minor ranges are the same in this example because it is isotropic
    major_range = V1.parameters[0]
    minor_range = V1.parameters[0]
    sill = V1.parameters[1]
    vtype = 'Exponential'

    # save variogram parameters as a list
    vario = [azimuth, nugget, major_range, minor_range, sill, vtype]

    k = 48         # number of neighboring data points used to estimate a given point
    rad = 50000     # 50 km search radius
    
    return Pred_grid_xy, torch_data, k, vario, rad, nst_trans


if __name__ == "__main__":
    
    Pred_grid_xy, torch_data, k, vario, rad, nst_trans = prepare_data()
    
    if torch.cuda.is_available():
        
        num_gpus = torch.cuda.device_count()
        multiplier = 20 # This is to avoid CUDA OUT OF MEM ERROR
        
        print("Starting")
        start_time = time.time()
        
        sgs = skrige_sgs_parallel(Pred_grid_xy, torch_data, k, vario, rad, num_gpus, multiplier)
        
        end_time = time.time()
        print(f"Total time to complete: {end_time-start_time}s")
        
        sgs = sgs.reshape(-1,1)
        sgs_trans = nst_trans.inverse_transform(sgs)
        torch.save(sgs_trans, "vectorized_sgs.pt")
        
    else: 
        
        print("Torch not properly configured to run on gpu.")

