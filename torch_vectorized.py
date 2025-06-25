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

    observed_coords = torch_data[:,:2].tolist()
    simulate_coords = [coord for coord in prediction_grid.tolist() if coord not in observed_coords]

    observed_coords = torch.tensor(observed_coords)
    simulate_coords = torch.tensor(simulate_coords)

    # Shuffle data to predict to create a random path
    index = torch.arange(len(simulate_coords)) 
    shuffle = index[torch.randperm(len(simulate_coords))]
    simulate_coords = simulate_coords[shuffle]

    full = torch.vstack((observed_coords, simulate_coords))

    azimuth = vario[0]
    major_range = vario[2]
    minor_range = vario[3]

    rotation_matrix = gst.make_rotation_matrix(azimuth, major_range, minor_range, "cpu")

    # create starting index for data from full to use for KNN
    begin = len(observed_coords)

    num_cells = len(simulate_coords)
    processes = num_gpus * multiplier
    cells_per_process = num_cells//processes

    i_list = [[i for i in range(j*cells_per_process, (j+1)*cells_per_process)] for j in range(processes-1)]
    i_list.append([i for i in range((processes-1)*cells_per_process,num_cells)])

    gpu_num = [i % num_gpus for i in range(processes)]

    args = zip(i_list, gpu_num, itertools.cycle([full]), itertools.cycle([vario]), itertools.cycle([radius]),
               itertools.cycle([num_points]), itertools.cycle([begin]), itertools.cycle([rotation_matrix]))

    kr_dictionary = {}

    mp.set_start_method('spawn', force=True)

    with mp.Pool(processes=num_gpus) as pool:
        # use python multiprocessing library to execute parallel_krige_sgs function in parallel
        out = pool.starmap(run_parallel_sgs_krig, args)

    # aggregate output into a dictionary to look up data by index
    for kr_dict_subset in out:

        kr_dictionary = kr_dictionary | kr_dict_subset


    sgs = pred_Z(kr_dictionary, full, torch_data[:,2], vario, 's')

    sgs = sgs.cpu()
    sgs = sgs[np.lexsort((sgs[:,0], -sgs[:,1]))]

    return sgs[:,2]


def run_parallel_sgs_krig(i_list, gpu_id, full, vario, radius, num_points, begin, rotation_matrix):
    
    # Assign the GPU
    torch.cuda.set_device(gpu_id)
    
    torch.cuda.empty_cache()
    
    # send data to GPU 
    full = full.cuda()
    rotation_matrix = rotation_matrix.cuda()
    
    # BEGIN VECTORIZED CODE
    offset = torch.tensor([i+begin for i in i_list]).cuda()

    loc = full[offset]

    search_candidates = torch.full((len(i_list), len(full), 2), float('nan')).cuda()
    for i, cur_offset in enumerate(offset):
        search_candidates[i, :cur_offset] = full[:cur_offset]
        
    batch_near, batch_indicies = nearest_neighbor_search(radius, num_points, loc, search_candidates)
    
    kr_dict_subset = {}
    
    for i, near in enumerate(batch_near):
        
        #remove NAN values
        near = near[~torch.isnan(near[:, 0])].reshape(-1,2)
        indicies = batch_indicies[i][~torch.isnan(batch_indicies[i])]

        k_weights, covariance_array = skriging(near, loc, vario, rotation_matrix)
        
        kr_dict_subset[i] = [covariance_array, indicies, k_weights]
    
    return kr_dict_subset


# MOSTLY VECTORIZED NNS
def nearest_neighbor_search(radius, num_points, loc, data2):
        
    locx = loc[:, 0].unsqueeze(1).repeat(1, data2.shape[1])
    locy = loc[:, 1].unsqueeze(1).repeat(1, data2.shape[1])

    x_tensor = data2[:, :, 0]
    y_tensor = data2[:, :, 1]

    centered_x = x_tensor - locx
    centered_y = y_tensor - locy

    distances = torch.sqrt(centered_x**2 + centered_y**2)
    angles = torch.atan2(centered_y, centered_x)

    # Stack the tensors into a single tensor
    stack = torch.stack((x_tensor, y_tensor, distances, angles), dim=2)

    indicies = torch.arange(data2.shape[1]).unsqueeze(0).repeat(data2.shape[0],1).cuda()

    # Filter out points outside the radius
    mask = torch.where((stack[:, :, 2] < radius), 1.0, float('nan')).cuda() 
    stack = stack * mask.unsqueeze(2).repeat(1, 1, 4)
    indicies = indicies * mask

    # Sort stack and indicies by distance
    sorted_dist_idxs = torch.argsort(stack[..., 2]).reshape(-1)
    stack_idxs = (torch.arange(stack.shape[0]).repeat_interleave(stack.shape[1]).reshape(-1))
    stack = stack[stack_idxs, sorted_dist_idxs, :].reshape(*stack.shape)
    indicies = indicies[stack_idxs, sorted_dist_idxs].reshape(*indicies.shape)

    # Use bucketize to find bin index for each angle
    bins = torch.tensor([-math.pi, -3*math.pi/4, -math.pi/2, -math.pi/4, 0,
                            math.pi/4, math.pi/2, 3*math.pi/4, math.pi]).cuda()
    bin_indices = torch.bucketize(stack[..., 3].contiguous(), bins, right=False) 

    # Allocate tensor for the result
    smallest = torch.full((len(loc), num_points, 2), float('nan')).cuda()
    index_list = torch.full((len(loc), num_points), float('nan')).cuda()
    oct_count = num_points // 8

    # Remember cant clean up NAN in smallest or index_list because nonequal number of NAN for each row 

    for i in range(1, bins.shape[0]):
        bin_mask = torch.where((bin_indices == i), 1.0, float('nan')).cuda()
        bin_points = (stack * bin_mask.unsqueeze(2).repeat(1, 1, 4))[..., :2]
        bin_index = indicies * bin_mask
        bin_points_count = torch.minimum(torch.tensor([oct_count]).cuda(), torch.sum(~torch.isnan(bin_index), dim=1))

        for j, count in enumerate(bin_points_count): 
            if count > 0:

                cur_loc_NN = bin_points[j]
                cur_loc_idx_list = bin_index[j]

                smallest[j, (i-1) * oct_count : (i-1) * oct_count + count, :] = cur_loc_NN[~torch.isnan(cur_loc_NN[:, 0])][:count]
                index_list[j, (i-1) * oct_count : (i-1) * oct_count + count] = cur_loc_idx_list[~torch.isnan(cur_loc_idx_list)][:count]

    return smallest, index_list


def skriging(near, loc, vario, rotation_matrix):
    
    numpoints = len(near)
    
    # covariance between data
    covariance_matrix = gst.Covariance.make_covariance_matrix(near, vario, rotation_matrix)
    
    # covariance between data and unknown
    covariance_array = gst.Covariance.make_covariance_array(
                    near, 
                    loc.unsqueeze(0).repeat(numpoints, 1), 
                    vario, 
                    rotation_matrix
                )
    
    k_weights = torch.linalg.lstsq(covariance_matrix, 
                                               covariance_array.unsqueeze(-1)).solution.squeeze(-1)
    
    return k_weights, covariance_array


def pred_Z(kr_dictionary, full, df, vario, krig):
    
    torch.cuda.set_device(0)
                 
    z_mean = torch.mean(df).cuda()
    z_lookup = torch.zeros(len(full)).cuda()
    z_lookup[:len(df)] = df
    
    for i in range(len(full) - len(df)):
        
        covariance_array, indicies, weights = kr_dictionary[i]
        near_ele = torch.tensor([z_lookup[int(idx)] for idx in indicies]).cuda()
                                
        if krig == 'o':
            z_mean = torch.mean(near_ele)
        
        # calculate kriging mean and variance
        est = z_mean + torch.dot(weights[:len(near_ele)].squeeze().cuda(), (near_ele - z_mean).to(dtype=torch.float64))
        var = torch.abs(vario[4] - torch.dot(weights[:len(near_ele)].squeeze().cuda(), covariance_array[:len(near_ele)].cuda()))
        
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
    
    return Pred_grid_xy, torch_data, k, vario, rad


if __name__ == "__main__":
    
    Pred_grid_xy, torch_data, k, vario, rad = prepare_data()
    
    if torch.cuda.is_available():
        
        num_gpus = torch.cuda.device_count()
        print(num_gpus)
        multiplier = 16 # This is to avoid CUDA OUT OF MEM ERROR
        
        sgs = skrige_sgs_parallel(Pred_grid_xy, torch_data, k, vario, rad, num_gpus, multiplier)
        
    else: 
        
        print("Torch not properly configured to run on gpu.")

