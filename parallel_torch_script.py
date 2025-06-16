import numpy as np
import pandas as pd
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
import numpy.linalg as linalg
from sklearn.preprocessing import QuantileTransformer
import skgstat as skg
import math
import itertools
import time
import torch

import sys
sys.path.append("../")

import gstatsim_torch as gst


def skrige_sgs_parallel(prediction_grid, torch_data, num_points, vario, radius, processes):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    observed_coords = torch_data[:,:2].tolist()
    simulate_coords = [coord for coord in prediction_grid.tolist() if coord not in observed_coords]

    observed_coords = torch.tensor(observed_coords, device = device)
    simulate_coords = torch.tensor(simulate_coords, device = device)

    # Shuffle data to predict to create a random path
    index = torch.arange(len(simulate_coords)) 
    shuffle = index[torch.randperm(len(simulate_coords))]
    simulate_coords = simulate_coords[shuffle]

    full = torch.vstack((observed_coords, simulate_coords))
    
    azimuth = vario[0]
    major_range = vario[2]
    minor_range = vario[3]

    rotation_matrix = gst.make_rotation_matrix(azimuth, major_range, minor_range, device)

    # create starting index for data from full to use for KNN
    begin = len(observed_coords)
    
    i = [i for i in range(len(simulate_coords))]
    args = zip(i, itertools.cycle([full]), itertools.cycle([vario]), itertools.cycle([radius]),
               itertools.cycle([num_points]), itertools.cycle([begin]), 
               itertools.cycle([rotation_matrix]), itertools.cycle([device]))
    
    mp.set_start_method('spawn', force=True)

    pool = mp.Pool(processes)

    kr_dictionary = {}
    
    # use python multiprocessing library to execute parallel_krige_sgs function in parallel
    out = pool.starmap(run_parallel_sgs_krig, args, chunksize=200)
    
    # aggregate output into a dictionary to look up data by index
    for (idx, weights, covariance_array, indicies) in out:
        
        kr_dictionary[idx] = [covariance_array, indicies, weights]

        
    sgs = pred_Z(kr_dictionary, full, torch_data[:,2], vario, 's', device)

    sgs = sgs.cpu()
    sgs = sgs[np.lexsort((sgs[:,0], -sgs[:,1]))]

    return sgs[:,2]


def run_parallel_sgs_krig(i, full, vario, radius, num_points, begin, rotation_matrix, device):
    
    curr_offset = begin + i

    loc = full[curr_offset]

    search_candidates = full[:curr_offset]

    near, indicies = nearest_neighbor_search(radius, num_points, loc, search_candidates, device)

    k_weights, covariance_array = skriging(near, loc, vario, rotation_matrix, device)
    
    return i, k_weights, covariance_array, indicies


def skriging(near, loc, vario, rotation_matrix, device):
    
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


def pred_Z(kr_dictionary, full, df, vario, krig, device):
                 
    z_mean = torch.mean(df) 
    z_lookup = torch.zeros(len(full), device = device)
    z_lookup[:len(df)] = df
    
    for i in range(len(full) - len(df)):
        
        covariance_array, indicies, weights = kr_dictionary[i]
        near_ele = torch.tensor([z_lookup[int(idx)] for idx in indicies], device=device)
                                
        if krig == 'o':
            z_mean = torch.mean(near_ele)
        
        # calculate kriging mean and variance
        est = z_mean + torch.dot(weights[:len(near_ele)].squeeze(), (near_ele - z_mean).to(dtype=torch.float64))
        var = torch.abs(vario[4] - torch.dot(weights[:len(near_ele)].squeeze(), covariance_array[:len(near_ele)]))
        
        z_lookup[len(df) + i] = torch.normal(est,torch.sqrt(var))
    
    full = torch.column_stack((full, z_lookup))
    
    return full


def nearest_neighbor_search(radius, num_points, loc, data2, device):
        
    locx = loc[0]
    locy = loc[1]

    x_tensor = data2[:, 0]
    y_tensor = data2[:, 1]

    centered_x = x_tensor - locx
    centered_y = y_tensor - locy
    
    distances = torch.sqrt(centered_x**2 + centered_y**2)
    angles = torch.atan2(centered_y, centered_x)

    # Stack the tensors into a single tensor
    stack = torch.stack((x_tensor, y_tensor, distances, angles), dim=1)

    # Initialize index list
    indicies = torch.arange(len(data2), device=device) 

    # Filter out points outside the radius
    mask = stack[:, 2] < radius  # The distances are at index 3
    stack = stack[mask]
    indicies = indicies[mask]

    # Sort the stack based on the distances
    sorted_indices = torch.argsort(stack[:, 2]) 
    stack = stack[sorted_indices]
    indicies = indicies[sorted_indices]

    # Use bucketize to find bin index for each angle
    bins = torch.tensor([-math.pi, -3*math.pi/4, -math.pi/2, -math.pi/4, 0,
                            math.pi/4, math.pi/2, 3*math.pi/4, math.pi], device=device)
    bin_indices = torch.bucketize(stack[:, 3].contiguous(), bins, right=True)  # The angles are at index 4

    # Allocate tensor for the result
    smallest = torch.full((num_points, 2), float('nan'), device=device)
    index_list = torch.full((num_points,), float('nan'), device=device)
    oct_count = num_points // 8

    # Collect points for each bin
    for i in range(1, bins.shape[0]):
        current_bin_mask = bin_indices == i
        current_bin_points = stack[current_bin_mask][:, :2]  # Get X, Y
        index_tmp = indicies[current_bin_mask]
        bin_points_count = min(oct_count, current_bin_points.shape[0])
        
        if bin_points_count > 0:
            smallest[(i-1) * oct_count : (i-1) * oct_count + bin_points_count, :] = current_bin_points[:bin_points_count, :]
            index_list[(i-1) * oct_count : (i-1) * oct_count + bin_points_count] = index_tmp[:bin_points_count]

    # Remove NaN values to get the final result
    near = smallest[~torch.isnan(smallest[:, 0])].reshape(-1, 2)
    index_list = index_list[~torch.isnan(index_list)]

    return near, index_list


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

    
    sgs = skrige_sgs_parallel(Pred_grid_xy, torch_data, k, vario, rad, 3)
    
