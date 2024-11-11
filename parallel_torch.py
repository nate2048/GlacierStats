import numpy as np
import pandas as pd
import multiprocessing as mp
import numpy as np
import pandas as pd
import numpy.linalg as linalg
import gstatsim as gs
import math
import itertools
import time
import torch
import gstatsim_torch as gst
import sys

def skrige_sgs(prediction_grid, torch_data, num_points, vario, radius):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    observed_coords = torch_data[:,:2].tolist()
    simulate_coords = [coord for coord in prediction_grid.tolist() if coord not in observed_coords]

    observed_coords = torch.tensor(observed_coords, device = device)
    simulate_coords = torch.tensor(simulate_coords, device = device)

    # Shuffle data to predict to create a random path
    index = torch.arange(len(simulate_coords)) 
    shuffle = index[torch.randperm(len(simulate_coords))]
    simulate_coords = simulate_coords[shuffle]

    full = torch.vstack((observed_coords, simulate_coords))

    # create starting index for data from full to use for KNN
    begin = len(observed_coords)

    kriging_p1 = {}
    kr_dictionary = {}
    largest = 0

    for i in range(len(simulate_coords)):

        # offset in all_xyk of location to simulate
        curr_offset = begin + i
        
        loc = full[curr_offset]

        search_candidates = full[:curr_offset]

        near, indicies = nearest_neighbor_search(radius, num_points, loc, search_candidates, device)
        covariance_matrix, covariance_array = skriging_p1(near, loc, vario, device)
        kriging_p1[i] = (covariance_matrix, covariance_array)
        kr_dictionary[i] = [covariance_array, indicies]

        if len(covariance_array) > largest:
            largest = len(covariance_array)

    k_weights, size_list = kriging_p2(kriging_p1, largest, device)

    for i in range(len(kr_dictionary)):
        kr_dictionary[i].append(k_weights[i,:size_list[i]])

    sgs = pred_Z(kr_dictionary, full, torch_data[:,2], vario, 's', device)

    sgs = sgs[np.lexsort((sgs[:,0], -sgs[:,1]))]

    return sgs[:,2]

def okrige_sgs(prediction_grid, torch_data, num_points, vario, radius):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    observed_coords = torch_data[:,:2].tolist()
    simulate_coords = [coord for coord in prediction_grid.tolist() if coord not in observed_coords]

    observed_coords = torch.tensor(observed_coords, device = device)
    simulate_coords = torch.tensor(simulate_coords, device = device)

    # Shuffle data to predict to create a random path
    index = torch.arange(len(simulate_coords)) 
    shuffle = index[torch.randperm(len(simulate_coords))]
    simulate_coords = simulate_coords[shuffle]

    full = torch.vstack((observed_coords, simulate_coords))

    # create starting index for data from full to use for KNN
    begin = len(observed_coords)

    kriging_p1 = {}
    kr_dictionary = {}
    largest = 0

    for i in range(len(simulate_coords)):
        # offset in all_xyk of location to simulate
        curr_offset = begin + i
        
        loc = full[curr_offset]

        search_candidates = full[:curr_offset]

        near, indicies = nearest_neighbor_search(radius, num_points, loc, search_candidates, device)
        covariance_matrix, covariance_array = okriging_p1(near, loc, vario, device)
        kriging_p1[i] = (covariance_matrix, covariance_array)
        kr_dictionary[i] = [covariance_array, indicies]

        if len(covariance_array) > largest:
            largest = len(covariance_array)
    
    k_weights, size_list = kriging_p2(kriging_p1, largest, device)

    for i in range(len(kr_dictionary)):
        kr_dictionary[i].append(k_weights[i,:size_list[i]])
    
    sgs = pred_Z(kr_dictionary, full, torch_data[:,2], vario, 's', device)

    sgs = sgs[np.lexsort((sgs[:,0], -sgs[:,1]))]

    return sgs[:,2]

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
        

def skriging_p1(near, loc, vario, device):

    numpoints = len(near)

    azimuth = vario[0]
    major_range = vario[2]
    minor_range = vario[3]

    rotation_matrix = gst.make_rotation_matrix(azimuth, major_range, minor_range, device)

    # covariance between data
    covariance_matrix = gst.Covariance.make_covariance_matrix(near, vario, rotation_matrix)

    # covariance between data and unknown
    covariance_array = gst.Covariance.make_covariance_array(
                    near, 
                    loc.unsqueeze(0).repeat(numpoints, 1), 
                    vario, 
                    rotation_matrix
                )

    return covariance_matrix, covariance_array

def okriging_p1(near, loc, vario, device):

    numpoints = len(near)

    azimuth = vario[0]
    major_range = vario[2]
    minor_range = vario[3]

    rotation_matrix = gst.make_rotation_matrix(azimuth, major_range, minor_range, device)

    # covariance between data
    covariance_matrix = torch.zeros(numpoints+1, numpoints+1)
    covariance_matrix[0:numpoints,0:numpoints] = gst.Covariance.make_covariance_matrix(near, vario, rotation_matrix)
    covariance_matrix[numpoints, 0:numpoints] = 1
    covariance_matrix[0:numpoints, numpoints] = 1

    # covariance between data and unknown
    covariance_array = torch.zeros(numpoints+1)
    covariance_array[0:numpoints] = gst.Covariance.make_covariance_array(
                                        near, 
                                        loc.unsqueeze(0).repeat(numpoints, 1), 
                                        vario, 
                                        rotation_matrix
                                    )
    covariance_array[numpoints] = 1

    return covariance_matrix, covariance_array


def kriging_p2(kriging_p1, largest, device):

    batch_matrix = torch.zeros(len(kriging_p1), largest, largest, device=device)
    batch_array = torch.zeros(len(kriging_p1), largest, 1, device=device)
    size_list = []

    for i in range(len(kriging_p1)):

        covariance_matrix, covariance_array = kriging_p1[i]
        size = len(covariance_array)
        size_list.append(size)
        batch_matrix[i, :size, :size] = covariance_matrix
        batch_array[i, :size] = covariance_array.unsqueeze(1)

    k_weights = torch.linalg.lstsq(batch_matrix, batch_array).solution

    return k_weights, size_list

def pred_Z(kr_dictionary, full, df, vario, krig, device):
                 
    z_mean = torch.mean(df) 
    z_lookup = torch.zeros(len(full), device = device)
    z_lookup[:len(df)] = df
    
    for i in range(len(full) - len(df)):
        
        covariance_array, indicies, weights = kr_dictionary[i]
        near_ele = torch.tensor([z_lookup[int(idx)] for idx in indicies])
                                
        if krig == 'o':
            z_mean = torch.mean(near_ele)
        
        # calculate kriging mean and variance
        est = z_mean + torch.dot(weights[:len(near_ele)].squeeze(), (near_ele - z_mean))
        var = torch.abs(vario[4] - torch.dot(weights[:len(near_ele)].squeeze(), covariance_array[:len(near_ele)].to(torch.float)))
        
        z_lookup[len(df) + i] = torch.normal(est,torch.sqrt(var))
    
    full = torch.column_stack((full, z_lookup))
    
    return full
