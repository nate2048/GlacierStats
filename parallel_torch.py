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
import sys

def skrige_sgs(prediction_grid, df, num_points, vario, radius, sgs):

    # Seperate observed data with data to predict
    observed_coords = df[['X', 'Y']].values.tolist()
    simulate_coords = [coord for coord in prediction_grid.tolist() if coord not in observed_coords]

    # Shuffle data to predict to create a random path
    np.random.shuffle(simulate_coords)

    full = np.vstack((np.array(observed_coords), np.array(simulate_coords)))

    # create starting index for data from full to use for KNN
    begin = len(observed_coords)

    kr_dictionary = {}

    for i in range(len(simulate_coords)):

        # offset in all_xyk of location to simulate
        curr_offset = begin + i
        
        loc = full[curr_offset]

        search_candidates = full[:curr_offset]

        near, indicies = NNS(search_candidates, radius, num_points, loc)
        k_weights, covariance_array = skriging(near, loc, vario)
        kr_dictionary[i] = (k_weights, covariance_array, indicies)

    
    sgs = pred_Z(kr_dictionary, full, df, vario, 's')
    
    sgs = sgs[np.lexsort((sgs[:,1], sgs[:,0]))]

    return sgs

def okrige_sgs(prediction_grid, df, num_points, vario, radius, sgs):

    # Seperate observed data with data to predict
    observed_coords = df[['X', 'Y']].values.tolist()
    simulate_coords = [coord for coord in prediction_grid.tolist() if coord not in observed_coords]

    # Shuffle data to predict to create a random path
    np.random.shuffle(simulate_coords)

    full = np.vstack((np.array(observed_coords), np.array(simulate_coords)))

    # create starting index for data from full to use for KNN
    begin = len(observed_coords)

    kr_dictionary = {}

    for i in range(len(simulate_coords)):
        # offset in all_xyk of location to simulate
        curr_offset = begin + i
        
        loc = full[curr_offset]

        search_candidates = full[:curr_offset]

        near, indicies = NNS(search_candidates, radius, num_points, loc)
        k_weights, covariance_array = okriging(near, loc, vario)
        kr_dictionary[i] = (k_weights, covariance_array, indicies)
    
    sgs = pred_Z(kr_dictionary, full, df, vario, 'o')
    
    sgs = sgs[np.lexsort((sgs[:,1], sgs[:,0]))]

    return sgs

def nearest_neighbor_search(data2, radius, num_points, loc, device):
    """
    Nearest neighbor octant search
    
    Parameters
    ----------
        radius : int, float
            search radius
        num_points : int
            number of points to search for
        loc : numpy.ndarray
            coordinates for grid cell of interest
        data2 : pandas DataFrame
            data 
    
    Returns
    -------
        near : numpy.ndarray
            nearest neighbors
    """
    
    locx = loc[0]
    locy = loc[1]

    x_tensor = data2[:, 0]
    y_tensor = data2[:, 1]
    z_tensor = data2[:, 2]

    centered_x = x_tensor - locx
    centered_y = y_tensor - locy
    
    distances = torch.sqrt(centered_x**2 + centered_y**2)
    angles = torch.atan2(centered_y, centered_x)

    # Stack the tensors into a single tensor
    stack = torch.stack((x_tensor, y_tensor, z_tensor, distances, angles), dim=1)

    # Filter out points outside the radius
    mask = stack[:, 3] < radius  # The distances are at index 3
    stack = stack[mask]

    # Sort the stack based on the distances
    sorted_indices = torch.argsort(stack[:, 3])  # The distances are at index 3
    stack = stack[sorted_indices]

    # Use bucketize to find bin index for each angle
    bins = torch.tensor([-math.pi, -3*math.pi/4, -math.pi/2, -math.pi/4, 0,
                            math.pi/4, math.pi/2, 3*math.pi/4, math.pi], device=device)
    bin_indices = torch.bucketize(stack[:, 4].contiguous(), bins, right=True)  # The angles are at index 4

    # Allocate tensor for the result
    smallest = torch.full((num_points, 3), float('nan'), device=device)
    oct_count = num_points // 8

    # Collect points for each bin
    for i in range(1, bins.shape[0]):
        current_bin_mask = bin_indices == i
        current_bin_points = stack[current_bin_mask][:, :3]  # Get X, Y, Z
        bin_points_count = min(oct_count, current_bin_points.shape[0])
        if bin_points_count > 0:
            smallest[(i-1) * oct_count : (i-1) * oct_count + bin_points_count, :] = current_bin_points[:bin_points_count, :]

    # Remove NaN values to get the final result
    near = smallest[~torch.isnan(smallest[:, 0])].reshape(-1, 3)
    
    return near


def NNS(search_candidates, radius, num_points, loc):

    centered = search_candidates - loc

    angles = np.arctan2(centered[:, 0], centered[:, 1])

    dist = np.linalg.norm(centered, axis = 1)

    radius_filter = dist < radius

    centered = centered[radius_filter,:]
    angles = angles[radius_filter]
    dist = dist[radius_filter]

    sort = np.argsort(dist)

    centered = centered[sort]
    angles = angles[sort]
    dist = dist[sort]

    bins = [-math.pi, -3*math.pi/4, -math.pi/2, -math.pi/4, 0, 
            math.pi/4, math.pi/2, 3*math.pi/4, math.pi + 1] 
    
    oct = np.zeros(len(angles))
    
    # get 
    for i, angle in enumerate(angles):
        for j in range (8):
            if angle >= bins[j] and angle < bins[j+1]:
                oct[i] = j

    oct_count = num_points // 8
    nearest = np.ones(shape=(num_points, 2)) * np.nan
    indicies = np.ones(num_points) * np.nan

    for i in range(8):
    
        octant = centered[oct == i][:oct_count]
        
        for j, row in enumerate(octant):
        
            nearest[i*oct_count+j,:] = row + loc
            indicies[i*oct_count+j] = np.where(np.all(search_candidates==(row+loc),axis=1))[0]
            #indicies[i*oct_count+j] = np.where(np.all(search_candidates==(row+loc),axis=1))[0]
            
    
    near = nearest[~np.isnan(nearest)].reshape(-1,2)
    indicies = indicies[~np.isnan(indicies)]

    return near, indicies


def skriging(near, loc, vario):

    numpoints = len(near)

    azimuth = vario[0]
    major_range = vario[2]
    minor_range = vario[3]
    
    if numpoints == 0:
        print("ZEROOOO:")
        sys.stdout.flush()

    rotation_matrix = gs.make_rotation_matrix(azimuth, major_range, minor_range)

    # covariance between data
    covariance_matrix = np.zeros(shape=((numpoints, numpoints)))
    covariance_matrix = gs.Covariance.make_covariance_matrix(near, vario, rotation_matrix)

    # covariance between data and unknown
    covariance_array = np.zeros(shape=(numpoints))
    k_weights = np.zeros(shape=(numpoints))
    covariance_array = gs.Covariance.make_covariance_array(near, np.tile(loc, numpoints), vario, rotation_matrix)
    covariance_matrix.reshape(((numpoints)), ((numpoints)))

    k_weights, res, rank, s = linalg.lstsq(covariance_matrix, covariance_array, rcond=None)
    
    return k_weights, covariance_array


def okriging(near, loc, vario):

    numpoints = len(near)

    azimuth = vario[0]
    major_range = vario[2]
    minor_range = vario[3]
    
    if numpoints == 0:
        print("ZEROOOO:")
        sys.stdout.flush()

    rotation_matrix = gs.make_rotation_matrix(azimuth, major_range, minor_range)

    # covariance between data
    covariance_matrix = np.zeros(shape=((numpoints+1, numpoints+1)))
    covariance_matrix[0:numpoints,0:numpoints] = gs.Covariance.make_covariance_matrix(near, vario, rotation_matrix)
    
    covariance_matrix[numpoints, 0:numpoints] = 1
    covariance_matrix[0:numpoints, numpoints] = 1

    # covariance between data and unknown
    covariance_array = np.zeros(shape=(numpoints+1))
    k_weights = np.zeros(shape=(numpoints+1))
    covariance_array[0:numpoints] = gs.Covariance.make_covariance_array(near, np.tile(loc, numpoints), vario, rotation_matrix)
    covariance_array[numpoints] = 1 
    covariance_matrix.reshape(((numpoints+1)), ((numpoints+1)))

    k_weights, res, rank, s = linalg.lstsq(covariance_matrix, covariance_array, rcond=None)
    
    return k_weights, covariance_array
    

def pred_Z(kr_dictionary, full, df, vario, krig):
                 
    z_mean = np.average(df['Nbed'].values) # CHANGE TO Z 
    z_lookup = np.zeros(len(full))
    z_lookup[:len(df)] = df['Nbed'].values
    
    for i in range(len(full) - len(df)):
        
        weights, covariance_array, indicies = kr_dictionary[i]
        near_ele = np.array([z_lookup[int(idx)] for idx in indicies])
                                
        if krig == 'o':
            z_mean = np.mean(near_ele)
        
        # calculate kriging mean and variance
        est = z_mean + np.sum(weights[:len(near_ele)] * (near_ele - z_mean))
        var = abs(vario[4] - np.sum(weights[:len(near_ele)] * covariance_array[:len(near_ele)]))
        
        z_lookup[len(df) + i] = np.random.default_rng().normal(est, math.sqrt(var))
    
    full = np.column_stack((full, z_lookup))
    
    return full


### HELPER FUNCTIONS ###


def make_rotation_matrix(azimuth, major_range, minor_range, device):
    """
    Make rotation matrix for accommodating anisotropy
    
    Parameters
    ----------
        azimuth : int, float
            angle (in degrees from horizontal) of axis of orientation
        major_range : int, float
            range parameter of variogram in major direction, or azimuth
        minor_range : int, float
            range parameter of variogram in minor direction, or orthogonal to azimuth
    
    Returns
    -------
        rotation_matrix : numpy.ndarray
            2x2 rotation matrix used to perform coordinate transformations
    """
    
    theta = (azimuth / 180.0) * math.pi
    
    rotation_matrix = torch.tensor(
        [[math.cos(theta), -math.sin(theta)],
         [math.sin(theta), math.cos(theta)]],
        dtype=torch.float64,
      	device=device
    )

    scaling_matrix = torch.tensor(
        [[1 / major_range, 0],
         [0, 1 / minor_range]],
        dtype=torch.float64,
      	device=device
    )

    rotation_matrix = torch.mm(rotation_matrix, scaling_matrix)
    
    return rotation_matrix

def covar(effective_lag, sill, nug, vtype):
    """
    Compute covariance
    
    Parameters
    ----------
        effective_lag : int, float
            lag distance that is normalized to a range of 1
        sill : int, float
            sill of variogram
        nug : int, float
            nugget of variogram
        vtype : string
            type of variogram model (Exponential, Gaussian, or Spherical)
    Raises
    ------
    AtrributeError : if vtype is not 'Exponential', 'Gaussian', or 'Spherical'

    Returns
    -------
        c : numpy.ndarray
            covariance
    """
    
    if vtype.lower() == 'exponential':
        c = (sill - nug) * torch.exp(-3 * effective_lag)
    elif vtype.lower() == 'gaussian':
        c = (sill - nug) * torch.exp(-3 * effective_lag.pow(2))
    elif vtype.lower() == 'spherical':
        c = sill - nug - 1.5 * effective_lag + 0.5 * effective_lag.pow(3)
        c = torch.where(effective_lag > 1, sill - 1, c)
    else: 
        raise AttributeError(f"vtype must be 'Exponential', 'Gaussian', or 'Spherical'")
    return c
    
def make_covariance_matrix(coord, vario, rotation_matrix):
    """
    Make covariance matrix showing covariances between each pair of input coordinates
    
    Parameters
    ----------
        coord : numpy.ndarray
            coordinates of data points
        vario : list
            list of variogram parameters [azimuth, nugget, major_range, minor_range, sill, vtype]
            azimuth, nugget, major_range, minor_range, and sill can be int or float type
            vtype is a string that can be either 'Exponential', 'Spherical', or 'Gaussian'
        rotation_matrix : numpy.ndarray
            rotation matrix used to perform coordinate transformations
    
    Returns
    -------
        covariance_matrix : numpy.ndarray 
            nxn matrix of covariance between n points
    """
    
    nug = vario[1]
    sill = vario[4]
    vtype = vario[5]
    coord = coord.to(dtype=torch.float64)
    mat = torch.matmul(coord, rotation_matrix)
    effective_lag = torch.cdist(mat, mat, p=2)  # Compute pairwise distances
    covariance_matrix = covar(effective_lag, sill, nug, vtype)

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
    mat2 = torch.matmul(coord2.reshape(-1, 2), rotation_matrix)
    effective_lag = torch.sqrt(torch.sum((mat1 - mat2).pow(2), dim=1))
    covariance_array = covar(effective_lag, sill, nug, vtype)

    return covariance_array