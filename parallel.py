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
import sys

def skrige_sgs(prediction_grid, df, num_points, vario, radius, processes, sgs):

    # Seperate observed data with data to predict
    observed_coords = df[['X', 'Y']].values.tolist()
    simulate_coords = [coord for coord in prediction_grid.tolist() if coord not in observed_coords]

    # Shuffle data to predict to create a random path
    np.random.shuffle(simulate_coords)

    full = np.vstack((np.array(observed_coords), np.array(simulate_coords)))

    # create starting index for data from full to use for KNN
    begin = len(observed_coords)

    # create iterable parameter list
    i = [i for i in range(len(simulate_coords))]
    args = zip(i, itertools.cycle([full]), itertools.cycle([vario]), itertools.cycle([radius]),
               itertools.cycle([num_points]), itertools.cycle([begin]), itertools.cycle(['s']))

    pool = mp.Pool(processes)
    start = time.time()
    kr_dictionary = {}
    out = pool.starmap(parallel_krige_sgs, args, chunksize=200)

    for (idx, weights, covariance_array, indicies) in out:
        
        # aggregate output into a dictionary to look up data by index
        kr_dictionary[idx] = (weights, covariance_array, indicies)
    
    sgs = pred_Z(kr_dictionary, full, df, vario, 's')
    
    sgs = sgs[np.lexsort((sgs[:,1], sgs[:,0]))]

    return sgs

def okrige_sgs(prediction_grid, df, num_points, vario, radius, processes, sgs):

    # Seperate observed data with data to predict
    observed_coords = df[['X', 'Y']].values.tolist()
    simulate_coords = [coord for coord in prediction_grid.tolist() if coord not in observed_coords]

    # Shuffle data to predict to create a random path
    np.random.shuffle(simulate_coords)

    full = np.vstack((np.array(observed_coords), np.array(simulate_coords)))

    # create starting index for data from full to use for KNN
    begin = len(observed_coords)

    # create iterable parameter list
    i = [i for i in range(len(simulate_coords))]
    args = zip(i, itertools.cycle([full]), itertools.cycle([vario]), itertools.cycle([radius]),
               itertools.cycle([num_points]), itertools.cycle([begin]), itertools.cycle(['o']))

    pool = mp.Pool(processes)
    start = time.time()
    kr_dictionary = {}
    out = pool.starmap(parallel_krige_sgs, args, chunksize=200)

    for (idx, weights, covariance_array, indicies) in out:
        
        # aggregate output into a dictionary to look up data by index
        kr_dictionary[idx] = (weights, covariance_array, indicies)
    
    sgs = pred_Z(kr_dictionary, full, df, vario, 'o')
    
    sgs = sgs[np.lexsort((sgs[:,1], sgs[:,0]))]

    return sgs


def parallel_krige_sgs(i, full, vario, radius, num_points, begin, krig):

     # offset in all_xyk of location to simulate
    curr_offset = begin + i
    
    loc = full[curr_offset]

    search_candidates = full[:curr_offset]

    near, indicies = NNS(search_candidates, radius, num_points, loc)
    if krig == 's':
        k_weights, covariance_array = skriging(near, loc, vario)
    elif krig == 'o':
        k_weights, covariance_array = okriging(near, loc, vario)

    return i, k_weights, covariance_array, indicies


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
    
    
    
    

