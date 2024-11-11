import numpy as np
import pandas as pd
import multiprocessing as mp
import numpy as np
import pandas as pd
import numpy.linalg as linalg
import gstatsim as gs
import math
import random
import itertools
import time
import sys

##########################
# Interpolation functions

def skrige_interp(prediction_grid, df, xx, yy, zz, num_points, vario, radius, processes):

    # Seperate observed data with data to predict
    observed_coords = df[[xx, yy]].values.tolist()
    simulate_coords = [coord for coord in prediction_grid.tolist() if coord not in observed_coords]
 
    mean = df[zz].mean()

    # create iterable parameter list
    i = [i for i in range(len(simulate_coords))]
    args = zip(i, itertools.cycle([simulate_coords]), itertools.cycle([df[[xx, yy, zz]].to_numpy()]), itertools.cycle([mean]),
               itertools.cycle([vario]), itertools.cycle([radius]), itertools.cycle([num_points]), itertools.cycle(['s']))

    pool = mp.Pool(processes)

    est_sk = np.zeros(shape=len(simulate_coords)) 
    var_sk = np.zeros(shape=len(simulate_coords))
    out = pool.starmap(parallel_krige, args, chunksize=200)

    for (idx, est_sk_out, var_sk_out) in out:
        
        # aggregate output into a dictionary to look up data by index
        est_sk[idx] = est_sk_out
        var_sk[idx] = var_sk_out

    full = np.zeros(shape=(len(prediction_grid), 4))
    full[:len(df), 0:3] = df[[xx, yy, zz]].values
    full[len(df):, 0:2] = simulate_coords
    full[len(df):, 2] = est_sk
    full[len(df):, 3] = var_sk
    
    full = full[np.lexsort((full[:,0], -full[:,1]))]

    return full[:,2], full[:,3]


def okrige_interp(prediction_grid, df, xx, yy, zz, num_points, vario, radius, processes):

    # Seperate observed data with data to predict
    observed_coords = df[[xx, yy]].values.tolist()
    simulate_coords = [coord for coord in prediction_grid.tolist() if coord not in observed_coords]
 
    mean = df[zz].mean()

    # create iterable parameter list
    i = [i for i in range(len(simulate_coords))]
    args = zip(i, itertools.cycle([simulate_coords]), itertools.cycle([df[[xx, yy, zz]].to_numpy()]), itertools.cycle([mean]),
               itertools.cycle([vario]), itertools.cycle([radius]), itertools.cycle([num_points]), itertools.cycle(['o']))

    pool = mp.Pool(processes)

    est_sk = np.zeros(shape=len(simulate_coords)) 
    var_sk = np.zeros(shape=len(simulate_coords))
    out = pool.starmap(parallel_krige, args, chunksize=200)

    for (idx, est_sk_out, var_sk_out) in out:
        
        # aggregate output into a dictionary to look up data by index
        est_sk[idx] = est_sk_out
        var_sk[idx] = var_sk_out

    full = np.zeros(shape=(len(prediction_grid), 4))
    full[:len(df), 0:3] = df[[xx, yy, zz]].values
    full[len(df):, 0:2] = simulate_coords
    full[len(df):, 2] = est_sk
    full[len(df):, 3] = var_sk
    
    full = full[np.lexsort((full[:,0], -full[:,1]))]

    return full[:,2], full[:,3]

def skrige_sgs(prediction_grid, df, xx, yy, zz, num_points, vario, radius, processes):

    # Seperate observed data with data to predict
    observed_coords = df[[xx, yy]].values.tolist()
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
    
    sgs = pred_Z(kr_dictionary, full, df[zz], vario, 's')
    
    sgs = sgs[np.lexsort((sgs[:,0], -sgs[:,1]))]

    return sgs[:,2]

def okrige_sgs(prediction_grid, df, xx, yy, zz, num_points, vario, radius, processes):

    # Seperate observed data with data to predict
    observed_coords = df[[xx, yy]].values.tolist()
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
    
    sgs = pred_Z(kr_dictionary, full, df[zz], vario, 'o')
    
    sgs = sgs[np.lexsort((sgs[:,0], -sgs[:,1]))]

    return sgs[:,2]


def cluster_sgs(prediction_grid, df, num_points, gamma, radius, processes):

    # Seperate observed data with data to predict
    observed_coords = df[['X', 'Y']].values.tolist()
    simulate_coords = [coord for coord in prediction_grid.tolist() if coord not in observed_coords]

    # Shuffle data to predict to create a random path
    np.random.shuffle(simulate_coords)

    full = np.vstack((np.array(observed_coords), np.array(simulate_coords)))

    # add column for cluster number
    new_col = np.ones((len(full), 1)) * np.nan
    full = np.hstack((full, new_col))
    full[:len(df),2] = df[['K']].squeeze()

    # create starting index for data from full to use for KNN
    begin = len(observed_coords)

    # create iterable parameter list
    i = [i for i in range(len(simulate_coords))]
    args = zip(i, itertools.cycle([full]), itertools.cycle([gamma]), itertools.cycle([radius]),
               itertools.cycle([num_points]), itertools.cycle([begin]))

    pool = mp.Pool(processes)

    kr_dictionary = {}
    out = pool.starmap(parallel_cluster_sgs, args, chunksize=200)

    for (idx, weights, covariance_array, indicies, cluster_num) in out:
        
        # aggregate output into a dictionary to look up data by index
        kr_dictionary[idx] = (weights, covariance_array, indicies, cluster_num)
    
    sgs = pred_Z_cluster(kr_dictionary, full, df, gamma)
    
    sgs = sgs[np.lexsort((sgs[:,0], -sgs[:,1]))]

    return sgs[:, 3]


def cokrige_mm1(prediction_grid, df1, xx1, yy1, zz1, df2, xx2, yy2, zz2, num_points, vario, radius, corrcoef, processes):

    # Seperate observed data with data to predict
    observed_coords = df1[[xx1, yy1]].values.tolist()
    simulate_coords = [coord for coord in prediction_grid.tolist() if coord not in observed_coords]

    mean_1 = np.average(df1[zz1]) 
    var_1 = np.var(df1[zz1]) # replaced var_1 = vario[4]
    vario[4] = np.var(df1[zz1]) 
    mean_2 = np.average(df2[zz2]) 
    var_2 = np.var(df2[zz2])

    # create iterable parameter list
    i = [i for i in range(len(simulate_coords))]
    args = zip(i, itertools.cycle([simulate_coords]), itertools.cycle([df1[[xx1, yy1, zz1]].to_numpy()]), itertools.cycle([mean_1]), itertools.cycle([var_1]),
               itertools.cycle([df2[[xx2, yy2, zz2]].to_numpy()]), itertools.cycle([mean_2]), itertools.cycle([var_2]),
               itertools.cycle([vario]), itertools.cycle([radius]), itertools.cycle([num_points]), itertools.cycle([corrcoef]))
    
    pool = mp.Pool(processes)

    est_cokrige = np.zeros(shape=len(simulate_coords)) 
    var_cokrige = np.zeros(shape=len(simulate_coords))
    out = pool.starmap(parallel_cokrige, args, chunksize=200)

    for (idx, est_cokrige_out, var_cokrige_out) in out:
        
        # aggregate output into a dictionary to look up data by index
        est_cokrige[idx] = est_cokrige_out
        var_cokrige[idx] = var_cokrige_out

    full = np.zeros(shape=(len(prediction_grid), 4))
    full[:len(df1), 0:3] = df1[[xx1, yy1, zz1]].values
    full[len(df1):, 0:2] = simulate_coords
    full[len(df1):, 2] = est_cokrige
    full[len(df1):, 3] = var_cokrige
    
    full = full[np.lexsort((full[:,0], -full[:,1]))]

    return full[:,2], full[:,3]

def cosim_mm1(prediction_grid, df1, xx1, yy1, zz1, df2, xx2, yy2, zz2, num_points, vario, radius, corrcoef, processes):

    # Seperate observed data with data to predict
    observed_coords = df1[[xx1, yy1]].values.tolist()
    simulate_coords = [coord for coord in prediction_grid.tolist() if coord not in observed_coords]

    # Shuffle data to predict to create a random path
    np.random.shuffle(simulate_coords)

    full = np.vstack((np.array(observed_coords), np.array(simulate_coords)))

    vario[4] = np.var(df1[zz1])

    # create starting index for data from full to use for KNN
    begin = len(observed_coords)

    # create iterable parameter list
    i = [i for i in range(len(simulate_coords))]
    args = zip(i, itertools.cycle([full]), itertools.cycle([df2[[xx2, yy2]].values]), itertools.cycle([vario]), itertools.cycle([radius]),
               itertools.cycle([num_points]), itertools.cycle([begin]), itertools.cycle([corrcoef]))
    
    pool = mp.Pool(processes)

    kr_dictionary = {}
    out = pool.starmap(parallel_cosim, args, chunksize=200)

    for (idx, weights, covariance_array, indicies, idx_df2) in out:
        
        # aggregate output into a dictionary to look up data by index
        kr_dictionary[idx] = (weights, covariance_array, indicies, idx_df2)

    mean_1 = np.average(df1[zz1]) 
    var_1 = np.var(df1[zz1])
    mean_2 = np.average(df2[zz2]) 
    var_2 = np.var(df2[zz2])

    sgs = pred_Z_cosim(kr_dictionary, full, df1[zz1], df2[zz2], mean_1, var_1, mean_2, var_2)
    
    sgs = sgs[np.lexsort((sgs[:,0], -sgs[:,1]))]

    return sgs[:,2]



############################
# Functions run in parallel

def parallel_krige(i, simulate_coords, df, mean, vario, radius, num_points, krig):
    
    loc = simulate_coords[i]
    var = vario[4]

    near = NNS_ele(df, radius, num_points, loc)
    xy_val = near[:,0:2]
    norm_data_val = near[:,2]

    if krig == 's':
        k_weights, covariance_array = skriging(xy_val, loc, vario)

        est_sk = mean + (np.sum(k_weights*(norm_data_val - mean))) 
        var_sk = var - np.sum(k_weights*covariance_array)

    elif krig == 'o':
        k_weights, covariance_array = okriging(xy_val, loc, vario)

        mean = np.mean(norm_data_val)
        num_pts = len(near)

        est_sk = mean + (np.sum(k_weights[:num_pts]*(norm_data_val - mean))) 
        var_sk = var - np.sum(k_weights[:num_pts]*covariance_array[:num_pts])

    if var_sk < 0:
        var_sk = 0

    return i, est_sk, var_sk


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


def parallel_cluster_sgs(i, full, gamma, radius, num_points, begin):

    # offset in all_xyk of location to simulate
    curr_offset = begin + i
    
    loc = full[curr_offset, 0:2]

    search_candidates = full[:curr_offset]

    near, cluster_num, indicies = NNS_cluster(search_candidates, radius, num_points, loc)
    vario = gamma.Variogram[int(cluster_num)]
    k_weights, covariance_array = skriging(near, loc, vario)

    return i, k_weights, covariance_array, indicies, cluster_num


def parallel_cokrige(i, simulate_coords, df1, mean1, var1, df2, mean2, var2, vario, radius, num_points, corrcoef):
    
    loc = simulate_coords[i]

    near1 = NNS_ele(df1, radius, num_points, loc)
    near2, _ = NNS_secondary(df2, loc)
    xy_val = np.append(near1[:,0:2], [near2[0:2]], axis = 0)
    norm_data_val = np.append(near1[:,2], [near2[2]])

    k_weights, covariance_array = cokriging(xy_val, loc, vario, corrcoef)

    num_pts = len(near1)

    part1 = mean1 + (np.sum(k_weights[:num_pts]*(norm_data_val[:num_pts] - mean1))/np.sqrt(var1))
    part2 = (np.sum(k_weights[num_pts]*(norm_data_val[num_pts] - mean2)))/np.sqrt(var2)

    est_cokrig = part1 + part2
    var_cokrig = var1 - np.sum(k_weights*covariance_array)

    return i, est_cokrig, var_cokrig

def parallel_cosim(i, full, df2, vario, radius, num_points, begin, corrcoef):

    curr_offset = begin + i
    
    loc = full[curr_offset]

    search_candidates = full[:curr_offset]

    near1, indicies = NNS(search_candidates, radius, num_points, loc)
    near2, idx = NNS_secondary(df2, loc)

    xy_val = np.append(near1, [near2], axis = 0)

    k_weights, covariance_array = cokriging(xy_val, loc, vario, corrcoef)

    return i, k_weights, covariance_array, indicies, idx


###########################
# Kriging helper functions

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


def cokriging(near, loc, vario, corrcoef):

    numpoints = len(near) - 1

    azimuth = vario[0]
    major_range = vario[2]
    minor_range = vario[3]

    rotation_matrix = gs.make_rotation_matrix(azimuth, major_range, minor_range)

    # covariance between data
    covariance_matrix = np.zeros(shape=((numpoints+1, numpoints+1)))
    covariance_matrix[0:numpoints+1,0:numpoints+1] = gs.Covariance.make_covariance_matrix(near, vario, rotation_matrix)

    # covariance between data and unknown
    covariance_array = np.zeros(shape=(numpoints+1))
    k_weights = np.zeros(shape=(numpoints+1))
    covariance_array[0:numpoints+1] = gs.Covariance.make_covariance_array(near, np.tile(loc, numpoints+1), vario, rotation_matrix)
    covariance_array[numpoints] = covariance_array[numpoints] * corrcoef

    # update covariance matrix with secondary info (gamma2 = rho12 * gamma1)
    covariance_matrix[numpoints, 0 : numpoints+1] = covariance_matrix[numpoints, 0 : numpoints+1] * corrcoef
    covariance_matrix[0 : numpoints+1, numpoints] = covariance_matrix[0 : numpoints+1, numpoints] * corrcoef
    covariance_matrix[numpoints, numpoints] = 1
    covariance_matrix.reshape(((numpoints + 1)), ((numpoints + 1)))

    k_weights, res, rank, s = linalg.lstsq(covariance_matrix, covariance_array, rcond=None)

    return k_weights, covariance_array



#####################################
# Nearest neighbors search functions

def NNS_ele(search_candidates, radius, num_points, loc):

    centered = search_candidates[:,:2] - loc

    angles = np.arctan2(centered[:, 0], centered[:, 1])

    dist = np.linalg.norm(centered, axis = 1)

    radius_filter = dist < radius

    search_candidates = search_candidates[radius_filter,:]
    angles = angles[radius_filter]
    dist = dist[radius_filter]

    sort = np.argsort(dist)

    search_candidates = search_candidates[sort]
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
    nearest = np.ones(shape=(num_points, 3)) * np.nan

    for i in range(8):
    
        octant = search_candidates[oct == i][:oct_count]
        
        for j, row in enumerate(octant):
        
            nearest[i*oct_count+j,:] = row
    
    near = nearest[~np.isnan(nearest)].reshape(-1,3)

    return near


def NNS(search_candidates, radius, num_points, loc):

    centered = search_candidates - loc

    idx = np.arange(len(search_candidates))

    angles = np.arctan2(centered[:, 0], centered[:, 1])

    dist = np.linalg.norm(centered, axis = 1)

    radius_filter = dist < radius

    centered = centered[radius_filter,:]
    angles = angles[radius_filter]
    dist = dist[radius_filter]
    idx = idx[radius_filter]

    sort = np.argsort(dist)

    centered = centered[sort]
    angles = angles[sort]
    dist = dist[sort]
    idx = idx[sort]

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
        indicies[i*oct_count:i*oct_count+len(octant)] = idx[oct == i][:len(octant)]
        
        for j, row in enumerate(octant):
        
            nearest[i*oct_count+j,:] = row + loc
    
    near = nearest[~np.isnan(nearest)].reshape(-1,2)
    indicies = indicies[~np.isnan(indicies)]

    return near, indicies


def NNS_cluster(search_candidates, radius, num_points, loc):

    K_list = search_candidates[:,2]

    centered = search_candidates[:,0:2] - loc

    idx = np.arange(len(search_candidates))

    angles = np.arctan2(centered[:, 0], centered[:, 1])

    dist = np.linalg.norm(centered, axis = 1)

    radius_filter = dist < radius

    centered = centered[radius_filter,:]
    angles = angles[radius_filter]
    dist = dist[radius_filter]
    K_list = K_list[radius_filter]
    idx = idx[radius_filter]

    rand_K = K_list[~np.isnan(K_list)]
    K = random.choice(rand_K)

    sort = np.argsort(dist)

    centered = centered[sort]
    angles = angles[sort]
    dist = dist[sort]
    idx = idx[sort]

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
        indicies[i*oct_count:i*oct_count+len(octant)] = idx[oct == i][:len(octant)]
        
        for j, row in enumerate(octant):
        
            nearest[i*oct_count+j,:] = row + loc
            
    
    near = nearest[~np.isnan(nearest)].reshape(-1,2)
    indicies = indicies[~np.isnan(indicies)]

    return near, K, indicies


def NNS_secondary(search_candidates, loc):

    centered = search_candidates[:,:2] - loc

    dist = np.linalg.norm(centered, axis = 1)

    sort = np.argsort(dist)

    search_candidates = search_candidates[sort]

    return search_candidates[0], sort[0]
    

##################################
# Elevation prediction functions

def pred_Z(kr_dictionary, full, df, vario, krig):
                 
    z_mean = np.average(df.values) # CHANGE TO Z 
    z_lookup = np.zeros(len(full))
    z_lookup[:len(df)] = df.values
    
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


def pred_Z_cluster(kr_dictionary, full, df, gamma):
                 
    z_mean = np.average(df['Nbed'].values) # CHANGE TO Z 
    z_lookup = np.zeros(len(full))
    z_lookup[:len(df)] = df['Nbed'].values
    
    for i in range(len(full) - len(df)):
        
        weights, covariance_array, indicies, cluster_num = kr_dictionary[i]
        near_ele = np.array([z_lookup[int(idx)] for idx in indicies])
        
        vario = gamma.Variogram[int(cluster_num)]
        
        # calculate kriging mean and variance
        est = z_mean + np.sum(weights[:len(near_ele)] * (near_ele - z_mean))
        var = abs(vario[4] - np.sum(weights[:len(near_ele)] * covariance_array[:len(near_ele)]))
        
        z_lookup[len(df) + i] = np.random.default_rng().normal(est, math.sqrt(var))
    
    full = np.column_stack((full, z_lookup))
    
    return full


def pred_Z_cosim(kr_dictionary, full, df1_ele, df2_ele, mean_1, var_1, mean_2, var_2):

    z_lookup = np.zeros(len(full))
    z_lookup[:len(df1_ele)] = df1_ele.values.ravel()

    for i in range(len(full) - len(df1_ele)):
        
        weights, covariance_array, indicies, idx_df2  = kr_dictionary[i]
        near_ele = np.array([z_lookup[int(idx)] for idx in indicies])
        near_ele = np.append(near_ele, [df2_ele[idx_df2]])
                                
        num_pts = len(indicies)
        
        # calculate kriging mean and variance
        part1 = mean_1 + np.sum(weights[:num_pts] * (near_ele[:num_pts] - mean_1)/np.sqrt(var_1))
        part2 = weights[num_pts] * (df2_ele[idx_df2] - mean_2)/np.sqrt(var_2)
        est_cokrige = part1 + part2 
        var_cokrige = var_1 - np.sum(weights*covariance_array)
        var_cokrige = np.absolute(var_cokrige) 
        
        z_lookup[len(df1_ele) + i] = np.random.default_rng().normal(est_cokrige, math.sqrt(var_cokrige))
    
    full = np.column_stack((full, z_lookup))

    return full