import pytest

import os

import pandas as pd
import numpy as np
import random

import sys
sys.path.append("../")

import gstatsim_dev as gs
import parallel


def test_ordinary_kriging():
    """
    Test of ordinary kriging.
    The test is roughly based on demos/3_Simple_kriging_and_ordinary_kriging.ipynb

    """
    # read demo data
    data_file_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '../demos/data/greenland_test_data.csv')
    df_bed = pd.read_csv(data_file_path)

    # grid data to 100 m resolution and remove coordinates with NaNs
    res = 1000
    df_grid, _, _, __import__ = gs.Gridding.grid_data(
        df_bed, 'X', 'Y', 'Bed', res)
    df_grid = df_grid[df_grid["Z"].isnull() == False]

    # define coordinate grid
    xmin = np.min(df_grid['X'])
    xmax = np.max(df_grid['X'])     # min and max x values
    ymin = np.min(df_grid['Y'])
    ymax = np.max(df_grid['Y'])     # min and max y values

    Pred_grid_xy = gs.Gridding.prediction_grid(xmin, xmax, ymin, ymax, res)

    # set variogram parameters
    azimuth = 0
    nugget = 0

    # the major and minor ranges are the same in this example because it is isotropic
    major_range = 19236.
    minor_range = 19236.
    sill = 22399.
    vtype = 'Exponential'

    # save variogram parameters as a list
    vario = [azimuth, nugget, major_range, minor_range, sill, vtype]

    k = 100         # number of neighboring data points used to estimate a given point
    rad = 50000     # 50 km search radius

    num_proc = 10   # set number of parallel processes

    # est_SK is the estimate and var_SK is the variance
    est_SK, var_SK = parallel.okrige(Pred_grid_xy, df_grid, 'X', 'Y', 'Z', k, vario, rad, num_proc)

    expected_est, expected_var = gs.Interpolation.okrige(Pred_grid_xy, df_grid, 'X', 'Y', 'Z', k, vario, rad)

    # assert
    np.testing.assert_array_almost_equal(est_SK, expected_est, decimal=1)
    np.testing.assert_array_almost_equal(var_SK, expected_var, decimal=1)


def test_simple_kriging():
    """
    Test of simple kriging.
    The test is roughly based on demos/3_Simple_kriging_and_ordinary_kriging.ipynb

    """
    # read demo data
    data_file_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '../demos/data/greenland_test_data.csv')
    df_bed = pd.read_csv(data_file_path)

    # grid data to 100 m resolution and remove coordinates with NaNs
    res = 1000
    df_grid, _, _, _ = gs.Gridding.grid_data(
        df_bed, 'X', 'Y', 'Bed', res)
    df_grid = df_grid[df_grid["Z"].isnull() == False]

    # define coordinate grid
    xmin = np.min(df_grid['X'])
    xmax = np.max(df_grid['X'])     # min and max x values
    ymin = np.min(df_grid['Y'])
    ymax = np.max(df_grid['Y'])     # min and max y values

    Pred_grid_xy = gs.Gridding.prediction_grid(xmin, xmax, ymin, ymax, res)

    # set variogram parameters
    azimuth = 0
    nugget = 0

    # the major and minor ranges are the same in this example because it is isotropic
    major_range = 19236.
    minor_range = 19236.
    sill = 22399.
    vtype = 'Exponential'

    # save variogram parameters as a list
    vario = [azimuth, nugget, major_range, minor_range, sill, vtype]

    k = 100         # number of neighboring data points used to estimate a given point
    rad = 50000     # 50 km search radius
    
    num_proc = 10   # set number of parallel processes

    # est_SK is the estimate and var_SK is the variance
    est_SK, var_SK = parallel.skrige(Pred_grid_xy, df_grid, 'X', 'Y', 'Z', k, vario, rad, num_proc)
 
    expected_est, expected_var = gs.Interpolation.skrige(Pred_grid_xy, df_grid, 'X', 'Y', 'Z', k, vario, rad)

    # assert
    np.testing.assert_array_almost_equal(est_SK, expected_est, decimal=1)
    np.testing.assert_array_almost_equal(var_SK, expected_var, decimal=1)


def test_sequential_gaussian_simulation_ordinary_kriging():
    """
    This tests the sequential gaussian simulation with ordinary kriging.
    The test is roughly based on demos/4_Sequential_Gaussian_Simulation.ipynb

    """
    # read demo data
    data_file_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '../demos/data/greenland_test_data.csv')
    df_bed = pd.read_csv(data_file_path)

    # Grid and transform data, compute variogram parameters
    # grid data to 100 m resolution and remove coordinates with NaNs
    res = 1000
    df_grid, _, _, _ = gs.Gridding.grid_data(df_bed, 'X', 'Y', 'Bed', res)

    # remove NaNs
    df_grid = df_grid[df_grid["Z"].isnull() == False]

    # Initialize grid
    # define coordinate grid
    xmin = np.min(df_grid['X'])
    xmax = np.max(df_grid['X'])     # min and max x values

    ymin = np.min(df_grid['Y'])
    ymax = np.max(df_grid['Y'])     # min and max y values

    Pred_grid_xy = gs.Gridding.prediction_grid(xmin, xmax, ymin, ymax, res)
    
    rng = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(123456789)))
    saved_state = rng.get_state() 

    # Sequential Gaussian simulation
    # set variogram parameters
    azimuth = 0
    nugget = 0
    k = 48         # number of neighboring data points used to estimate a given point
    rad = 50000    # 50 km search radius

    # the major and minor ranges are the same in this example because it is isotropic
    major_range = minor_range = 31852.
    sill = 0.7
    vtype = 'Exponential'
    
    num_proc = 10   # set number of parallel processes

    # save variogram parameters as a list
    vario = [azimuth, nugget, major_range, minor_range, sill, vtype]
    
    df_grid = df_grid.reset_index(drop=True)
    df_grid = df_grid.reindex(index = np.lexsort((df_grid['X'], -df_grid['Y'])))
    
    # ordinary kriging
    sim = parallel.okrige_sgs(Pred_grid_xy, df_grid, 'X', 'Y', 'Z', k, vario, rad, num_proc, seed=rng)
    
    rng.set_state(saved_state)

    # as we set the numpy random seed, the simulation is deterministic and we can compare to the following (rounded) results
    expected_sim = gs.Interpolation.okrige_sgs(Pred_grid_xy, df_grid, 'X', 'Y', 'Z', k, vario, rad, seed=rng)

    # assert
    np.testing.assert_array_almost_equal(sim, expected_sim, decimal=1)


def test_sequential_gaussian_simulation_simple_kriging():
    """
    This tests the sequential gaussian simulation with simple kriging.
    The test is roughly based on demos/4_Sequential_Gaussian_Simulation.ipynb

    """
    data_file_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '../demos/data/greenland_test_data.csv')
    df_bed = pd.read_csv(data_file_path)

    # Grid and transform data, compute variogram parameters
    # grid data to 100 m resolution and remove coordinates with NaNs
    res = 1000
    df_grid, grid_matrix, rows, cols = gs.Gridding.grid_data(
        df_bed, 'X', 'Y', 'Bed', res)

    # remove NaNs
    df_grid = df_grid[df_grid["Z"].isnull() == False]

    # maximum range distance
    maxlag = 50000
    # num of bins
    n_lags = 70

    # Initialize grid
    # define coordinate grid
    xmin = np.min(df_grid['X'])
    xmax = np.max(df_grid['X'])     # min and max x values

    ymin = np.min(df_grid['Y'])
    ymax = np.max(df_grid['Y'])     # min and max y values

    Pred_grid_xy = gs.Gridding.prediction_grid(xmin, xmax, ymin, ymax, res)
    
    rng = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(123456789)))
    saved_state = rng.get_state() 

    # Sequential Gaussian simulation
    # set variogram parameters
    azimuth = 0
    nugget = 0
    k = 48         # number of neighboring data points used to estimate a given point
    rad = 50000    # 50 km search radius

    # the major and minor ranges are the same in this example because it is isotropic
    major_range = minor_range = 31852.
    sill = 0.7
    vtype = 'Exponential'
    
    num_proc = 10   # set number of parallel processes

    # save variogram parameters as a list
    vario = [azimuth, nugget, major_range, minor_range, sill, vtype]
    
    df_grid = df_grid.reset_index(drop=True)
    df_grid = df_grid.reindex(index = np.lexsort((df_grid['X'], -df_grid['Y'])))

    # simple kriging
    sim = parallel.skrige_sgs(Pred_grid_xy, df_grid, 'X', 'Y', 'Z', k, vario, rad, num_proc, seed=rng)
    
    rng.set_state(saved_state)

    # as we set the numpy random seed, the simulation is deterministic and we can compare to the following (rounded) results
    expected_sim = gs.Interpolation.skrige_sgs(Pred_grid_xy, df_grid, 'X', 'Y', 'Z', k, vario, rad, seed=rng)

    # assert
    np.testing.assert_array_almost_equal(sim, expected_sim, decimal=1)


if __name__ == '__main__':
    import pytest
    pytest.main()
