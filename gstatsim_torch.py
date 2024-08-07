
#!/usr/bin/env python
# coding: utf-8

### geostatistical tools

import numpy as np
import numpy.linalg as linalg
import pandas as pd
import sklearn as sklearn
from sklearn.neighbors import KDTree
import math
from scipy.spatial import distance_matrix
from scipy.interpolate import Rbf
from tqdm import tqdm
import random
from sklearn.metrics import pairwise_distances
import torch
import cProfile, pstats, io


def profile(fnc):
    
    """A decorator that uses cProfile to profile a function"""
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner

############################

# Grid data

############################

class Gridding:
    def prediction_grid(xmin, xmax, ymin, ymax, res):
        """
        Make prediction grid
        
        Parameters
        ----------
            xmin : float, int
                minimum x extent
            xmax : float, int
                maximum x extent
            ymin : float, int
                minimum y extent
            ymax : float, int
                maximum y extent
            res : float, int
                grid cell resolution
        
        Returns
        -------
            prediction_grid_xy : numpy.ndarray
                x,y array of coordinates
        """ 
        
        cols = torch.ceil((xmax - xmin + res)/res)
        rows = torch.ceil((ymax - ymin + res)/res)  
        x = torch.linspace(xmin, xmin+(cols*res) - res, steps=int(cols))
        y = torch.linspace(ymin, ymin+(rows*res) - res, steps=int(rows))
        xx, yy = torch.meshgrid(x, y, indexing='xy') 
        yy = torch.flip(yy, [0, 1]) 
        x = torch.reshape(xx, (int(rows)*int(cols), 1))
        y = torch.reshape(yy, (int(rows)*int(cols), 1))
        prediction_grid_xy = torch.cat((x,y), axis = 1)
        
        return prediction_grid_xy
 
    def make_grid(xmin, xmax, ymin, ymax, res):
        """
        Generate coordinates for output of gridded data  
        
        Parameters
        ----------
            xmin : float, int
                minimum x extent
            xmax : float, int
                maximum x extent
            ymin : float, int
                minimum y extent
            ymax : float, int
                maximum y extent
            res : float, int
                grid cell resolution
        
        Returns
        -------
            prediction_grid_xy : numpy.ndarray
                x,y array of coordinates
            rows : int
                number of rows 
            cols : int 
                number of columns
        """ 
        
        cols = int(torch.ceil((xmax - xmin)/res))
        rows = int(torch.ceil((ymax - ymin)/res))
        x = torch.arange(xmin,xmax,res); y = torch.arange(ymin,ymax,res)
        xx, yy = torch.meshgrid(x,y, indexing='xy') 
        x = torch.reshape(xx, (rows*cols, 1)) 
        y = torch.reshape(yy, (rows*cols, 1))
        prediction_grid_xy = torch.cat((x,y), axis = 1)
        
        return prediction_grid_xy, cols, rows
    
    def grid_data(df, xx, yy, zz, res):
        """
        Grid conditioning data
        
        Parameters
        ----------
            df : pandas DataFrame 
                dataframe of conditioning data and coordinates
            xx : string 
                column name for x coordinates of input data frame
            yy : string
                column name for y coordinates of input data frame
            zz : string
                column for z values (or data variable) of input data frame
            res : float, int
                grid cell resolution
        
        Returns
        -------
            df_grid : pandas DataFrame
                dataframe of gridded data
            grid_matrix : numpy.ndarray
                matrix of gridded data
            rows : int
                number of rows in grid_matrix
            cols : int
                number of columns in grid_matrix
        """ 
        df = df.rename(columns = {xx: "X", yy: "Y", zz: "Z"})
        data = torch.tensor(df.values)

        xmin = torch.min(data[:,0])
        xmax = torch.max(data[:,0])
        ymin = torch.min(data[:,1])
        ymax = torch.max(data[:,1])
        
        # make array of grid coordinates
        grid_coord, cols, rows = Gridding.make_grid(xmin, xmax, ymin, ymax, res) 

        origin = torch.tensor([xmin,ymin])
        resolution = torch.tensor([res,res])
        
        # shift and re-scale the data by subtracting origin and dividing by resolution
        data[:,:2] = torch.round((data[:,:2]-origin)/resolution) 

        grid_sum = torch.zeros((rows,cols))
        grid_count = torch.zeros((rows,cols))

        for i in range(data.shape[0]):
            xindex = data[i,1].type(torch.int32)
            yindex = data[i,0].type(torch.int32)

            if ((xindex >= rows) | (yindex >= cols)):
                continue

            grid_sum[xindex,yindex] += data[i,2]
            grid_count[xindex,yindex] += 1 


        grid_matrix = grid_sum / grid_count 
        grid_array = torch.reshape(grid_matrix,[rows*cols]) 
        grid_sum = torch.reshape(grid_sum,[rows*cols]) 
        grid_count = torch.reshape(grid_count,[rows*cols]) 

        # make dataframe    
        grid_total = torch.vstack((grid_coord[:,0], grid_coord[:,1], 
                               grid_sum, grid_count, grid_array)).T   
        df_grid = pd.DataFrame(grid_total, 
                               columns = ['X', 'Y', 'Sum', 'Count', 'Z']) 
        torch_data = torch.vstack((grid_coord[:,0], grid_coord[:,1], grid_array)).T   
        
        return df_grid, torch_data, rows, cols


###################################

# RBF trend estimation

###################################

def rbf_trend(grid_matrix, smooth_factor, res):
    """
    Estimate trend using radial basis functions
    
    Parameters
    ----------
        grid_matrix : numpy.ndarray
            matrix of gridded conditioning data
        smooth_factor : float
            Parameter controlling smoothness of trend. Values greater than 
            zero increase the smoothness of the approximation.
        res : float
            grid cell resolution
            
    Returns
    -------
        trend_rbf : numpy.ndarray
            RBF trend estimate
    """ 
    sigma = np.rint(smooth_factor/res)
    ny, nx = grid_matrix.shape
    rbfi = Rbf(np.where(~np.isnan(grid_matrix))[1],
               np.where(~np.isnan(grid_matrix))[0], 
               grid_matrix[~np.isnan(grid_matrix)],smooth = sigma)

    # evaluate RBF
    yi = np.arange(nx)
    xi = np.arange(ny)
    xi,yi = np.meshgrid(xi, yi, indexing='xy')
    trend_rbf = rbfi(xi, yi)   
    
    return trend_rbf


####################################

# Nearest neighbor octant search

####################################

class NearestNeighbor:

    def center(arrayx, arrayy, centerx, centery):
        """
        Shift data points so that grid cell of interest is at the origin
        
        Parameters
        ----------
            arrayx : numpy.ndarray
                x coordinates of data
            arrayy : numpy.ndarray
                y coordinates of data
            centerx : float
                x coordinate of grid cell of interest
            centery : float
                y coordinate of grid cell of interest
        
        Returns
        -------
            centered_array : torch.tensor
                array of coordinates that are shifted with respect to grid cell of interest
        """ 
        
        centerx = arrayx - centerx
        centery = arrayy - centery
        centered_array = torch.stack((centerx, centery), dim=1)
        
        return centered_array

    def distance_calculator(centered_array):
        """
        Compute distances between coordinates and the origin
        
        Parameters
        ----------
            centered_array : torch.tensor
                array of coordinates
        
        Returns
        -------
            dist : torch.tensor
                array of distances between coordinates and origin
        """ 
        
        dist = torch.linalg.norm(centered_array, axis=1)
        
        return dist

    def angle_calculator(centered_array):
        """
        Compute angles between coordinates and the origin
        
        Parameters
        ----------
            centered_array : torch.tensor
                array of coordinates
        
        Returns
        -------
            angles : torch.tensor
                array of angles between coordinates and origin
        """ 
        
        angles = torch.atan2(centered_array[:, 0], centered_array[:, 1])
        
        return angles
      
    def nearest_neighbor_search(radius, num_points, loc, data2, device):
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

    
    def nearest_neighbor_search_cluster(radius, num_points, loc, data2):
        """
        Nearest neighbor octant search when doing sgs with clusters
        
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
            cluster_number : int
                nearest neighbor cluster number
        """ 
        
        locx = loc[0]
        locy = loc[1]
        data = data2.copy()
        centered_array = NearestNeighbor.center(data['X'].values, data['Y'].values, 
                                locx, locy)
        data["dist"] = NearestNeighbor.distance_calculator(centered_array).cpu().numpy()
        data["angles"] = NearestNeighbor.angle_calculator(centered_array).cpu().numpy()
        data = data[data.dist < radius] 
        data = data.sort_values('dist', ascending = True)
        data = data.reset_index() 
        cluster_number = data.K[0]
        # look into numpy
        bins = [-math.pi, -3*math.pi/4, -math.pi/2, -math.pi/4, 0, 
                math.pi/4, math.pi/2, 3*math.pi/4, math.pi]
        data["Oct"] = pd.cut(data.angles, bins = bins, labels = list(range(8))) 
        oct_count = num_points // 8
        smallest = np.ones(shape=(num_points, 3)) * np.nan

        for i in range(8):
            octant = data[data.Oct == i].iloc[:oct_count][['X','Y','Z']].values
            for j, row in enumerate(octant):
                smallest[i*oct_count+j,:] = row 
        near = smallest[~np.isnan(smallest)].reshape(-1,3) 
        
        return near, cluster_number

    def nearest_neighbor_secondary(loc, data2):
        """
        Find the neareset neighbor secondary data point to grid cell of interest
        
        Parameters
        ----------
            loc : numpy.ndarray
                coordinates for grid cell of interest
            data2 : pandas DataFrame
                secondary data
        
        Returns
        ------- 
            nearest_second : float
                nearest neighbor value to secondary data
        """ 
        
        locx = loc[0]
        locy = loc[1]
        data = data2.copy()
        centered_array = NearestNeighbor.center(data['X'].values, data['Y'].values, 
                                locx, locy)
        data["dist"] = NearestNeighbor.distance_calculator(centered_array)
        data = data.sort_values('dist', ascending = True) 
        data = data.reset_index() 
        nearest_second = data.iloc[0][['X','Y','Z']].values 
        
        return nearest_second

    def find_colocated(df1, xx1, yy1, zz1, df2, xx2, yy2, zz2): 
        """
        Find colocated data between primary and secondary variables
        
        Parameters
        ----------
            df1 : pandas DataFrame
                data frame of primary conditioning data
            xx1 : string
                column name for x coordinates of input data frame for primary data
            yy1 : string
                column name for y coordinates of input data frame for primary data
            zz1 : string
                column for z values (or data variable) of input data frame for primary data
            df2 : pandas DataFrame
                data frame of secondary data
            xx2 : string
                column name for x coordinates of input data frame for secondary data
            yy2 : string
                column name for y coordinates of input data frame for secondary data
            zz2 : string
                column for z values (or data variable) of input data frame for secondary data
        Returns
        -------
            df_colocated : pandas DataFrame
                data frame of colocated values
        """ 
        
        df1 = df1.rename(columns = {xx1: "X", yy1: "Y", zz1: "Z"}) 
        df2 = df2.rename(columns = {xx2: "X", yy2: "Y", zz2: "Z"}) 
        secondary_variable_xy = df2[['X','Y']].values
        secondary_variable_tree = KDTree(secondary_variable_xy) 
        primary_variable_xy = df1[['X','Y']].values
        nearest_indices = np.zeros(len(primary_variable_xy)) 

        # query search tree
        for i in range(0,len(primary_variable_xy)):
            nearest_indices[i] = secondary_variable_tree.query(primary_variable_xy[i:i+1,:],
                                                               k=1,return_distance=False)
        nearest_indices = np.transpose(nearest_indices)
        secondary_data = df2['Z']
        colocated_secondary_data = secondary_data[nearest_indices]
        df_colocated = pd.DataFrame(np.array(colocated_secondary_data).T, columns = ['colocated'])
        df_colocated.reset_index(drop=True, inplace=True)

        return df_colocated

###############################

# adaptive partioning

###############################

def adaptive_partitioning(df_data, xmin, xmax, ymin, ymax, i, max_points, min_length, max_iter=None):
    """
    Rercursively split clusters until they are all below max_points, but don't go smaller than min_length
    
    Parameters
    ----------
        df_data : pandas DataFrame 
            DataFrame with X, Y, and K (cluster id) columns
        xmin : float
            min x value of this partion
        xmax : float
            max x value of this partion
        ymin : float
            min y value of this partion
        ymax : float
            max y value of this partion
        i : int
            keeps track of total calls to this function
        max_points : int
            all clusters will be "quartered" until points below this
        min_length : float
            minimum side length of sqaures, preference over max_points
        max_iter : int
            maximum iterations if worried about unending recursion
    Returns
    -------
        df_data : pandas DataFrame
            updated DataFrame with new cluster assigned
        i : int
            number of iterations
    """
    # optional 'safety' if there is concern about runaway recursion
    if max_iter is not None:
        if i >= max_iter:
            return df_data, i
    
    dx = xmax - xmin
    dy = ymax - ymin
    
    # >= and <= greedy so we don't miss any points
    xleft = (df_data.X >= xmin) & (df_data.X <= xmin+dx/2)
    xright = (df_data.X <= xmax) & (df_data.X >= xmin+dx/2)
    ybottom = (df_data.Y >= ymin) & (df_data.Y <= ymin+dy/2)
    ytop = (df_data.Y <= ymax) & (df_data.Y >= ymin+dy/2)
    
    # index the current cell into 4 quarters
    q1 = df_data.loc[xleft & ybottom]
    q2 = df_data.loc[xleft & ytop]
    q3 = df_data.loc[xright & ytop]
    q4 = df_data.loc[xright & ybottom]
    
    # for each quarter, qaurter if too many points, else assign K and return
    for q in [q1, q2, q3, q4]:
        if (q.shape[0] > max_points) & (dx/2 > min_length):
            i = i+1
            df_data, i = adaptive_partitioning(df_data, q.X.min(), 
                                               q.X.max(), q.Y.min(), q.Y.max(), i, 
                                               max_points, min_length, max_iter)
        else:
            qcount = df_data.K.max()
            # ensure zero indexing
            if np.isnan(qcount) == True:
                qcount = 0
            else:
                qcount += 1
            df_data.loc[q.index, 'K'] = qcount
            
    return df_data, i


#########################

# Rotation Matrix

#########################

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


###########################

# Covariance functions

###########################

class Covariance:
    
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
        covariance_matrix = Covariance.covar(effective_lag, sill, nug, vtype)

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
        covariance_array = Covariance.covar(effective_lag, sill, nug, vtype)

        return covariance_array

######################################

# Simple Kriging Function

######################################

class Interpolation: 

    def skrige(prediction_grid, data, num_points, vario, radius, quiet=False):
        """
        Simple kriging interpolation
        
        Parameters
        ----------
            prediction_grid : numpy.ndarray
                x,y coordinate numpy array of prediction grid, or grid cells that will be estimated
            df : pandas DataFrame
                data frame of conditioning data
            xx : string
                column name for x coordinates of input data frame
            yy : string 
                column name for y coordinates of input data frame
            zz : string
                column for z values (or data variable) of input data frame
            num_points : int
                the number of conditioning points to search for
            vario : list
                list of variogram parameters [azimuth, nugget, major_range, minor_range, sill, vtype]
                azimuth, nugget, major_range, minor_range, and sill can be int or float type
                vtype is a string that can be either 'Exponential', 'Spherical', or 'Gaussian'
            radius : int, float
                search radius
            quiet : bool
                If False, a progress bar will be printed to the console.
               Default is False
        
        Returns
        -------
            est_sk : numpy.ndarray
                simple kriging estimate for each coordinate in prediction_grid
            var_sk : numpy.ndarray 
                simple kriging variance 
        """
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # Unpack variogram parameters
        azimuth = vario[0]
        major_range = vario[2]
        minor_range = vario[3]
        rotation_matrix = make_rotation_matrix(azimuth, major_range, minor_range, device)

        # X, Y, Z tensor
        data = data.to(device)
        
        mean_1 = data[:,2].mean() 
        var_1 = vario[4]
        est_sk = torch.zeros(len(prediction_grid), device=device)
        var_sk = torch.zeros(len(prediction_grid), device=device)

        # Convert prediction_grid to tensor
        prediction_grid = prediction_grid.to(device)

        # build the iterator
        if not quiet:
            _iterator = enumerate(tqdm(prediction_grid, position=0, leave=True))
        else:
            _iterator = enumerate(prediction_grid)

        # for each coordinate in the prediction grid
        for z, predxy in enumerate(_iterator):
            test_idx = torch.all(torch.eq(data[:, :2], prediction_grid[z]), dim=1)
            if not test_idx.any():
                # gather nearest points within radius
                nearest = NearestNeighbor.nearest_neighbor_search(radius, num_points, prediction_grid[z], data, device)
                    
                norm_data_val = nearest[:, -1]
                xy_val = nearest[:, :-1]
                new_num_pts = nearest.shape[0]

                # covariance between data
                covariance_matrix = Covariance.make_covariance_matrix(xy_val, vario, rotation_matrix)

                # covariance between data and unknown
                covariance_array = Covariance.make_covariance_array(
                    xy_val, 
                    prediction_grid[z].unsqueeze(0).repeat(new_num_pts, 1), 
                    vario, 
                    rotation_matrix
                )

                k_weights = torch.linalg.lstsq(covariance_matrix, covariance_array.unsqueeze(-1)).solution.squeeze(-1)
                
                est_sk[z] = mean_1 + torch.dot(k_weights, (norm_data_val - mean_1).to(dtype=torch.float64))
                var_sk[z] = var_1 - torch.dot(k_weights, covariance_array)
                var_sk[z] = torch.clamp(var_sk[z], min=0)
            else:
                est_sk[z] = data[test_idx, 2].item()
                var_sk[z] = 0

        est_sk, var_sk = est_sk.cpu().numpy(), var_sk.cpu().numpy()
        return est_sk, var_sk

    def okrige(prediction_grid, data, num_points, vario, radius, quiet=False):
        """
        Ordinary kriging interpolation
        
        Parameters
        ----------
            prediction_grid : numpy.ndarray
                x,y coordinate numpy array of prediction grid, or grid cells that will be estimated
            df : pandas DataFrame
                data frame of conditioning data
            xx : string
                column name for x coordinates of input data frame
            yy : string 
                column name for y coordinates of input data frame
            zz : string
                column for z values (or data variable) of input data frame
            num_points : int
                the number of conditioning points to search for
            vario : list
                list of variogram parameters [azimuth, nugget, major_range, minor_range, sill, vtype]
                azimuth, nugget, major_range, minor_range, and sill can be int or float type
                vtype is a string that can be either 'Exponential', 'Spherical', or 'Gaussian'
            radius : int, float
                search radius
            quiet : bool
                If False, a progress bar will be printed to the console.
               Default is False
        
        Returns
        -------
            est_ok : numpy.ndarray
                ordinary kriging estimate for each coordinate in prediction_grid
            var_ok : numpy.ndarray
                ordinary kriging variance 
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # Unpack variogram parameters
        azimuth = vario[0]
        major_range = vario[2]
        minor_range = vario[3]
        rotation_matrix = make_rotation_matrix(azimuth, major_range, minor_range, device)

        # X, Y, Z tensor
        data = data.to(device)
        
        var_1 = vario[4]
        est_ok = torch.zeros(len(prediction_grid), device=device)
        var_ok = torch.zeros(len(prediction_grid), device=device)

        # Convert prediction_grid to tensor
        prediction_grid = prediction_grid.to(device)

        # build the iterator
        if not quiet:
            _iterator = enumerate(tqdm(prediction_grid, position=0, leave=True))
        else:
            _iterator = enumerate(prediction_grid)

        # for each coordinate in the prediction grid
        for z, predxy in enumerate(_iterator):
            test_idx = torch.all(torch.eq(data[:, :2], prediction_grid[z]), dim=1)
            if not test_idx.any():
                # gather nearest points within radius
                nearest = NearestNeighbor.nearest_neighbor_search(radius, num_points, prediction_grid[z], data, device)
                    
                norm_data_val = nearest[:, -1]
                
                local_mean = torch.mean(norm_data_val)
                
                xy_val = nearest[:, :-1]
                new_num_pts = nearest.shape[0]

                # covariance between data
                covariance_matrix = torch.ones((new_num_pts+1, new_num_pts+1), device=device, dtype=torch.float64)
                covariance_matrix[0:new_num_pts,0:new_num_pts] = Covariance.make_covariance_matrix(xy_val, vario, rotation_matrix)

                # covariance between data and unknown
                covariance_array = torch.ones((new_num_pts+1), device=device, dtype=torch.float64) 
                covariance_array[0:new_num_pts] = Covariance.make_covariance_array(
                    xy_val, 
                    prediction_grid[z].unsqueeze(0).repeat(new_num_pts, 1), 
                    vario, 
                    rotation_matrix
                )

                k_weights = torch.linalg.lstsq(covariance_matrix, covariance_array.unsqueeze(-1)).solution.squeeze(-1)
                
                est_ok[z] = local_mean + torch.dot(k_weights[0:new_num_pts], (norm_data_val - local_mean).to(dtype=torch.float64))
                var_ok[z] = var_1 - torch.dot(k_weights[0:new_num_pts], covariance_array[0:new_num_pts])
                var_ok[z] = torch.clamp(var_ok[z], min=0)
            else:
                est_ok[z] = data[test_idx, 2].item()
                var_ok[z] = 0

        est_ok, var_ok = est_ok.cpu().numpy(), var_ok.cpu().numpy()
        return est_ok, var_ok
  
    def skrige_sgs(prediction_grid, data, num_points, vario, radius, quiet=False):
        """
        Sequential Gaussian simulation using simple kriging 
        
        Parameters
        ----------
            prediction_grid : numpy.ndarray
                x,y coordinate numpy array of prediction grid, or grid cells that will be estimated
            df : pandas DataFrame
                data frame of conditioning data
            xx : string
                column name for x coordinates of input data frame
            yy : string 
                column name for y coordinates of input data frame
            zz : string
                column for z values (or data variable) of input data frame
            num_points : int
                the number of conditioning points to search for
            vario : list
                list of variogram parameters [azimuth, nugget, major_range, minor_range, sill, vtype]
                azimuth, nugget, major_range, minor_range, and sill can be int or float type
                vtype is a string that can be either 'Exponential', 'Spherical', or 'Gaussian'
            radius : int, float
                search radius
            quiet : bool
                If False, a progress bar will be printed to the console.
               Default is False
        
        Returns
        -------
            sgs : numpy.ndarray
                simulated value for each coordinate in prediction_grid
        """
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # unpack variogram parameters
        azimuth = vario[0]
        major_range = vario[2]
        minor_range = vario[3]
        rotation_matrix = make_rotation_matrix(azimuth, major_range, minor_range, device) 
        
        # X, Y, Z tensor
        data = data.to(device)

        xyindex = torch.arange(len(prediction_grid)) 
        xyindex = xyindex[torch.randperm(len(prediction_grid))]
        mean_1 = data[:,2].mean() 
        var_1 = vario[4]
        sgs = torch.zeros(len(prediction_grid), device=device) 

        # build the iterator
        if not quiet:
            _iterator = enumerate(tqdm(prediction_grid, position=0, leave=True))
        else:
            _iterator = enumerate(prediction_grid)

        for idx, predxy in _iterator:
            z = xyindex[idx] 
            test_idx = torch.all(torch.eq(data[:, :2], prediction_grid[z]), dim=1)
            if not test_idx.any():
                
                # get nearest neighbors
                nearest = NearestNeighbor.nearest_neighbor_search(radius, num_points, prediction_grid[z],
                                                                  data, device)  
                norm_data_val = nearest[:,-1]   
                xy_val = nearest[:,:-1]   
                new_num_pts = len(nearest) 

                # covariance between data
                covariance_matrix = Covariance.make_covariance_matrix(xy_val, vario, rotation_matrix)

                # covariance between data and unknown
                covariance_array = Covariance.make_covariance_array(xy_val, 
                    prediction_grid[z].unsqueeze(0).repeat(new_num_pts, 1), 
                    vario, 
                    rotation_matrix
               )
                 
                k_weights = torch.linalg.lstsq(covariance_matrix, 
                                               covariance_array.unsqueeze(-1)).solution.squeeze(-1)
                # get estimates
                est =  mean_1 + torch.dot(k_weights, (norm_data_val - mean_1).to(dtype=torch.float64)) 
                var = var_1 - torch.dot(k_weights, covariance_array)
                var = torch.absolute(var) 
                sgs[z] = torch.normal(est,torch.sqrt(var)) 
            else:
                sgs[z] = data[test_idx, 2].item()

            coords = prediction_grid[z,:]

            new = torch.cat((torch.squeeze(coords), sgs[z].reshape(1)))
            data = torch.cat((data,new.unsqueeze(0)), dim=0) 
                             
        return data
   
    def okrige_sgs(prediction_grid, df, xx, yy, zz, num_points, vario, radius, quiet=False):
        """
        Sequential Gaussian simulation using ordinary kriging 
        
        Parameters
        ----------
            prediction_grid : numpy.ndarray
                x,y coordinate numpy array of prediction grid, or grid cells that will be estimated
            df : pandas DataFrame
                data frame of conditioning data
            xx : string
                column name for x coordinates of input data frame
            yy : string 
                column name for y coordinates of input data frame
            zz : string
                column for z values (or data variable) of input data frame
            num_points : int
                the number of conditioning points to search for
            vario : list
                list of variogram parameters [azimuth, nugget, major_range, minor_range, sill, vtype]
                azimuth, nugget, major_range, minor_range, and sill can be int or float type
                vtype is a string that can be either 'Exponential', 'Spherical', or 'Gaussian'
            radius : int, float
                search radius
            quiet : bool
                If False, a progress bar will be printed to the console.
               Default is False
        
        Returns
        -------
            sgs : numpy.ndarray
                simulated value for each coordinate in prediction_grid
        """

        # unpack variogram parameters
        azimuth = vario[0]
        major_range = vario[2]
        minor_range = vario[3]
        rotation_matrix = make_rotation_matrix(azimuth, major_range, minor_range) 

        df = df.rename(columns = {xx: "X", yy: "Y", zz: "Z"}) 
        xyindex = np.arange(len(prediction_grid)) 
        random.shuffle(xyindex)
        var_1 = vario[4]
        sgs = np.zeros(shape=len(prediction_grid))  

        # build the iterator
        if not quiet:
            _iterator = enumerate(tqdm(prediction_grid, position=0, leave=True))
        else:
            _iterator = enumerate(prediction_grid)

        for idx, predxy in _iterator:
            z = xyindex[idx] 
            test_idx = np.sum(prediction_grid[z]==df[['X', 'Y']].values,axis = 1)
            if np.sum(test_idx==2)==0:
                
                # gather nearest neighbor points
                nearest = NearestNeighbor.nearest_neighbor_search(radius, num_points, 
                                                  prediction_grid[z], df[['X','Y','Z']]) 
                norm_data_val = nearest[:,-1]   
                xy_val = nearest[:,:-1]   
                local_mean = np.mean(norm_data_val) 
                new_num_pts = len(nearest) 

                # covariance between data
                covariance_matrix = np.zeros(shape=((new_num_pts+1, new_num_pts+1))) 
                covariance_matrix[0:new_num_pts,0:new_num_pts] = Covariance.make_covariance_matrix(xy_val, 
                                                                                        vario, rotation_matrix)
                covariance_matrix[new_num_pts,0:new_num_pts] = 1
                covariance_matrix[0:new_num_pts,new_num_pts] = 1

                # Set up Right Hand Side (covariance between data and unknown)
                covariance_array = np.zeros(shape=(new_num_pts+1))
                k_weights = np.zeros(shape=(new_num_pts+1))
                covariance_array[0:new_num_pts] = Covariance.make_covariance_array(xy_val, 
                                                                        np.tile(prediction_grid[z], new_num_pts), 
                                                                        vario, rotation_matrix)
                covariance_array[new_num_pts] = 1 
                covariance_matrix.reshape(((new_num_pts+1)), ((new_num_pts+1)))

                k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix, 
                                                          covariance_array, rcond = None)           
                est = local_mean + np.sum(k_weights[0:new_num_pts]*(norm_data_val - local_mean)) 
                var = var_1 - np.sum(k_weights[0:new_num_pts]*covariance_array[0:new_num_pts]) 
                var = np.absolute(var)

                sgs[z] = np.random.normal(est,math.sqrt(var),1) 
            else:
                sgs[z] = df['Z'].values[np.where(test_idx==2)[0][0]] 

            coords = prediction_grid[z:z+1,:]
            df = pd.concat([df,pd.DataFrame({'X': [coords[0,0]], 'Y': [coords[0,1]], 'Z': [sgs[z]]})], sort=False) 

        return sgs


    def cluster_sgs(prediction_grid, df, xx, yy, zz, kk, num_points, df_gamma, radius, quiet=False):
        """
        Sequential Gaussian simulation where variogram parameters are different for each k cluster. Uses simple kriging 
        
        Parameters
        ----------
            prediction_grid : numpy.ndarray
                x,y coordinate numpy array of prediction grid, or grid cells that will be estimated
            df : pandas DataFrame
                data frame of conditioning data
            xx : string
                column name for x coordinates of input data frame
            yy : string 
                column name for y coordinates of input data frame
            zz : string
                column for z values (or data variable) of input data frame
            kk : string
                column of k cluster numbers for each point
            num_points : int
                the number of conditioning points to search for
            vario : list
                list of variogram parameters [azimuth, nugget, major_range, minor_range, sill, vtype]
                azimuth, nugget, major_range, minor_range, and sill can be int or float type
                vtype is a string that can be either 'Exponential', 'Spherical', or 'Gaussian'
            radius : int, float
                search radius
            quiet : bool
                If False, a progress bar will be printed to the console.
               Default is False
        
        Returns
        -------
            sgs : numpy.ndarray
                simulated value for each coordinate in prediction_grid
        """

        df = df.rename(columns = {xx: "X", yy: "Y", zz: "Z", kk: "K"})  
        xyindex = np.arange(len(prediction_grid)) 
        random.shuffle(xyindex)
        mean_1 = np.average(df["Z"].values) 
        sgs = np.zeros(shape=len(prediction_grid)) 

        # build the iterator
        if not quiet:
            _iterator = enumerate(tqdm(prediction_grid, position=0, leave=True))
        else:
            _iterator = enumerate(prediction_grid)

        for idx, predxy in _iterator:
            z = xyindex[idx] 
            test_idx = np.sum(prediction_grid[z]==df[['X', 'Y']].values,axis = 1)
            if np.sum(test_idx==2)==0: 
                
                # gather nearest neighbor points and K cluster value
                nearest, cluster_number = NearestNeighbor.nearest_neighbor_search_cluster(radius, 
                                                                                          num_points, 
                                                                                          prediction_grid[z],
                                                                                          df[['X','Y','Z','K']])  
                vario = df_gamma.Variogram[cluster_number] 
                norm_data_val = nearest[:,-1]   
                xy_val = nearest[:,:-1]   

                # unpack variogram parameters
                azimuth = vario[0]
                major_range = vario[2]
                minor_range = vario[3]
                var_1 = vario[4]
                rotation_matrix = make_rotation_matrix(azimuth, major_range, minor_range) 
                new_num_pts = len(nearest)

                # covariance between data
                covariance_matrix = np.zeros(shape=((new_num_pts, new_num_pts))) 
                covariance_matrix[0:new_num_pts,0:new_num_pts] = Covariance.make_covariance_matrix(xy_val, 
                                                                                                   vario, 
                                                                                                   rotation_matrix) 
                
                # covariance between data and unknown
                covariance_array = np.zeros(shape=(new_num_pts)) 
                k_weights = np.zeros(shape=(new_num_pts))
                covariance_array = Covariance.make_covariance_array(xy_val, np.tile(prediction_grid[z], new_num_pts), 
                                                                    vario, rotation_matrix)
                covariance_matrix.reshape(((new_num_pts)), ((new_num_pts)))
                k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix, covariance_array, rcond = None) 
                est = mean_1 + np.sum(k_weights*(norm_data_val - mean_1)) 
                var = var_1 - np.sum(k_weights*covariance_array)
                var = np.absolute(var) 

                sgs[z] = np.random.normal(est,math.sqrt(var),1) 
            else:
                sgs[z] = df['Z'].values[np.where(test_idx==2)[0][0]]
                cluster_number = df['K'].values[np.where(test_idx==2)[0][0]]

            coords = prediction_grid[z:z+1,:] 
            df = pd.concat([df,pd.DataFrame({'X': [coords[0,0]], 'Y': [coords[0,1]], 
                                             'Z': [sgs[z]], 'K': [cluster_number]})], sort=False)

        return sgs

    def cokrige_mm1(prediction_grid, df1, xx1, yy1, zz1, df2, xx2, yy2, zz2, num_points, vario, radius, corrcoef, quiet=False):
        """
        Simple collocated cokriging under Markov model 1 assumptions
        
        Parameters
        ----------
            prediction_grid : numpy.ndarray
                x,y coordinate numpy array of prediction grid, or grid cells that will be estimated
            df1 : pandas DataFrame
                data frame of primary conditioning data
            xx1 : string
                column name for x coordinates of input data frame for primary data
            yy1 : string 
                column name for y coordinates of input data frame for primary data
            zz1 : string
                column for z values (or data variable) of input data frame for primary data
            df2 : pandas DataFrame
                data frame of secondary data
            xx2 : string
                column name for x coordinates of input data frame for secondary data
            yy2 : string
                column name for y coordinates of input data frame for secondary data
            zz2 : string
                column for z values (or data variable) of input data frame for secondary data
            num_points : int
                the number of conditioning points to search for
            vario : list
                list of variogram parameters [azimuth, nugget, major_range, minor_range, sill, vtype]
                azimuth, nugget, major_range, minor_range, and sill can be int or float type
                vtype is a string that can be either 'Exponential', 'Spherical', or 'Gaussian'
            radius : int, float
                search radius
            corrcoef : float
                correlation coefficient between primary and secondary data
            quiet : bool
                If False, a progress bar will be printed to the console.
               Default is False
        
        Returns
        -------
            est_cokrige : numpy.ndarray
                cokriging estimate for each point in coordinate grid
            var_cokrige : numpy.ndarray
                cokriging variances
        """
        
        # unpack variogram parameters for rotation matrix
        azimuth = vario[0]
        major_range = vario[2]
        minor_range = vario[3]
        rotation_matrix = make_rotation_matrix(azimuth, major_range, minor_range) 

        df1 = df1.rename(columns = {xx1: "X", yy1: "Y", zz1: "Z"})
        df2 = df2.rename(columns = {xx2: "X", yy2: "Y", zz2: "Z"})

        mean_1 = np.average(df1['Z']) 
        var_1 = np.var(df1['Z']) # replaced var_1 = vario[4]
        vario[4] = np.var(df1['Z']) 
        mean_2 = np.average(df2['Z']) 
        var_2 = np.var(df2['Z'])

        est_cokrige = np.zeros(shape=len(prediction_grid)) 
        var_cokrige = np.zeros(shape=len(prediction_grid))

        # build the iterator
        if not quiet:
            _iterator = enumerate(tqdm(prediction_grid, position=0, leave=True))
        else:
            _iterator = enumerate(prediction_grid)

        for z, predxy in _iterator:
            test_idx = np.sum(prediction_grid[z]==df1[['X', 'Y']].values,axis = 1)
            if np.sum(test_idx==2)==0: #
                
                # get nearest neighbors
                nearest = NearestNeighbor.nearest_neighbor_search(radius, num_points, 
                                                                  prediction_grid[z], 
                                                                  df1[['X','Y','Z']])           
                nearest_second = NearestNeighbor.nearest_neighbor_secondary(prediction_grid[z], 
                                                                            df2[['X','Y','Z']]) 
                norm_data_val = nearest[:,-1] 
                norm_data_val = np.append(norm_data_val, [nearest_second[-1]]) 
                xy_val = nearest[:, :-1] 
                xy_second = nearest_second[:-1] 
                xy_val = np.append(xy_val, [xy_second], axis = 0) 
                new_num_pts = len(nearest)

                # covariance between data points
                covariance_matrix = np.zeros(shape=((new_num_pts + 1, new_num_pts + 1)))
                covariance_matrix[0:new_num_pts+1, 0:new_num_pts+1] = Covariance.make_covariance_matrix(xy_val, 
                                                                                                        vario, rotation_matrix) 

                # covariance between data and unknown
                covariance_array = np.zeros(shape=(new_num_pts + 1)) 
                k_weights = np.zeros(shape=(new_num_pts + 1))
                covariance_array[0:new_num_pts+1] = Covariance.make_covariance_array(xy_val, 
                                                                                     np.tile(prediction_grid[z], 
                                                                                             new_num_pts + 1), 
                                                                                     vario, rotation_matrix)
                covariance_array[new_num_pts] = covariance_array[new_num_pts] * corrcoef 

                # update covariance matrix with secondary info (gamma2 = rho12 * gamma1)
                covariance_matrix[new_num_pts, 0 : new_num_pts+1] = covariance_matrix[new_num_pts, 0 : new_num_pts+1] * corrcoef
                covariance_matrix[0 : new_num_pts+1, new_num_pts] = covariance_matrix[0 : new_num_pts+1, new_num_pts] * corrcoef
                covariance_matrix[new_num_pts, new_num_pts] = 1
                covariance_matrix.reshape(((new_num_pts + 1)), ((new_num_pts + 1)))

                # solve kriging system
                k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix, covariance_array, rcond = None) 
                part1 = mean_1 + np.sum(k_weights[0:new_num_pts]*(norm_data_val[0:new_num_pts] - mean_1)/np.sqrt(var_1))
                part2 = k_weights[new_num_pts] * (nearest_second[-1] - mean_2)/np.sqrt(var_2)
                               
                est_cokrige[z] = part1 + part2 
                var_cokrige[z] = var_1 - np.sum(k_weights*covariance_array) 
            else:
                est_cokrige[z] = df1['Z'].values[np.where(test_idx==2)[0][0]]
                var_cokrige[z] = 0

        return est_cokrige, var_cokrige

    def cosim_mm1(prediction_grid, df1, xx1, yy1, zz1, df2, xx2, yy2, zz2, num_points, vario, radius, corrcoef, quiet=False):
        """
        Cosimulation under Markov model 1 assumptions
        
        Parameters
        ----------
            prediction_grid : numpy.ndarray
                x,y coordinate numpy array of prediction grid, or grid cells that will be estimated
            df1 : pandas DataFrame
                data frame of primary conditioning data
            xx1 : string
                column name for x coordinates of input data frame for primary data
            yy1 : string 
                column name for y coordinates of input data frame for primary data
            zz1 : string
                column for z values (or data variable) of input data frame for primary data
            df2 : pandas DataFrame
                data frame of secondary data
            xx2 : string
                column name for x coordinates of input data frame for secondary data
            yy2 : string
                column name for y coordinates of input data frame for secondary data
            zz2 : string
                column for z values (or data variable) of input data frame for secondary data
            num_points : int
                the number of conditioning points to search for
            vario : list
                list of variogram parameters [azimuth, nugget, major_range, minor_range, sill, vtype]
                azimuth, nugget, major_range, minor_range, and sill can be int or float type
                vtype is a string that can be either 'Exponential', 'Spherical', or 'Gaussian'
            radius : int, float
                search radius
            corrcoef : float
                correlation coefficient between primary and secondary data
            quiet : bool
                If False, a progress bar will be printed to the console.
               Default is False

        Returns
        -------
            cosim : numpy.ndarray
                cosimulation for each point in coordinate grid
        """
            
        # unpack variogram parameters
        azimuth = vario[0]
        major_range = vario[2]
        minor_range = vario[3]
        rotation_matrix = make_rotation_matrix(azimuth, major_range, minor_range)
        df1 = df1.rename(columns = {xx1: "X", yy1: "Y", zz1: "Z"}) 
        df2 = df2.rename(columns = {xx2: "X", yy2: "Y", zz2: "Z"})
        xyindex = np.arange(len(prediction_grid)) 
        random.shuffle(xyindex)

        mean_1 = np.average(df1['Z']) 
        var_1 = np.var(df1['Z']) # replaced var_1 = vario[4]
        vario[4] = np.var(df1['Z']) 
        mean_2 = np.average(df2['Z']) 
        var_2 = np.var(df2['Z'])
   
        cosim = np.zeros(shape=len(prediction_grid))

        # build the iterator
        if not quiet:
            _iterator = enumerate(tqdm(prediction_grid, position=0, leave=True))
        else:
            _iterator = enumerate(prediction_grid)

        # for each coordinate in the prediction grid
        for idx, predxy in _iterator:
            z = xyindex[idx]
            test_idx = np.sum(prediction_grid[z]==df1[['X', 'Y']].values,axis = 1)
            if np.sum(test_idx==2)==0:
                
                # get nearest neighbors
                nearest = NearestNeighbor.nearest_neighbor_search(radius, num_points, 
                                                                  prediction_grid[z], 
                                                                  df1[['X','Y','Z']]) 
                nearest_second = NearestNeighbor.nearest_neighbor_secondary(prediction_grid[z], 
                                                                            df2[['X','Y','Z']])
                norm_data_val = nearest[:,-1]
                norm_data_val = np.append(norm_data_val, [nearest_second[-1]]) 
                xy_val = nearest[:, :-1] 
                xy_second = nearest_second[:-1]
                xy_val = np.append(xy_val, [xy_second], axis = 0) 
                new_num_pts = len(nearest)

                # covariance between data poitns
                covariance_matrix = np.zeros(shape=((new_num_pts + 1, new_num_pts + 1))) 
                covariance_matrix[0:new_num_pts+1, 0:new_num_pts+1] = Covariance.make_covariance_matrix(xy_val, 
                                                                                                        vario, rotation_matrix) 

                # covariance between data and unknown
                covariance_array = np.zeros(shape=(new_num_pts + 1))
                k_weights = np.zeros(shape=(new_num_pts + 1))
                covariance_array[0:new_num_pts+1] = Covariance.make_covariance_array(xy_val, 
                                                                                     np.tile(prediction_grid[z], 
                                                                                             new_num_pts + 1), 
                                                                                     vario, rotation_matrix)
                covariance_array[new_num_pts] = covariance_array[new_num_pts] * corrcoef 

                # update covariance matrix with secondary info (gamma2 = rho12 * gamma1)
                covariance_matrix[new_num_pts, 0 : new_num_pts+1] = covariance_matrix[new_num_pts, 0 : new_num_pts+1] * corrcoef
                covariance_matrix[0 : new_num_pts+1, new_num_pts] = covariance_matrix[0 : new_num_pts+1, new_num_pts] * corrcoef
                covariance_matrix[new_num_pts, new_num_pts] = 1
                covariance_matrix.reshape(((new_num_pts + 1)), ((new_num_pts + 1)))

                # solve kriging system
                k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix, covariance_array, rcond = None) 
                part1 = mean_1 + np.sum(k_weights[0:new_num_pts]*(norm_data_val[0:new_num_pts] - mean_1)/np.sqrt(var_1))
                part2 = k_weights[new_num_pts] * (nearest_second[-1] - mean_2)/np.sqrt(var_2)
                est_cokrige = part1 + part2 
                var_cokrige = var_1 - np.sum(k_weights*covariance_array)
                var_cokrige = np.absolute(var_cokrige) 

                cosim[z] = np.random.normal(est_cokrige,math.sqrt(var_cokrige),1) 
            else:
                cosim[z] = df1['Z'].values[np.where(test_idx==2)[0][0]]

            coords = prediction_grid[z:z+1,:]
            df1 = pd.concat([df1,pd.DataFrame({'X': [coords[0,0]], 'Y': [coords[0,1]], 'Z': [cosim[z]]})], sort=False) 

        return cosim

__all__ = ['Gridding', 'NearestNeighbor', 'Covariance', 'Interpolation', 'rbf_trend', 
    'adaptive_partitioning', 'make_rotation_matrix']

def __dir__():
    return __all__

def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f'module {__name__} has no attribute {name}')
    return globals()[name]