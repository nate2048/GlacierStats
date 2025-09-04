import xarray as xr
import pandas as pd
import sys
import numpy as np
import torch
import time

sys.path.append("../")
import torch_vectorized

if __name__ == "__main__":
    
    
    ######################################## PREPARE DATA ########################################
    bedmap = xr.open_dataset('Bedmap_Antarctica/bedmap3_mod_1000.nc')
    vario_params = xr.open_dataset('Bedmap_Antarctica/continental_variogram_1000.nc')
    
    thick_cond = np.where(bedmap.mask.values == 4, 0, bedmap.thick_cond.values)
    # elevation values
    bed_cond = bedmap.surface_topography.values - thick_cond
    # Masks out continent
    ice_rock_msk = (bedmap.mask.values == 1) | (bedmap.mask.values == 4) | (bedmap.mask.values == 2)
    # mask out irrelevatant elevation values
    bed_cond = np.where(ice_rock_msk, bed_cond, np.nan)
    xx, yy = np.meshgrid(bedmap.x, bedmap.y)

    # Mask out conditioning data
    cond_msk = ~np.isnan(bed_cond)
    x_cond = torch.from_numpy(xx[cond_msk])
    y_cond = torch.from_numpy(yy[cond_msk])
    data_cond = torch.from_numpy(bed_cond[cond_msk] - bedmap.trend.values[cond_msk])

    # Mask out simulation coordinates
    sim_mask = ~cond_msk * ice_rock_msk
    x_sim = torch.from_numpy(xx[sim_mask])
    y_sim = torch.from_numpy(yy[sim_mask])
    # Mask out simulation variogram params
    azimuth_arr = torch.from_numpy(vario_params.azimuth.values[sim_mask])
    nugget_arr = torch.zeros(azimuth_arr.shape)
    major_range_arr = torch.from_numpy(vario_params.major_range.values[sim_mask])
    minor_range_arr = torch.from_numpy(vario_params.minor_range.values[sim_mask])
    sill_arr = torch.from_numpy(vario_params.sill.values[sim_mask])
    smooth_arr = torch.from_numpy(vario_params.smooth.values[sim_mask])

    # Make torch tensors
    xy_cond = torch.cat((x_cond.unsqueeze(1), x_cond.unsqueeze(1)), dim=1)
    xy_sim = torch.cat((x_sim.unsqueeze(1), y_sim.unsqueeze(1)), dim=1)
    vario_sim = torch.cat((azimuth_arr.unsqueeze(1), nugget_arr.unsqueeze(1), major_range_arr.unsqueeze(1), 
                           minor_range_arr.unsqueeze(1), sill_arr.unsqueeze(1), smooth_arr.unsqueeze(1)), dim=1)
    
    k = 24          # number of neighboring data points used to estimate a given point
    rad = 50000     # 50 km search radius
    
    num_gpus = torch.cuda.device_count()
    multiplier = 800_000 # This is to avoid CUDA OUT OF MEM ERROR
        
    start_time = time.time()
        
    sgs = torch_vectorized.skrige_sgs(xy_cond, data_cond, xy_sim, vario_sim, k, rad, num_gpus, multiplier)

    end_time = time.time()
    print(f"Total time to complete: {end_time-start_time}s")
    
    torch.save(sgs, "vectorized_sgs.pt")

