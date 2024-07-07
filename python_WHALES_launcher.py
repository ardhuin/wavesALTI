# -*- coding: utf-8 -*-
"""
Created on November 2018

The code launches the WHALES retracker using original mission files

Works for the following missions: Jason, Jason-2, Jason-3 ... 

@author: Marcello Passaro
"""
import argparse
# import cmath
import netCDF4
from netCDF4 import Dataset
import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import scipy.io
import os
import time
from compute_instr_corr_SWH_WHALES import compute_instr_corr_SWH_WHALES
# import sys
# from read_functions import wf_reader

from Retracker_MP import *

from WHALES_withRangeAndEpoch import *

from scipy.io import matlab
# import pandas as pd

from import_weights_mat import import_weights_mat


def get_options():
    parser = argparse.ArgumentParser(
        description='Retrack a SGDR file with WHALES')

    parser.add_argument(
        '-m', '--mission', type=str,
        choices=['envisat', 'jason1', 'jason2', 'jason3', 'saral', 'cs2_lrm',
                 'jason3f', 'cfosat'],
        help='satellite mission'
    )
    parser.add_argument(
        '-i', '--input', type=str,
        help='path to the SGDR file'
    )
    parser.add_argument(
        '-o', '--output', type=str, default='.',
        help='path to the output repository'
    )
    return parser.parse_args()


options = get_options()
filename = options.input
mission = options.mission
saving_directory = options.output

saving_name = os.path.join(saving_directory, os.path.basename(filename))

cal2 = 'on'

add_instr_corr_SWH = 'no'
import_weights = 'yes'



# Mission-dependent parameters and files to be loaded
if mission in ['envisat']:
    my_path_instr_corr_SWH = ''
    my_path_weights = 'weights/weights_n1.mat'
    tau=3.125 #gate spacing in ns
    Theta=1.35 *np.pi/180 #modified http://www.aviso.oceanobs.com/fileadmin/documents/OSTST/2010/oral/PThibaut_Jason2.pdf  % The antenna 3dB bandwidth (degrees transformed in radians)
    SigmaP=0.53*tau #from Gomez Enri 2006. Otherwise use:%1.6562; %ns =0.53*3.125ns
elif mission in ['jason1']:
    my_path_instr_corr_SWH = 'instr_corr/SWHinstrcorr_MLE4_jason1SGDRc.mat'
    my_path_weights = 'weights/weights.mat'
    tau=3.125
    Theta=1.29 *np.pi/180
    SigmaP=0.513*tau
elif mission in ['jason2']:
    my_path_instr_corr_SWH = 'instr_corr/SWHinstrcorr_WHALES_jason2SGDRd.mat'
    my_path_weights = 'weights/weights.mat'
    tau=3.125
    Theta=1.29 *np.pi/180
    SigmaP=0.513*tau
elif mission in ['jason3', 'jason3f']:
    my_path_instr_corr_SWH = 'instr_corr/SWHinstrcorr_WHALES_jason3SGDRd.mat'
    my_path_weights = 'weights/weights.mat'
    tau=3.125
    Theta=1.29 *np.pi/180
    SigmaP=0.513*tau
elif mission.lower() in ['altika', 'saral', 'saral_igdr']:
    my_path_instr_corr_SWH = ''
    my_path_weights = 'weights/weights_alt.mat'
    tau=3.125*320/480
    Theta=0.605 *np.pi/180
    SigmaP=0.513*tau
elif mission in ['cs2_lrm']:
    my_path_instr_corr_SWH = ''
    my_path_weights = 'weights/weights_cs2_lrm.mat'
    tau=3.125 #gate spacing in ns
    Theta=1.1992 *np.pi/180 #modified http://www.aviso.oceanobs.com/fileadmin/documents/OSTST/2010/oral/PThibaut_Jason2.pdf  % The antenna 3dB bandwidth (degrees transformed in radians)
    SigmaP=0.513*tau     

#        if mission.lower() == 'ers2_r' or mission.lower() == 'ers2_r_2cm':           
#            tau=3.03
#            Theta=1.3 *np.pi/180
#            SigmaP=0.513*tau    

if my_path_instr_corr_SWH != '':
    my_path_instr_corr_SWH = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), my_path_instr_corr_SWH)
if my_path_weights != '':
    my_path_weights = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), my_path_weights)

if import_weights == 'yes':
    import h5py
    mat_weights = h5py.File(my_path_weights,'r')
    residual_std=np.transpose(mat_weights['residual_tot'])
    flag_edges=np.transpose(mat_weights['flag_edges'])

    # residual_std,flag_edges=import_weights_mat(my_path_weights)


# 2) FUNCTION DEFINITIONS

def moving_average(a, n):
    a = np.concatenate((np.zeros((n + 1,)), a, np.zeros((n,))));
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def wf_reader(filename):
    S = netCDF4.Dataset(filename, 'r')

    return S


####### ---------------------------------------------------------------######


# 3) Launcher
counting_swh = 0  # Only needed to check which is the first waveform to be reprocessed, in order to use the swh_instr_corr interpolation only once

S = wf_reader(filename)

if mission in ['jason1', 'jason2', 'jason3']:
    # Getting 20 Hz data
    S_time = np.ma.getdata(S.variables['time_20hz'][:])
    S_height = np.ma.getdata(S.variables['alt_20hz'][:])
    S_swh = np.ma.getdata(S.variables['swh_20hz_ku'][:])
    S_tracker = np.ma.getdata(S.variables['tracker_20hz_ku'][:])
    S_range = np.ma.getdata(S.variables['range_20hz_ku'][:])
    S_waveform = np.ma.getdata(S.variables['waveforms_20hz_ku'][:])
    S_lat = np.ma.getdata(S.variables['lat_20hz'][:])
    S_lon = np.ma.getdata(S.variables['lon_20hz'][:])
    S_landmask = np.ma.getdata(S.variables['surface_type'][:])
    S_offnadir = np.ma.getdata(S.variables['off_nadir_angle_wf_20hz_ku'][:])
    S_atmos_corr = np.ma.getdata(S.variables['atmos_corr_sig0_ku'][:])
    # This field is at 1-Hz, so it has to be reshaped
    S_atmos_corr = np.transpose(np.tile(S_atmos_corr, (np.shape(S_time)[1], 1)))

    S_scaling_factor = np.ma.getdata(S.variables['scaling_factor_20hz_ku'][:])

elif mission in ['jason3f']:
    # AVISO SGDR version F
    S_time = np.ma.getdata(S['data_20'].variables['time'][:])
    S_time = np.reshape(S_time, (np.shape(S_time)[0], 1))

    S_height = np.ma.getdata(S['data_20'].variables['altitude'][:])
    S_height = np.reshape(S_height, (np.shape(S_time)[0], 1))

    S_swh = np.ma.getdata(S['data_20']['ku'].variables['swh_ocean'][:])
    S_swh = np.reshape(S_swh, (np.shape(S_time)[0], 1))

    S_tracker = np.ma.getdata(
        S['data_20']['ku'].variables['tracker_range_calibrated'][:])
    S_tracker = np.reshape(S_tracker, (np.shape(S_time)[0], 1))

    S_range = np.ma.getdata(S['data_20']['ku'].variables['range_ocean'][:])
    S_range = np.reshape(S_range, (np.shape(S_time)[0], 1))

    S_waveform = np.ma.getdata(
        S['data_20']['ku'].variables['power_waveform'][:])
    S_waveform = np.reshape(
        S_waveform, (np.shape(S_time)[0], 1, np.shape(S_waveform)[1]))
    print(S_waveform.shape)

    S_lat = np.ma.getdata(S['data_20'].variables['latitude'][:])
    S_lat = np.reshape(S_lat, (np.shape(S_time)[0], 1))

    S_lon = np.ma.getdata(S['data_20'].variables['longitude'][:])
    S_lon = np.reshape(S_lon, (np.shape(S_time)[0], 1))

    S_landmask = np.ma.getdata(
        S['data_20'].variables['surface_classification_flag'][:])
    S_landmask = np.reshape(S_landmask, (np.shape(S_time)[0], 1))

    S_offnadir = np.ma.getdata(
        S['data_20']['ku'].variables['off_nadir_angle_wf_ocean'][:])
    S_offnadir = np.reshape(S_offnadir, (np.shape(S_time)[0], 1))

    atmos_corr = np.ma.getdata(
        S['data_01']['ku'].variables['sig0_cor_atm'][:])
    # This field is at 1-Hz, so it has to be reshaped
    S_atmos_corr = atmos_corr[S['data_20'].variables['index_1hz_measurement']]
    S_atmos_corr = np.reshape(S_atmos_corr, (np.shape(S_time)[0], 1))

    S_scaling_factor = np.ma.getdata(
        S['data_20']['ku'].variables['sig0_scaling_factor'][:])
    S_scaling_factor = np.reshape(S_scaling_factor, (np.shape(S_time)[0], 1))

    mission = 'jason3'

elif mission in ['envisat']:

    S_time = np.ma.getdata(S.variables['time_20'][:])
    print(np.shape(S_time))
    S_time = np.reshape(S_time, (np.shape(S_time)[0], 1))
    print(np.shape(S_time))

    # Time at 1-Hz for interpolation of fields available only at 1-Hz
    S_time_1hz = np.ma.getdata(S.variables['time_01'][:])
    S_time_1hz = np.reshape(S_time_1hz, (np.shape(S_time_1hz)[0], 1))

    S_height = np.ma.getdata(S.variables['alt_20'][:])
    S_height = np.reshape(S_height, (np.shape(S_time)[0], 1))

    S_swh = np.ma.getdata(S.variables['swh_ocean_20_ku'][:])
    S_swh = np.reshape(S_swh, (np.shape(S_time)[0], 1))
    print(np.shape(S_height))

    S_tracker = np.ma.getdata(S.variables['tracker_range_20_ku'][:])
    S_tracker = np.reshape(S_tracker, (np.shape(S_time)[0], 1))

    S_range = np.ma.getdata(S.variables['range_ocean_20_ku'][:])
    S_range = np.reshape(S_range, (np.shape(S_time)[0], 1))

    S_waveform = np.ma.getdata(S.variables['waveform_fft_20_ku'][:])
    # S_waveform=np.reshape(S_waveform,(np.shape(S_time)[0],1) )

    S_lat = np.ma.getdata(S.variables['lat_20'][:])
    S_lat = np.reshape(S_lat, (np.shape(S_time)[0], 1))

    S_lon = np.ma.getdata(S.variables['lon_20'][:])
    S_lon = np.reshape(S_lon, (np.shape(S_time)[0], 1))

    # S_landmask=np.ma.getdata( S.variables['surf_type_20'][:] )
    # S_landmask=np.reshape(S_landmask,(np.shape(S_time)[0],1) )

    # OFF NADIR ANGLE FROM WAVEFORM
    # S_offnadir=np.ma.getdata( S.variables['off_nadir_angle_wf_ocean_20_ku'][:] )     #degrees^2
    # S_offnadir=np.reshape(S_offnadir,(np.shape(S_time)[0],1) )
    # S_offnadir=S_time*0.0

    # OFF NADIR ANGLE FROM PLATFORM
    S_offnadir = np.ma.getdata(S.variables['off_nadir_angle_pf_01'][:])
    S_offnadir = np.reshape(S_offnadir, (np.shape(S_time_1hz)[0], 1))
    S_offnadir = np.interp(S_time[:, 0], S_time_1hz[:, 0], S_offnadir[:, 0])
    S_offnadir = np.reshape(S_offnadir, (np.shape(S_time)[0], 1))

    S_atmos_corr = np.ma.getdata(S.variables['atm_cor_sig0_01_ku'][:])
    S_atmos_corr = np.reshape(S_atmos_corr, (np.shape(S_time_1hz)[0], 1))
    # This field is at 1-Hz, so it has to be reshaped
    S_atmos_corr = np.interp(S_time[:, 0], S_time_1hz[:, 0], S_atmos_corr[:, 0])
    S_atmos_corr = np.reshape(S_atmos_corr, (np.shape(S_time)[0], 1))

    S_scaling_factor = np.ma.getdata(S.variables['scale_factor_20_ku'][:])
    S_scaling_factor = np.reshape(S_scaling_factor, (np.shape(S_time)[0], 1))

elif mission in ['saral', 'altika']:
    S_time = np.ma.getdata(S.variables['time_40hz'][:])
    S_height = np.ma.getdata(S.variables['alt_40hz'][:])
    S_swh = np.ma.getdata(S.variables['swh_40hz'][:])
    S_tracker = np.ma.getdata(S.variables['tracker_40hz'][:])
    S_range = np.ma.getdata(S.variables['range_40hz'][:])
    S_waveform = np.ma.getdata(S.variables['waveforms_40hz'][:])
    S_lat = np.ma.getdata(S.variables['lat_40hz'][:])
    S_lon = np.ma.getdata(S.variables['lon_40hz'][:])
    S_landmask = np.ma.getdata(S.variables['surface_type'][:])

    # OFF NADIR ANGLE FROM WAVEFORM
    # S_offnadir=np.ma.getdata( S.variables['off_nadir_angle_wf_40hz'][:] )
    ##Unrealistic offnadir angles of value higher than 0.3 degrees, which would affect Range estimation (Dorandeau et al. 2004) are removed
    ##Note that in Altika there are some values put as 3267, where the retracking of the offnadir likely failed
    # index_offnadir=np.where(np.abs(S_offnadir)>0.3)[0]
    # S_offnadir[index_offnadir]=0.
    ##Off nadir angle field is filtered with an alongtrack filter of 30 seconds as suggested by Amarouche et al. (2004)

    # OFF NADIR ANGLE FROM PLATFORM
    S_offnadir = np.ma.getdata(S.variables['off_nadir_angle_pf'][:])
    # This field is at 1-Hz, so it has to be reshaped
    S_offnadir = np.transpose(np.tile(S_offnadir, (np.shape(S_time)[1], 1)))

    S_atmos_corr = np.ma.getdata(S.variables['atmos_corr_sig0'][:])
    # This field is at 1-Hz, so it has to be reshaped
    S_atmos_corr = np.transpose(np.tile(S_atmos_corr, (np.shape(S_time)[1], 1)))

    S_scaling_factor = np.ma.getdata(S.variables['scaling_factor_40hz'][:])

elif mission in ['cs2_lrm']:

    S_time = np.ma.getdata(S.variables['time_20_ku'][:])
    print(np.shape(S_time))
    S_time = np.reshape(S_time, (np.shape(S_time)[0], 1))
    print(np.shape(S_time))

    # Time at 1-Hz for interpolation of fields available only at 1-Hz
    S_time_1hz = np.ma.getdata(S.variables['time_avg_01_ku'][:])
    S_time_1hz = np.reshape(S_time_1hz, (np.shape(S_time_1hz)[0], 1))

    S_height = np.ma.getdata(S.variables['alt_20_ku'][:])
    S_height = np.reshape(S_height, (np.shape(S_time)[0], 1))

    # S_swh=np.ma.getdata( S.variables['swh_ocean_20_ku'][:] )
    # S_swh=np.reshape(S_swh,(np.shape(S_time)[0],1) )
    # print(np.shape(S_height))

    S_tracker = np.ma.getdata(
        S.variables['window_del_20_ku'][:] * (299792458.0) / 2.0)
    S_tracker = np.reshape(S_tracker, (np.shape(S_time)[0], 1))

    # S_range=np.ma.getdata( S.variables['range_ocean_20_ku'][:] )
    # S_range=np.reshape(S_range,(np.shape(S_time)[0],1) )

    S_waveform = np.ma.getdata(S.variables['pwr_waveform_20_ku'][:])
    # S_waveform=np.reshape(S_waveform,(np.shape(S_time)[0],1) )

    S_lat = np.ma.getdata(S.variables['lat_20_ku'][:])
    S_lat = np.reshape(S_lat, (np.shape(S_time)[0], 1))

    S_lon = np.ma.getdata(S.variables['lon_20_ku'][:])
    S_lon = np.reshape(S_lon, (np.shape(S_time)[0], 1))

    # S_landmask=np.ma.getdata( S.variables['surf_type_20'][:] )
    # S_landmask=np.reshape(S_landmask,(np.shape(S_time)[0],1) )

    # S_offnadir=np.ma.getdata( S.variables['off_nadir_angle_wf_ocean_20_ku'][:] )     #degrees^2
    # S_offnadir=np.reshape(S_offnadir,(np.shape(S_time)[0],1) )

    # OFF NADIR ANGLE FROM PLATFORM
    S_offnadir = np.ma.getdata(
        S.variables['off_nadir_pitch_angle_str_20_ku'][:])
    # S_offnadir=np.reshape(S_offnadir,(np.shape(S_time_1hz)[0],1) )
    # S_offnadir=np.interp(S_time[:,0],S_time_1hz[:,0],S_offnadir[:,0])
    S_offnadir = np.reshape(S_offnadir, (np.shape(S_time)[0], 1))

    # S_atmos_corr=np.ma.getdata( S.variables['atm_cor_sig0_01_ku'][:] )
    # S_atmos_corr=np.reshape(S_atmos_corr,(np.shape(S_time_1hz)[0],1) )
    ##This field is at 1-Hz, so it has to be reshaped
    # S_atmos_corr=np.interp(S_time[:,0],S_time_1hz[:,0],S_atmos_corr[:,0])
    # S_atmos_corr=np.reshape(S_atmos_corr,(np.shape(S_time)[0],1) )

    # S_scaling_factor=np.ma.getdata( S.variables['scale_factor_20_ku'][:] )
    # S_scaling_factor=np.reshape(S_scaling_factor,(np.shape(S_time)[0],1) )

# WHALES RETRACKING ATTEMPT
landmask = np.empty(np.shape(S_time)) * np.nan

swh_WHALES = np.empty(np.shape(S_time)) * np.nan

Err_WHALES = np.empty(np.shape(S_time)) * np.nan
Epoch_WHALES = np.empty(np.shape(S_time)) * np.nan
Amplitude_WHALES = np.empty(np.shape(S_time)) * np.nan

sigma0_WHALES = np.empty(np.shape(S_time)) * np.nan

time_20hz = np.empty(np.shape(S_time)) * np.nan

altitude = np.empty(np.shape(S_time)) * np.nan
range_WHALES = np.empty(np.shape(S_time)) * np.nan

swh_WHALES_instr_corr = np.empty(np.shape(S_time)) * np.nan

#
# Now looping over waveforms for retracking
# First loop is on 1 Hz data, second loop is on higher rate data 
#
for index_waveforms_row in np.arange(0, np.shape(S_time)[0], 1):
    print("Retracking waveform  " + str(index_waveforms_row) + "  of  " +
              str(np.size(S_time)))
    for index_waveforms_col in np.arange(0, np.shape(S_time)[1], 1):
        
 
        input = {}
        if cal2 == 'on':
            if mission == 'jason3':
# Application of CAL-2 where known
# FA: I would be happy to have more details on this "cal-2" 
            #J1_filter = np.loadtxt('J1_MeanFilterKu')
            #J1_filter_norm = J1_filter / np.mean(J1_filter[11:115])
                J3_filter = np.loadtxt('J3_MeanFilterKu')
                J3_filter_norm = J3_filter / np.mean(J3_filter[11:115])
                input['waveform'] = S_waveform[
                    index_waveforms_row, index_waveforms_col, :] / \
                                    J3_filter_norm[11:115]
            elif mission == 'jason2':
                J2_filter = np.loadtxt('J2_MeanFilterKu')
                J2_filter_norm = J2_filter / np.mean(J2_filter[11:115])
                input['waveform'] = S_waveform[index_waveforms_row,
                                    index_waveforms_col, :] / J2_filter_norm[
                                                              11:115]
            elif mission == 'saral':
                saral_filter = np.loadtxt('ALK_MeanFilter')
                saral_filter_norm = saral_filter / np.mean(saral_filter)
                input['waveform'] = S_waveform[index_waveforms_row,
                                    index_waveforms_col, :] / saral_filter_norm
            elif mission == 'envisat':
                input['waveform'] = S_waveform[index_waveforms_row, :]
            elif mission == 'cs2_lrm':
                input['waveform'] = S_waveform[index_waveforms_row, :]
        else:
            input['waveform'] = S_waveform[index_waveforms_row,
                                index_waveforms_col, :]

        ' raw range in [m] '
        input['uralt'] = S_tracker[index_waveforms_row, index_waveforms_col]

        ' hsat '
        input['hsat'] = S_height[index_waveforms_row, index_waveforms_col]
        ' mission '
        #if mission == 'jason3':
        #    input['mission'] = 'jason3'
        input['mission'] = mission

        ' off nadir angle in degree '
        input['xi'] = S_offnadir[index_waveforms_row, index_waveforms_col]
        if import_weights == 'yes':
            input['weights_flag'] = flag_edges
            input['weights'] = residual_std

        input['tau'] = tau
        input['Theta']  = Theta 
        input['SigmaP']  = SigmaP
#
# Calls retracker 
#
        retracker = WHALES_withRangeAndEpoch(input)
#
# Post-processing
#
        # Quality flag of WHALES, based on the normalised fitting error on the leading edge
        if (retracker.Error) > 0.3 and (np.isnan(retracker.Error) == 0):
            Err_WHALES[index_waveforms_row, index_waveforms_col] = 1
        elif retracker.Error <= 0.3:
            Err_WHALES[index_waveforms_row, index_waveforms_col] = 0

        swh_WHALES[index_waveforms_row, index_waveforms_col] = retracker.SWH
        
        Epoch_WHALES[index_waveforms_row, index_waveforms_col] = retracker.Epoch

        if mission in ['envisat', 'envisat_over']:
            sigma0_WHALES[
                index_waveforms_row, index_waveforms_col] = retracker.Amplitude + \
                                                            S_atmos_corr[
                                                                index_waveforms_row, index_waveforms_col] + \
                                                            S_scaling_factor[
                                                                index_waveforms_row, index_waveforms_col] - 33.1133
        elif mission in ['jason2', 'jason1', 'saral', 'saral_igdr', 'jason3']:
            sigma0_WHALES[
                index_waveforms_row, index_waveforms_col] = retracker.Amplitude + \
                                                            S_atmos_corr[
                                                                index_waveforms_row, index_waveforms_col] + \
                                                            S_scaling_factor[
                                                                index_waveforms_row, index_waveforms_col]
        elif mission in ['cs2_lrm']:
            sigma0_WHALES[
                index_waveforms_row, index_waveforms_col] = retracker.Amplitude

        range_WHALES[index_waveforms_row, index_waveforms_col] = retracker.range

        Amplitude_WHALES[
            index_waveforms_row, index_waveforms_col] = retracker.Norm_Amplitude

        # APPLICATION OF INSTRUMENTAL CORRECTION FOR SWH           
        if add_instr_corr_SWH == 'yes':
            if counting_swh == 0:
                swh_WHALES_instr_corr[
                    index_waveforms_row, index_waveforms_col], interpolator_instr_corr_SWH = compute_instr_corr_SWH_WHALES(
                    swh_WHALES[index_waveforms_row, index_waveforms_col],
                    my_path_instr_corr_SWH, mission)
                counting_swh = 1
            else:
                swh_WHALES_instr_corr[
                    index_waveforms_row, index_waveforms_col], interpolator_instr_corr_SWH = compute_instr_corr_SWH_WHALES(
                    swh_WHALES[index_waveforms_row, index_waveforms_col],
                    my_path_instr_corr_SWH, mission,
                    interpolator_instr_corr_SWH)
    print('Hs:',np.mean(swh_WHALES[index_waveforms_row,:]),' std:',np.std(swh_WHALES[index_waveforms_row,:]))


#
# End of loop: writes all output to NetCDF file
# 
w_nc_fid = Dataset(saving_name, 'w',
                   format='NETCDF3_CLASSIC')
w_nc_fid.createDimension('time', np.shape(time_20hz)[0])
w_nc_fid.createDimension('records', np.shape(time_20hz)[1])

w_nc_var = w_nc_fid.createVariable('time_20hz', 'f8', ('time', 'records'),
                                   zlib=True)
w_nc_var.setncatts({'long_name': u"time_20hz", \
                    'units': u"s", \
                    'comment': u"time in seconds"})
w_nc_fid.variables['time_20hz'][:] = S_time

w_nc_var = w_nc_fid.createVariable('lat_20hz', 'f8', ('time', 'records'),
                                   zlib=True)
w_nc_var.setncatts({'long_name': u"lat_20hz", \
                    'units': u"deg", \
                    'comment': u" "})
w_nc_fid.variables['lat_20hz'][:] = S_lat

w_nc_var = w_nc_fid.createVariable('lon_20hz', 'f8', ('time', 'records'),
                                   zlib=True)
w_nc_var.setncatts({'long_name': u"lon_20hz", \
                    'units': u"deg", \
                    'comment': u" "})
w_nc_fid.variables['lon_20hz'][:] = S_lon

w_nc_var = w_nc_fid.createVariable('swh_WHALES_20hz', 'f8', ('time', 'records'),
                                   zlib=True)
w_nc_var.setncatts({'long_name': u"swh_WHALES_20hz", \
                    'units': u"m", \
                    'comment': u" "})
w_nc_fid.variables['swh_WHALES_20hz'][:] = swh_WHALES

w_nc_var = w_nc_fid.createVariable('swh_WHALES_instr_corr_20hz', 'f8',
                                   ('time', 'records'), zlib=True)
w_nc_var.setncatts({'long_name': u"swh_WHALES_instr_corr_20hz", \
                    'units': u"m", \
                    'comment': u" "})
w_nc_fid.variables['swh_WHALES_instr_corr_20hz'][:] = swh_WHALES_instr_corr

w_nc_var = w_nc_fid.createVariable('sigma0_WHALES_20hz', 'f8',
                                   ('time', 'records'), zlib=True)
w_nc_var.setncatts({'long_name': u"sigma0_WHALES_20hz", \
                    'units': u"dB", \
                    'comment': u" "})
w_nc_fid.variables['sigma0_WHALES_20hz'][:] = sigma0_WHALES

w_nc_var = w_nc_fid.createVariable('range_WHALES_20hz', 'f8',
                                   ('time', 'records'), zlib=True)
w_nc_var.setncatts({'long_name': u"range_WHALES_20hz", \
                    'units': u"m", \
                    'comment': u" "})
w_nc_fid.variables['range_WHALES_20hz'][:] = range_WHALES

w_nc_var = w_nc_fid.createVariable('epoch_WHALES_20hz', 'f8',
                                   ('time', 'records'), zlib=True)
w_nc_var.setncatts({'long_name': u"epoch_WHALES_20hz", \
                    'units': u"m", \
                    'comment': u" "})
w_nc_fid.variables['epoch_WHALES_20hz'][:] = Epoch_WHALES

w_nc_var = w_nc_fid.createVariable('swh_WHALES_qual_20hz', 'f8',
                                   ('time', 'records'), zlib=True)
w_nc_var.setncatts({'long_name': u"quality flag for Significant waveheight", \
                    'units': u"count", \
                    'comment': u"0=Good, 1=Bad"})
w_nc_fid.variables['swh_WHALES_qual_20hz'][:] = Err_WHALES


w_nc_fid.close()  # close the new file
