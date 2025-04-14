#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================
# === 0. Import Packages ===========================================================
# ==================================================================================
# from obspy import read, read_inventory,UTCDateTime
# -- to work with basic stuff
import glob
import os
import sys
import time
# to deal with .pickle
import pickle
import pandas as pd
# to interpolate values
import scipy.interpolate as spi
import scipy.integrate as spint
from scipy.signal import fftconvolve
# to work with mathematical and numerical stuff
import numpy as np
# -- to work with dates
import datetime as dt
from datetime import datetime
# -- to work with argument parsing
import argparse

# XARRAY
import xarray as xr
# import xrft

import warnings
warnings.filterwarnings("ignore")

# -- to create plots
import matplotlib as mpl
import matplotlib.pyplot as plt
# to work with dates on plots
import matplotlib.dates as mdates
# to work with text on plots
import matplotlib.text as mtext
# to work with colors / colorbar
import matplotlib.colors as mcolors
import matplotlib.cm as cmx

mpl.rcParams.update({'font.size': 14,'savefig.facecolor':'white'})

twopi = 2* np.pi

from wave_physics_functions import *

def decode_two_columns_time(coord,name_dim):
	factor = xr.DataArray([1e6, 1], dims=name_dim)
	time_us = (coord * factor).sum(dim=name_dim).assign_attrs(units="microseconds since 2009-01-01 00:00:00 0:00")
	#n_tim since 2009-01-01 00:00:00 0:00
	return time_us	

def get_datetime_fromDOY(YYYY,DOY):
	return pd.to_datetime(dt.datetime(YYYY,1,1))+pd.Timedelta(days=DOY-1)

# ==================================================================================
# === 1. Functions to get lists of files (L2,L2S,L2P) ==============================
# ==================================================================================
##### From CNES L2 file, get associated ODL L2S file (ribbons)
def file_L2S_from_file_L2(filetrack,PATH_ODL,v,nbeam,islocal=0):
	if islocal:
		indstart = 6
	else:
		indstart = 12

	YYYY = filetrack.split('/')[indstart]
	DDD = filetrack.split('/')[indstart+1]
	filenc = filetrack.split('/')[-1]
	filefolder = filenc.replace('L2_','L2S').replace('.nc','_'+v)
	filefile = filefolder.replace('L2S__','L2S'+f'{(3+nbeam)*2:02d}')+'.nc'
	file_ODL=PATH_ODL+YYYY+'/'+DDD+'/'+filefolder+'/'+filefile
	return file_ODL

def file_L2P_from_file_L2(file_L2,PATH_L2,PATH_L2P):
	file_L2P = ( file_L2[:len(PATH_L2)+5] + file_L2[len(PATH_L2)+9:] ).replace(PATH_L2,PATH_L2P).replace('____','PBOX')
	return file_L2P
	
def file_L2_from_file_L2P(file_L2P,PATH_L2P,PATH_L2):
	T1 = pd.Timestamp(file_L2P[-34:-19])
	file_L2 = file_L2P.replace('/CFO_','/'+f'{T1.day_of_year:03d}'+'/CFO_').replace(PATH_L2P,PATH_L2).replace('PBOX','____')

	return file_L2

def get_tracks_between_dates(T1,T2,typeTrack=0,pathstorage=None,str_to_remove=None,inciangle=8,verbose=False):
	# typeTrack : 0 = nadir L2 CNES, 1 = offnadir ODL, 2 = nadir L3 CMEMS
	if pathstorage==None:
		if typeTrack==0:
			pathstorage='/home/datawork-cersat-public/provider/cnes/satellite/l2/cfosat/swim/swi_l2____/op05/5.1.2/'
		elif typeTrack==1:
			pathstorage='/home/ref-cfosat-public/datasets/swi_l2s/v1.0/'
		elif typeTrack==2:
			pathstorage='/home/ref-cmems-public/tac/wave/WAVE_GLO_WAV_L3_SWH_NRT_OBSERVATIONS_014_001/dataset-wav-alti-l3-swh-rt-global-cfo/'

#	if (T2-T1)>pd.Timedelta("7 days"):
#		print('To avoid too many data loading, your time span should not exceed 7 days')
	if (typeTrack == 0)|(typeTrack==1):
		date_ini_fold=get_datetime_fromDOY(T1.year,T1.day_of_year)
		
	else:	
		date_ini_fold=pd.to_datetime(dt.datetime(T1.year,T1.month,1))
		
	list_tracks = []
	exceptss = []
	exceptss_counts = []
	count = 0
	count_last = 0
	date_last_appended = date_ini_fold
	while date_ini_fold<T2:
		yr = date_ini_fold.year
		doy = date_ini_fold.day_of_year
		month = date_ini_fold.month
		if (typeTrack == 0)|(typeTrack==1):
			path_date_ini = os.path.join(pathstorage,str(yr),f'{doy:03d}')
		else:
			path_date_ini = os.path.join(pathstorage,str(yr),f'{month:02d}')
		try:
			list_files=sorted(os.listdir(path_date_ini))
			
			for itrack in range(len(list_files)):
				name_track = list_files[itrack]
				if typeTrack == 1:
					name_track = name_track+'/'+name_track[:16]+f'{inciangle:02d}'+name_track[18:]+'.nc'
				
				d1 = pd.to_datetime(name_track[22:37])
				d2 = pd.to_datetime(name_track[38:53])
								
				if (name_track[-3:]=='.nc')&(d1<T2) & (d2>T1):
					file_track = os.path.join(path_date_ini,name_track)
					if str_to_remove==None:	
						list_tracks.append(file_track)
						date_last_appended = date_ini_fold
						count_last = count
					elif np.size(str_to_remove)==1:
						if str_to_remove not in name_track:
							list_tracks.append(file_track)
							date_last_appended = date_ini_fold
							count_last = count
					else:
						if all(st1 not in name_track for st1 in str_to_remove):
							list_tracks.append(file_track)
							date_last_appended = date_ini_fold
							count_last = count
		except Exception as inst:
			exceptss.append(str(inst)+'  '+str(date_ini_fold))
			exceptss_counts.append(count)
			
		# -- final part of the while loop
		if (typeTrack == 0)|(typeTrack==1):
			date_ini_fold = date_ini_fold + pd.Timedelta(days=1)
		else:
			month=month+1
			date_ini_fold = pd.to_datetime(dt.datetime(yr+(month-1)//12,((month-1)%12)+1,1))
		count = count +1
	
	if verbose==True:
		A=np.array(exceptss_counts)
		for ia,_ in enumerate(A[A<count_last]):
			print(exceptss[ia])
	
	return list_tracks,date_last_appended

def get_files_for_1_year(yea,nbeam=2,str_to_remove=['OPER','TEST'],PATH_L2S='/home/ref-cfosat-public/datasets/swi_l2s/v1.0/',\
			PATH_L2 = '/home/datawork-cersat-public/provider/cnes/satellite/l2/cfosat/swim/swi_l2____/op05/5.1.2/',\
			PATH_L2P = '/home1/datawork/mdecarlo/CFOSAT_L2P/swim_l2p_box_nrt/'):
	if yea==2020:
		T1=pd.to_datetime('2020-01-01')
		T2=pd.to_datetime('2021-01-01')
		# --- read files L2 and L2S between T1 and T2 ----------------------------------------
		list_L20,_ = get_tracks_between_dates(T1,T2,str_to_remove=str_to_remove,inciangle=(3+nbeam)*2,verbose=False)
		list_L2P0 = [file_L2P_from_file_L2(f,PATH_L2,PATH_L2P) for f in list_L20]
		
		list_L2S0,_ = get_tracks_between_dates(T1,T2,typeTrack=1,str_to_remove=str_to_remove,inciangle=(3+nbeam)*2)
	
	elif yea==2021:
		T1=pd.to_datetime('2021-01-01')
		T2=pd.to_datetime('2022-01-01')
		
		# -- get tracks between dates L2 v 5.1.2   -------------------
		list_L200,last_date_append = get_tracks_between_dates(T1,T2,str_to_remove=str_to_remove,inciangle=(3+nbeam)*2,verbose=False)
		# -- get tracks between dates L2 v 5.2.0   -------------------
		pathstorage='/home/datawork-cersat-public/provider/cnes/satellite/l2/cfosat/swim/swi_l2____/op05/5.2.0/'
		list_L201,_ = get_tracks_between_dates(last_date_append,T2,pathstorage=pathstorage,str_to_remove=str_to_remove,inciangle=(3+nbeam)*2)
		list_L2P00 = [file_L2P_from_file_L2(f,PATH_L2,PATH_L2P) for f in list_L200]
		list_L2P01 = [file_L2P_from_file_L2(f,pathstorage,PATH_L2P) for f in list_L201]

		list_L20 = list_L200+list_L201
		list_L2P0 = list_L2P00+list_L2P01

		list_L2S0,_ = get_tracks_between_dates(T1,T2,typeTrack=1,str_to_remove=str_to_remove,inciangle=(3+nbeam)*2)

	# --- change tracks L2 to L2S naming ------------------------------
	list_L2_as_L2S = [file_L2S_from_file_L2(fl,PATH_L2S,'1.0.0',nbeam,islocal=0) for fl in list_L20]
	# --- get tracks L2(as L2S) in L2S list ------------------------------
	list_L2S_intersec = [f for f in list_L2_as_L2S if f in list_L2S0]

	indfile = [list_L2_as_L2S.index(x) for x in list_L2S_intersec]
	list_L2_intersec = [list_L20[indi] for indi in indfile]
	list_L2P = [list_L2P0[indi] for indi in indfile]
	print('nb files intersection L2 and L2S :', len(list_L2_intersec),' over ',len(list_L20))
	
	return list_L2_intersec,list_L2S_intersec,list_L2P

# ==================================================================================
# === 1. Functions to xr.open CFOSAT files =========================================
# ==================================================================================
def preprocess_boxes_env_work_quiet(ds0):
	#ds = ds0[["min_lat_l2","max_lat_l2","min_lon_l2","max_lon_l2",
	ds = ds0[["lat_spec_l2", "lon_spec_l2", "wave_param", "k_spectra" ,"phi_vector","nadir_swh_box", "nadir_swh_box_std", "flag_valid_swh_box","indices_boxes"]]
	ds = ds.assign(time_box=decode_two_columns_time(ds0.time_l2,"n_tim"))

	# Prepare transformations (jacobian and slope to waves)
	ds=ds.assign(wave_spectra_kth_hs=ds0['pp_mean']*(ds['k_spectra']**-1))
	ds=ds.assign({"dk":(("nk"),np.gradient(ds['k_spectra'].compute().data))})

	ds = ds.rename_vars({"k_spectra":"k_vector"})
	da_concat=xr.concat([ds0["min_lat_l2"],ds0["max_lat_l2"],ds0["max_lat_l2"],ds0["min_lat_l2"],ds0["min_lat_l2"]],'n_corners')
	ds=ds.assign(lat_corners=da_concat)
	da_concat=xr.concat([ds0["min_lon_l2"],ds0["min_lon_l2"],ds0["max_lon_l2"],ds0["max_lon_l2"],ds0["min_lon_l2"]],'n_corners')
	ds=ds.assign(lon_corners=da_concat)
	ds=ds.rename_vars({"lat_spec_l2":"lat","lon_spec_l2":"lon",})
	ds=ds.swap_dims({'n_posneg':'isBabord'})
	ds=ds.swap_dims({'n_box':'time0'})
	return xr.decode_cf(ds)
	
def read_boxes_from_L2_CNES_env_work_quiet(L2):
	ds_L2=xr.open_mfdataset(L2,concat_dim="time0", combine="nested",decode_times=False,
		decode_coords=False,data_vars='minimal',coords="minimal",compat='override',autoclose=True,
		preprocess=preprocess_boxes_env_work_quiet)
	return ds_L2

def preprocess_offnadir_env_work_quiet(ds0):
	ds = ds0[["phi","lat","lon","k","time", "wave_spectra","dk"]]
	ds = ds.assign({"wave_spectra_kth_hs":((ds.wave_spectra*ds.k**-1))})
	ds = ds.swap_dims({'time':'time0'})
	ds = ds.reset_coords('time')
	ds = ds.rename_vars({'time':'time_per_angle'})
	ds = ds.rename_vars({"k":"k_vector"})
	ds = ds.rename_vars({"phi":"phi_vector"})
	ds = ds.swap_dims({"k":"nk"})
	return ds
	
def read_l2s_offnadir_files_env_work_quiet(offnadir_files):
	ds_l2s = xr.open_mfdataset(offnadir_files,concat_dim="l2s_angle", combine="nested",decode_coords=False,autoclose=True,
		data_vars='minimal',coords="all",compat='override',preprocess=preprocess_offnadir_env_work_quiet)
	return ds_l2s


def get_indices_macrocycles(inds):
	# inds = ds_l20['indices_boxes'].isel(n_box=35,n_posneg=1,n_beam_l1a=3+nbeam)
	indssel = []
	for i in np.arange(0,inds.size,2):
		if np.isfinite(inds.isel(ni=i)):
			indssel.append(np.arange(inds.isel(ni=i),inds.isel(ni=i+1)+1,dtype='int'))
	return np.concatenate(indssel)
	
# ==================================================================================
# === 2. 2D Functions ==============================================================
# ==================================================================================
def myconv2D(x, h):
    assert np.shape(x) == np.shape(h), 'Inputs to periodic convolution '\
                               'must be of the same period, i.e., shape.'

    X = np.fft.fft2(x)
    H = np.fft.fft2(h)
    
    nx = np.size(X,0)
    ny = np.size(X,1)

    return np.roll(np.real(np.fft.ifft2(np.multiply(X, H))),[nx//2+1,ny//2+1],axis=[0,1])

# --- functions from env2 to env ---------------------------------------------------
def from_env2_to_env(spec_env2,Hs):
    return spec_env2*2*(4-np.pi)/(Hs**2)

def from_env_to_spec_Hs(spec_env):
    coeff = (4*np.sqrt(2/np.pi))**2#8
    return coeff*spec_env

def calc_footprint_diam(Hs,pulse_width = 1/(320*1e6),Rorbit=519*1e3,Rearth = 6370*1e3):
    clight= 299792458
    Airemax_div_pi = Rorbit*(clight*pulse_width + 2 * Hs)/(1+(Rorbit/Rearth))
    return 2*np.sqrt(Airemax_div_pi)

# ==================================================================================
# === 2. Function to work over a 2D spectrum =======================================
# ==================================================================================
def prep_interp_grid(dkmin=0.28/(2**10),kmax=0.28):
	# --- define k interpolation vector (positive part) ----
	kX00 = np.arange(0,kmax,dkmin)
	# --- duplicate to have both positive and negative parts --------------
	kX0 = np.concatenate((-np.flip(kX00)-kX00[1],kX00))
	# --- generate 2D grid ------------------------
	kX,kY = np.meshgrid(kX0,kX0 , indexing='ij')
	# --- compute associated K, Phi(in deg) ---------
	kK = np.sqrt(kX**2+kY**2)
	kPhi = np.arctan2(kY,kX)*180/np.pi
	kPhi[kPhi<0]=kPhi[kPhi<0]+360

	kK2 = xr.DataArray(kK, coords=[("kx", kX0), ("ky",kX0)])

	kPhi2 = xr.DataArray(kPhi, coords=[("kx", kX0), ("ky",kX0)])
	kKkPhi2s = xr.Dataset(
		{'kK': kK2,
		'kPhi': kPhi2}
		).stack(flattened=["kx", "ky"])
	return kKkPhi2s

def from_spec_CFOSAT_to_2S(ds):
	# ds_CNES_sel[['k_vector','phi_vector','wave_spectra_kth_hs','dk']]
	Spec_1 = ds.copy(deep=True)
	Spec_1["phi_vector"].values = (Spec_1["phi_vector"].compute().data+180.)%360
	Spec_2 = xr.concat([ds,Spec_1],dim="n_phi",data_vars='minimal')
	Spec_2['wave_spectra_kth_hs'].values = Spec_2['wave_spectra_kth_hs'].values/2
	# Spec_2['wave_spectra_kth_hs'].values[np.isnan(Spec_2['wave_spectra_kth_hs'].values)]=0

	Spec_2 = Spec_2.sortby('phi_vector')
	return Spec_2

def from_spec_CFOSAT_to_1S(ds):
	Spec_1 = ds.copy(deep=True)
	Spec_1["phi_vector"].values = (Spec_1["phi_vector"].compute().data+180.)%360
	Spec_1['wave_spectra_kth_hs'].values = 0*Spec_1['wave_spectra_kth_hs'].values
	Spec_2 = xr.concat([ds,Spec_1],dim="n_phi",data_vars='minimal')
	Spec_2['wave_spectra_kth_hs'].values = 1.*Spec_2['wave_spectra_kth_hs'].values
	# Spec_2['wave_spectra_kth_hs'].values[np.isnan(Spec_2['wave_spectra_kth_hs'].values)]=0

	Spec_2 = Spec_2.sortby('phi_vector')
	return Spec_2

def interp_from_spec_0360(spec,kKkPhi2s):
	try:
		spec_bis = xr.concat([spec.isel(n_phi=slice(-2,None)),spec,spec.isel(n_phi=slice(0,2))],dim="n_phi",data_vars='minimal')
		# --- change the first and last new values to have a 2pi revolution ---------------
		A = np.concatenate([[-360],[-360],np.zeros((spec.dims['n_phi'])),[360],[360]])
		factor = xr.DataArray(A, dims="n_phi")
		spec_bis['phi_vector'].values = spec_bis['phi_vector']+factor
		spec_bis = spec_bis.interpolate_na(dim='n_phi')
		spec_bis = spec_bis.isel(n_phi=slice(1,-1))

		dphis = np.diff(spec_bis['phi_vector'].values)
		dphi = xr.DataArray(0.5*(dphis[0:-1]+dphis[1:]),dims='n_phi')*np.pi/180
		Hsnew = 4*np.sqrt((spec['wave_spectra_kth_hs']*dphi*spec['dk']).sum(dim=['n_phi','nk']).data)
		# print('inside interp function : ',spec_bis['wave_spectra_kth_hs'].sizes,' k vector : ',spec_bis['k_vector'].sizes)
		Ekxky0, _, _ = spectrum_to_kxky(1,np.squeeze(spec_bis['wave_spectra_kth_hs'].compute().data),  spec_bis['k_vector'].compute().data,spec_bis["phi_vector"].compute().data)

		# define the spectrum as a dataArray to apply the interp
		Speckxky = xr.DataArray(Ekxky0, dims=("nk", "n_phi"), coords={"nk":spec_bis['k_vector'].astype(np.float64) , "n_phi":spec_bis["phi_vector"]})
		del Ekxky0
		# --- apply interpolation -----------------------------------
		B = Speckxky.interp(nk=kKkPhi2s.kK,n_phi=kKkPhi2s.kPhi,kwargs={"fill_value": 0})
		B.name ='Ekxky_new'
		B0 = B.reset_coords(("nk","n_phi"))
		Ekxky_2 = B0.Ekxky_new.unstack(dim='flattened')
		# Ekxky_2 : -fmax : 0 :fmax

		return Ekxky_2, Hsnew   
	except Exception as inst:
		print('in interp : ',inst,', line number : ',sys.exc_info()[2].tb_lineno)

def define_filter_Gaussian(Xa_c,Ya_c,L0_L2,isnormdx=0):
	twopi = 2*np.pi
	[Xa_c2,Ya_c2] = np.meshgrid(Xa_c, Ya_c, indexing='ij')
	phi_x00 = np.exp(-0.5* Xa_c2**2 / (L0_L2)**2 )*np.exp(-0.5* Ya_c2**2 / (L0_L2)**2)
	if isnormdx == 0:
		divi = (L0_L2**2*twopi)
	else:
		divi = np.sum(phi_x00)
	phi_x0 = xr.DataArray(phi_x00/divi,
				dims=['x','y'],
				coords={
				    "x" : Xa_c,
				    "y" : Ya_c,
				    },
				)
	return phi_x0
	
def define_filter_annexA(Xa_c,Ya_c,DiamChelton,nkx_c,nky_c,dx_c,dy_c):
# Uses approximation r0**2/rc**2 = R0/Hs 
	twopi = 2*np.pi
	rc = DiamChelton/2
	[Xa_c2,Ya_c2] = np.meshgrid(Xa_c, Ya_c, indexing='ij')
	
	r0 = np.sqrt((Xa_c2)**2+(Ya_c2)**2)

# Defines a Gaussian filter scaled with rc 
	G_Lc20 = np.exp(-0.5* r0**2 / (rc)**2 )
	G_Lc2 = G_Lc20/(rc**2*twopi)

	Id = np.zeros(np.shape(G_Lc2))
	Id[nkx_c//2,nky_c//2]=1/(dx_c*dy_c)

#  This is the same as Jr0= A / (pi*h*Hs) * J 
	Jr0 = (4*dx_c*dy_c/(np.pi*rc**2)) * (r0/rc)**2 * (6 - ((2*r0/rc)**4)) * np.exp(- 4 * r0**4 / rc**4)
# WHY THIS NORMALIZATION ??? 
	Jr0 = Jr0/np.sum(Jr0*dx_c*dy_c)

	Jr1 = fftconvolve((Id-G_Lc2),Jr0,mode='same')*dx_c*dy_c

	Filter_new = (G_Lc2+Jr1)
# WHY THIS NORMALIZATION AGAIN  ??? 
	Filter_new = Filter_new /np.sum(Filter_new*dx_c*dy_c)	

	phi_x0 = xr.DataArray(Filter_new,
				dims=['x','y'],
				coords={
				    "x" : Xa_c,
				    "y" : Ya_c,
				    },
				)
	return phi_x0	

def define_filter_J2_annexA(Xa_c,Ya_c,DiamChelton,nkx_c,nky_c,dx_c,dy_c):

	twopi = 2*np.pi
	rc = DiamChelton/2
	[Xa_c2,Ya_c2] = np.meshgrid(Xa_c, Ya_c, indexing='ij')
	
	r0 = np.sqrt((Xa_c2)**2+(Ya_c2)**2)

	G_Lc20 = np.exp(-0.5* r0**2 / (rc)**2 )
	G_Lc2 = G_Lc20/(rc**2*twopi)

	Id = np.zeros(np.shape(G_Lc2))
	Id[nkx_c//2,nky_c//2]=1/(dx_c*dy_c)

	J20 = (dx_c*dy_c/(np.pi*rc**2)) * (2 - ((2*r0/rc)**4)) * np.exp(- 4 * r0**4 / rc**4)
	J20 = J20/np.sum(J20*dx_c*dy_c)

	Jr2 = fftconvolve((Id-G_Lc2),J20,mode='same')*dx_c*dy_c

	Filter_new = (Jr2)
	Filter_new = Filter_new /np.sum(Filter_new*dx_c*dy_c)	

	phi_x0 = xr.DataArray(Filter_new,
				dims=['x','y'],
				coords={
				    "x" : Xa_c,
				    "y" : Ya_c,
				    },
				)
	return phi_x0		
	
def estimate_stdHs_from_spec2D(ds_sel0,kKkPhi2s,L1S,Hs=None):
	try:
		# ---- 1. Spec L2 ---------------------------
		# ---- 1.1 turn spec to phi = [0:360] ---------------------------
		ds_sel1 = ds_sel0.copy(deep=True)
		ds_sel1["phi_vector"].values = (ds_sel1["phi_vector"].compute().data+180.)%360
		ds_sel = xr.concat([ds_sel0,ds_sel1], dim="n_phi", data_vars='minimal')
		ds_sel['wave_spectra_kth_hs'].values = ds_sel['wave_spectra_kth_hs'].values/2
		ds_sel = ds_sel.sortby('phi_vector')

		# ---- 1.2 Interp to new kx,ky grid ---------------------------
		Ekxky,Hs_0 = interp_from_spec_0360(ds_sel,kKkPhi2s)
		if Hs is None:
			Hs = Hs_0
		dkx = np.gradient(Ekxky.kx)[0]
		dky = np.gradient(Ekxky.ky)[0]
		kx_c = Ekxky.kx.compute().data
		nkx_c = Ekxky.sizes['kx']
		nky_c = Ekxky.sizes['ky']
		dx_c = twopi/(nkx_c*dkx)
		dy_c = twopi/(nky_c*dky)
		Xa_c = dx_c*(np.arange(-nkx_c//2,nkx_c//2)+0.5)
		Ya_c = dy_c*(np.arange(-nky_c//2,nky_c//2)+0.5)
		
		# -- Compute Lambda2 -------------------------------------------
		Lambda2 = Hs_0**4/(256*np.sum(Ekxky.compute().data**2)*dkx*dky)
		# ---- 1.3 Convolution ---------------------------
		Spec2D_env2_from_convol2D = 8*(myconv2D(Ekxky,np.flip(Ekxky)))*dkx*dky
		
		
		# ---- 1.4 Define filters ---------------------------------
		Diam_chelton = calc_footprint_diam(Hs)
		# --- integrate up to various k1 --------------------
		# L1S = [(7/5)*1e3,7*1e3,77*1e3,80*1e3]
		int_specHs= np.zeros((2,len(L1S)+1))
		
		L0_L2 = Diam_chelton/9
		dkL0 = 1/(np.sqrt(2)*L0_L2)
		nxa = int(4*dkL0//dkx)
		
		# --- Cut the convolution ---------------
		Spec2D_env2_from_convol2D = Spec2D_env2_from_convol2D[(nkx_c//2-nxa):(nkx_c//2+nxa),(nky_c//2-nxa):(nky_c//2+nxa)]
		Spec2D_Hs_from_convol2D = from_env_to_spec_Hs(from_env2_to_env(Spec2D_env2_from_convol2D,Hs))
		
		for ifilt in range(2):
			if ifilt ==0:
			# ---- 1.4.a Filter Gaussian ------------------------------
				phi_x0 = define_filter_Gaussian(Xa_c,Ya_c,L0_L2)
				
			elif ifilt==1:
			# ---- 1.4.b Good Filter (Annex A) ------------------------
				phi_x0 = define_filter_annexA(Xa_c,Ya_c,Diam_chelton,nkx_c,nky_c,dx_c,dy_c)
				
			phi0_hat_k = xrft.power_spectrum(phi_x0, dim=["x", "y"])*twopi*twopi # function of ktild = k/twopi
			phi0_hat_k = phi0_hat_k[(nkx_c//2-nxa):(nkx_c//2+nxa),(nky_c//2-nxa):(nky_c//2+nxa)]
			# phi0_hat_k = phi0_hat_ktild0*twopi*twopi	
				
			# ---- 1.5 Apply filters and integrate---------------------
			SpecHs_filt_div = phi0_hat_k*Spec2D_Hs_from_convol2D/(dkx*dky)
			# -- int over y ---------------------
			SpecHs_filt_div_inty = SpecHs_filt_div.sum('freq_y')*dky
			spec_Hs_funcx = spi.interp1d(kx_c[(nkx_c//2-nxa):(nkx_c//2+nxa)],SpecHs_filt_div_inty)
			
			# -- store the total integral --------------
			int_specHs[ifilt,-1] = SpecHs_filt_div_inty.sum('freq_x')*dkx
			
			for i1,L1 in enumerate(L1S):
				k1 = twopi/(2*L1)
				int_specHs_remove = spint.quad_vec(spec_Hs_funcx,-k1,k1)[0]
				int_specHs[ifilt,i1] = int_specHs[ifilt,-1] - int_specHs_remove

		return np.sqrt(int_specHs), Hs_0  ,Lambda2 
	except Exception as inst:
		print('inside estimate_std function ',inst,'line bis :',sys.exc_info()[2].tb_lineno)
		return np.zeros((2,len(L1S)+1)), 0  ,0 	
	
# ==================================================================================
# === 1. Function to work over a track (only 2D) ===================================
# ==================================================================================
def function_one_track_2D(indf,files,nbeam,isbabord):
	print(indf,' start ')
	file_L2 = files[0]
	file_L2S = files[1]
	file_L2P = files[2]
	L1S = [(7/5)*1e3,7*1e3,77*1e3,80*1e3]
	print(file_L2S)
	try:
		kKkPhi2s = prep_interp_grid(dkmin=0.28/(2**10))#dkmin=0.00016755)
		
		# --- read files CNES ------------ 
		ds_boxes = read_boxes_from_L2_CNES_env_work_quiet(file_L2)

		# --- read files ODL ------------ 
		ds_l2s = read_l2s_offnadir_files_env_work_quiet(file_L2S).isel(l2s_angle=0)
		
		ntim = ds_boxes.dims['time0']
		time_box = np.zeros((ntim))
		
		# --- get the flags from L2P -----------------------------------
		ds_L2P = xr.open_dataset(file_L2P).isel(n_posneg = isbabord)
		
		flag_valid_L2P_swh_box0 = ds_L2P['flag_valid_swh_box'].compute().data
		flag_valid_L2P_spec_box0 = 1*((ds_L2P['flag_valid_pp_mean'].sum(dim=['nk', 'n_phi']).compute().data) > 1) 
                                     
		ds_L2P.close()
		
		print('ntim = ',ntim)
		# --- Initialize variables -----------------------------------
		std_Hs_L2_2D = np.zeros((ntim,2,len(L1S)+1))
		std_Hs_L2S_2D = np.zeros((ntim,2,len(L1S)+1))

		Hs_box = np.zeros((ntim))
		std_Hs_box = np.zeros((ntim))
		lat_box = np.zeros((ntim))
		lon_box = np.zeros((ntim))
		
		Hs_L2_2D = np.zeros((ntim))
		Hs_L2S_2D = np.zeros((ntim))
		Lambda2_L2_2D = np.zeros((ntim))
		Lambda2_L2S_2D = np.zeros((ntim))
			
		for it in range(ntim):
			# print(it, 'over ',ntim,' -------------')
			ds_CNES_sel = ds_boxes.isel(isBabord=isbabord,time0=it,n_beam_l2=nbeam,n_beam_l1a=3+nbeam)
			# --- 0. Get global values from box ------------------------------
			std_Hs_box[it] = ds_CNES_sel['nadir_swh_box_std'].compute().data
			Hs_box[it] = ds_CNES_sel['nadir_swh_box'].compute().data
			time_box[it] = ds_CNES_sel['time_box'].compute().data
			lat_box[it] = ds_CNES_sel['lat'].compute().data
			lon_box[it] = ds_CNES_sel['lon'].compute().data
		
			if ds_CNES_sel['flag_valid_swh_box'] == 1:
				# --- invalid --------------------------
				std_Hs_L2_2D[it,:,:] = np.nan
				Lambda2_L2_2D[it] = np.nan
				Hs_L2_2D[it] = np.nan
				
				std_Hs_L2S_2D[it,:,:] = np.nan
				Lambda2_L2S_2D[it] = np.nan
				Hs_L2S_2D[it] = np.nan
				
			else:
				# --- valid ----------------------------
				# ---- 0.a select indices for macrocycles L2S ------
				ind_L2S = get_indices_macrocycles(ds_CNES_sel['indices_boxes'])
				# ---- 0.b select macrocycles L2S from indices ------
				ds_ODL_sel = ds_l2s.isel(time0 = ind_L2S).swap_dims({'time0':'n_phi'}).set_coords('phi_vector')	
				# ---- 1. Spec L2 ---------------------------
				std_Hs_L2_2D[it,:,:], Hs_L2_2D[it], Lambda2_L2_2D[it] = estimate_stdHs_from_spec2D(ds_CNES_sel, kKkPhi2s, L1S)
	    
				# ---- 2. Spec L2S ---------------------------
				std_Hs_L2S_2D[it,:,:], Hs_L2S_2D[it], Lambda2_L2S_2D[it] = estimate_stdHs_from_spec2D(ds_ODL_sel, kKkPhi2s, L1S)

		print(indf,' end of work in file')
		return indf, time_box, Hs_box, std_Hs_box, lat_box, lon_box, std_Hs_L2_2D, Hs_L2_2D,Lambda2_L2_2D, std_Hs_L2S_2D, Hs_L2S_2D, Lambda2_L2S_2D, flag_valid_L2P_swh_box0, flag_valid_L2P_spec_box0

	except Exception as inst:
		print(inst,indf,'line :',sys.exc_info()[2].tb_lineno, file_L2S)
		return indf,0,0,0,0,0,0,0,0,0,0,0,0,0,0






