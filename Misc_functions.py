#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================
# === 0. Import Packages ===========================================================
# ==================================================================================
import glob
import os
import sys
import pathlib

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.text as mtext
import matplotlib.colors as mcolors
import matplotlib as mpl
import matplotlib.cm as cmx
from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.lines import Line2D

# to interpolate values

import numpy as np
import netCDF4 as nc
import xarray as xr

import datetime as dt
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
twopi = 2*np.pi


mpl.rcParams.update({'font.size': 18,'savefig.facecolor':'white'})

## - Define class LegendTitle to have subtitles in Legend
## - Define class HandlerColumnLines to have multiple lines in 1 column as legend entry
## - py_files(PATH, suffix='.nc')
## - get_rolling_values(x,kernel_size)
## - get_scattering_info(x,kernel_size,type_metric=0)
## - filter_big_change_from_ref(X,Xref,thresh)
## - haversine(lat1, lon1, lat2, lon2)
## - crosses_land(disttocoast,lat1, lon1, lat2, lon2)
## - interpolate_along_track(lon_mod,lat_mod,field_mod,lon_obs,lat_obs) 
## - get_contour(X_array,Y_array)
## - copy_NCfile(originalfile,targetfile,isdebug)
## - add_ocean_label(lon,lat)
## - init_map_cartopy(ax0,limx=(-180,180),limy=(-80,80))
## -
## -
## -
## -
'''
- read_offnadir_files(filenames):  Function that reads the offnadir files and appends the results outputs: time_swim,lat_offnadir,lon_offnadir,phi,phi_geo,seg_lat,seg_lon,modulation_spectra,fluctuation_spectra,wave_spectra,klin,inci,seg_inci,kwave,dk
- read_nadir_data(filename):time_alti0, lat_alti0, lon_alti0, swh_alti0: all values along track
#            - time_alti, lat_alti, lon_alti, swh_alti: filtered values (by availability)
- read_nadir_data_wind(filename): reads one nadir file at 5 Hz (native)
- read_nadir_data_wind_1_5Hz(filename): time_alti0,lat_alti0,lon_alti0,swh_alti_1hz0,wind_alti_1hz0,swh_alti_5hz0,wind_alti_5hz0,time_alti,lat_alti,lon_alti,swh_alti_1hz,wind_alti_1hz,swh_alti_5hz,wind_alti_5hz

- read_box_data_one_side(filename,iswest):
- read_Model_fields(filename): # 2D
- copy_NCfile(originalfile,targetfile,isdebug):
- apply_box_selection_to_track(lon,lat,time,field,lon_min,lon_max,lat_min,lat_max):
- draw_spectrum_macrocyle(axs,fluctu_spec_mm,phi_geo_mm,phi_loc_mm,freq,wvlmin,wvlmax,dphi,cNorm,cMap): 
- interpolate_along_track(lon_mod,lat_mod,field_mod,lon_obs,lat_obs):
- get_contour(X_array,Y_array):
- class LegendTitle(object):
- get_macrocycles(lon_offnadir,lat_offnadir,phi_loc,phi_geo,lon_min,lon_max,lat_min,lat_max):
- 
'''


# --------------------------------------------------------------------------------
# -- class LegendTitle to have subtitles in Legend
class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()
    
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, r'\textbf{' + orig_handle + '}', usetex=True, **self.text_props)
        handlebox.add_artist(title)
        return title

# --------------------------------------------------------------------------------
# -- class HandlerColumnLines to have multilines as column in Legend
class HandlerColumnLines(HandlerLineCollection):
    """
    Custom Handler for LineCollection instances.
    """
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # figure out how many lines there are
        numlines = len(orig_handle.get_segments())
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)
        leglines = []
        # divide the vertical space where the lines will go
        # into equal parts based on the number of lines
        ydata = np.full_like(xdata, height / (numlines + 1))
        # for each line, create the line at the proper location
        # and set the dash pattern
        for i in range(numlines):
            legline = Line2D(xdata, ydata * (numlines - i) - ydescent)
            self.update_prop(legline, orig_handle, legend)
            # set color, dash pattern, and linewidth to that
            # of the lines in linecollection
            try:
                color = orig_handle.get_colors()[i]
            except IndexError:
                color = orig_handle.get_colors()[0]
            try:
                dashes = orig_handle.get_dashes()[i]
            except IndexError:
                dashes = orig_handle.get_dashes()[0]
            try:
                lw = orig_handle.get_linewidths()[i]
            except IndexError:
                lw = orig_handle.get_linewidths()[0]
            lw=1
            if dashes[1] is not None:
                legline.set_dashes(dashes[1])
            else:
                legline.set_dashes('')
            legline.set_color(color)
            legline.set_transform(trans)
            legline.set_linewidth(lw)
            leglines.append(legline)
        return leglines

def py_files(root,suffix='.nc'): 
    """Recursively iterate all the .nc files in the root directory and below""" 
    for path, dirs, files in os.walk(root):
        if suffix[0]=='.':
            yield from (os.path.join(path, file) for file in files if pathlib.Path(file).suffix == suffix)
        else:
            yield from (os.path.join(path, file) for file in files if file[-len(suffix):] == suffix)

# --------------------------------------------------------------------------------
# --- 1. Get values for a rolling windows ----------------------------------------
# --------------------------------------------------------------------------------
def get_rolling_values(x,kernel_size):
    lenx=len(x)
    rank_kernel=kernel_size//2
    B=np.zeros(lenx+2*rank_kernel)
    B[rank_kernel:-rank_kernel]=x
    
    indA = np.atleast_2d(np.arange(rank_kernel)) + np.atleast_2d(np.arange(lenx)).T
    
    return B[indA]
    
# --------------------------------------------------------------------------------
# --- 2. Get scattering info -----------------------------------------------------
# --------------------------------------------------------------------------------  
def get_scattering_info(x,kernel_size,type_metric=0):
    y=get_rolling_values(x,kernel_size)
    med = np.nanmedian(y,axis=1)

    if type_metric==0: # std
        r = np.std(y,axis=1)
    elif type_metric==1: # RMS from median
        r=np.sum((y-np.tile((med),(kernel_size//2,1)).T)**2,axis=1)
    elif type_metric==2: # Max-min
        r=(np.amax(y,axis=1)-np.amin(y,axis=1))/np.nanmedian(y,axis=1)
    return r    

# --------------------------------------------------------------------------------
# --- 3. Filter big change from ref ---------------------------------------------
# --------------------------------------------------------------------------------     
def filter_big_change_from_ref(X,Xref,thresh):
    condition1= (np.isfinite(X))&(X!=0)&(np.isfinite(Xref))&(Xref!=0)
    ind = np.where(condition1)[0]
    
    X0 = np.inf*np.ones(X.shape)
    X0[ind]=np.exp(np.abs(np.log(X[ind]/Xref[ind])))
    
    Xlim=thresh*np.nanmedian(np.abs(X0[np.isfinite(X0)]))

    ind_suspect1 = np.where(np.abs(X0)>Xlim)[0]
    X1 = np.copy(X)
    X1[ind_suspect1]=np.nan
    
    return X1    
    
# -- function HAVERSINE ------------------------------------
# Calculates the distance [km] between 2 points on the globe
# using the Haversine formula
# inputs : lat1, lon1, lat2, lon2
# outputs: distance between the 2points in km
def haversine(lat1, lon1, lat2, lon2):
    ''' This code is contributed by ChitraNayal from https://www.geeksforgeeks.org/haversine-formula-to-find-distance-between-two-points-on-a-sphere/
    distance between latitudes and longitudes (in km) '''
    dLat = (lat2 - lat1) * np.pi / 180.0
    dLon = (lon2 - lon1) * np.pi / 180.0
    
    # convert to radians
    lat1 = lat1 * np.pi / 180.0
    lat2 = lat2 * np.pi / 180.0
    
    # apply formulae
    a = (pow(np.sin(dLat / 2), 2) +
         pow(np.sin(dLon / 2), 2) *
             np.cos(lat1) * np.cos(lat2));
    
    rad = 6371
    c = 2 * np.arcsin(np.sqrt(a))
    return rad * c        

# -- function CROSSES_LAND ------------------------
def crosses_land(disttocoast,lat1,lon1,lat2,lon2):
    ''' Function to determine if there is a continent between 2 points.
    with disttocoast obtained from \n 
    disttocoast = xr.open_dataarray('/home1/datahome/mdecarlo/Python_functions/distance2coast.nc')
    '''
    Num_steps = 1000
    lons = np.linspace(lon1,lon2,Num_steps)
    lats = np.linspace(lat1,lat2,Num_steps)


    lonxx = xr.DataArray(lons, dims=("llx"), coords={"llx":np.arange(Num_steps)})
    latxx = xr.DataArray(lats, dims=("llx"), coords={"llx":np.arange(Num_steps)})

    tointer = xr.Dataset({'lon': lonxx,'lat': latxx})

    D = disttocoast.interp(lon=tointer.lon,lat=tointer.lat,kwargs={'fill_value':np.nan}).data
    return np.any(np.isnan(D))

# ---  Linear ND Interpolator -----------------------------------------------
def interpolate_along_track(lon_mod,lat_mod,field_mod,lon_obs,lat_obs):
#  np.shape(field_mod)=(len(lat_mod),len(lon_mod))
    import scipy.interpolate as spi
    x, y = np.meshgrid(lon_mod, lat_mod)
    hs_interp = spi.LinearNDInterpolator(((x.flatten(),y.flatten())),field_mod.flatten())
    field_interp = hs_interp(lon_obs,lat_obs)
    
    return field_interp

def compute_dB(P,P0=10**-12):
    return 10*np.log10(P/P0)   
     
# -- function GET_CONTOUR ------------------------------------
# inputs : X_array,Y_array : 2D arrays which contours need to be extracted
# outputs: X_bound,Y_bound
def get_contour(X_array,Y_array):
    X_bound = np.zeros(2*np.size(X_array,0)+2*np.size(X_array,1))
    Y_bound = np.zeros(2*np.size(Y_array,0)+2*np.size(Y_array,1))
    
    count0 = np.size(X_array,0)
    X_bound[0:count0]=X_array[:,0]
    count1 = count0+np.size(X_array,1)
    X_bound[count0:count1]=X_array[-1,:]
    count0 = count1
    count1 = count1 + np.size(X_array,0)
    X_bound[count0:count1]=X_array[-1::-1,-1]
    count0 = count1
    count1 = count1 + np.size(X_array,1)
    X_bound[count0:count1]=X_array[0,-1::-1]
    
    count0 = np.size(Y_array,0)
    Y_bound[0:count0]= Y_array[:,0]
    count1 = count0+np.size(Y_array,1)
    Y_bound[count0:count1]=Y_array[-1,:]
    count0 = count1
    count1 = count1 + np.size(Y_array,0)
    Y_bound[count0:count1]=Y_array[-1::-1,-1]
    count0 = count1
    count1 = count1 + np.size(Y_array,1)
    Y_bound[count0:count1]=Y_array[0,-1::-1]
    
    return X_bound,Y_bound 

def get_contour_box_fromvectors(x,y):
    lenX = len(x)
    lenY = len(y)
    x_contour=np.zeros(2*lenX+2*lenY)
    y_contour=np.zeros(2*lenX+2*lenY)

    x_contour[0:lenX]=x
    y_contour[0:lenX]=np.ones(lenX)*y[0]
    x_contour[lenX:lenX+lenY]=x[-1]*np.ones(lenY)
    y_contour[lenX:lenX+lenY]=y
    x_contour[lenX+lenY:2*lenX+lenY]=x[-1::-1]
    y_contour[lenX+lenY:2*lenX+lenY]=y[-1]*np.ones(lenX)
    x_contour[2*lenX+lenY:2*(lenX+lenY)]=x[0]*np.ones(lenY)
    y_contour[2*lenX+lenY:2*(lenX+lenY)]=y[-1::-1]

    return x_contour,y_contour

    
# ---- add ocean label ---------------------------------
def add_ocean_label(lon,lat,shape=None,order_k = None):
    import geopandas
    if shape is None:
        shape = geopandas.read_file("/home/mdecarlo/Documents/DATA/GOaS_v1/GOaS_v1_20211214/goas_v01.shp")
    ocean_label=np.zeros(np.shape(lon),dtype=int)-1 # set default = -1
    points_geo=geopandas.points_from_xy(lon,lat)
    lS = len(shape)
    if order_k is None:
        order_k=np.arange(lS)# I don't remember why order_k...
    ocean_names=[]
    ocean_names_short=[]
    ocean_names.append('Land') # set default to Land
    ocean_names_short.append('Land') # set default to Land
    for ik,k in enumerate(order_k):#range(lS):
        # --- get the name of oceans - both short and complete 
        A=shape.name[k].split(' ',-1)
        if len(A)<=3:
            if len(A)==3:
                A[0]=A[0][0]+'.'
            B = ' '.join(A[:-1])
        elif len(A)>3:
            B = A[-2]
        ocean_names.append(shape.name[k])
        ocean_names_short.append(B)
        # --- for each polygon check if points are inside : ------------------------
        # -- convex_hull : simplifies polygon and makes 'within()' quicker
        # SK = shape.geometry[order_k[k]]
        SK = shape.geometry[k]
        if SK.geom_type=='MultiPolygon':
            for k_in in range(len(SK.geoms)):
                SG = SK.geoms[k_in].convex_hull
                ind = points_geo.within(SG)
                # ocean_label[ind]=order_k[k]
                ocean_label[ind] = k
        else:
            SG = SK.convex_hull
            ind=points_geo.within(SG)
            # ocean_label[ind]=order_k[k]
            ocean_label[ind] = k

    return ocean_label,ocean_names,ocean_names_short


def init_map_cartopy(ax0,limx=(-180,180),limy=(-80,80)):
    ''' output : ax0, g1
    where g1 are the gridlines
    '''
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature    
    
    ax0.set_extent([limx[0],limx[1],limy[0],limy[1]],crs=ccrs.PlateCarree())
    # Grid Lines of the map ----------------------
    g1 = ax0.gridlines(ccrs.PlateCarree(), draw_labels=True)
    g1.xlabels_top = False
    g1.ylabels_right = False

    g1.xlabel_style = {'size': 18}
    g1.ylabel_style = {'size': 18}

    # --- Add land features ---- 
    # Definition ---
    land = cfeature.NaturalEarthFeature(
    category='physical',
    name='land',
    scale='50m',
    edgecolor='face',
    zorder=-10,
    facecolor=cfeature.COLORS['land'])
    # add the features ---
    ax0.add_feature(land)
    ax0.coastlines()

    return ax0, g1


def multiple_histo_groupby(DatasetGroupedby,xbins,var,ax,space=0.2,xticklabels=None,xticklabelsRot=90,xticks=None,ylabel=None,groupLabels=None):
# --- Histo of the variable 'var' from a dataset groupedby
    LenGroup=len(DatasetGroupedby)
    x=np.arange(len(xbins)-1)
    print(len(x))
    x0 = x - 0.5 + (space/2)
    width = (1-space)/LenGroup
    # x = 0.5*(bins[1:]+bins[0:-1])
    count_label=0
    for grlabel, group in DatasetGroupedby:
        counts, _ = np.histogram(group[var],bins=xbins)
        if groupLabels != None:
                grlabel=groupLabels[count_label]
        _= ax.bar(x0 + count_label*width, counts, width, label=grlabel)
        count_label=count_label+1
# ax.hist(ds_old.Ocean_flag,bins=np.arange(-2,11)+0.5,alpha=0.5)
    if xticks!=None:
        _=ax.set_xticks(xticks)
    else:
        _=ax.set_xticks(x)
    if ylabel!=None:
        _=ax.set_ylabel(ylabel)
    if xticklabels!=None:
        _=ax.set_xticklabels(xticklabels,rotation=xticklabelsRot)
    _=plt.legend()
    plt.grid()

    
def get_datetime_fromDOY(YYYY,DOY):
    import pandas as pd
    return pd.to_datetime(dt.datetime(YYYY,1,1))+pd.Timedelta(days=DOY-1)
    

def get_nearest_model_interpolator(lon_lims=[-180., 180.],lat_lims=[-78.,83.],lon_step=0.5,lat_step=0.5):
    import scipy.interpolate as spi
    lon_mod = np.arange(lon_lims[0],lon_lims[1],lon_step)
    lat_mod = np.arange(lat_lims[0],lat_lims[1],lat_step)
    lon_interpolator=spi.interp1d(lon_mod,lon_mod,kind='nearest')
    lat_interpolator=spi.interp1d(lat_mod,lat_mod,kind='nearest')

    return lon_interpolator, lat_interpolator

def get_nearest_model_interpolator_index(lon_lims=[-180., 180.],lat_lims=[-78.,83.],lon_step=0.5,lat_step=0.5):
    import scipy.interpolate as spi
    lon_mod = np.arange(lon_lims[0],lon_lims[1],lon_step)
    lat_mod = np.arange(lat_lims[0],lat_lims[1],lat_step)
    ilon_interpolator=spi.interp1d(lon_mod,np.arange(len(lon_mod)),kind='nearest')
    ilat_interpolator=spi.interp1d(lat_mod,np.arange(len(lat_mod)),kind='nearest')

    return ilon_interpolator, ilat_interpolator
    
def read_Hs_model(date1,path=None):
    if path == None:
        path='/home/ref-ww3/GLOBMULTI_ERA5_GLOBCUR_01/GLOB-30M/'
    yr = date1.year
    T1str=date1.strftime('%Y%m')
    path_year=os.path.join(path,str(yr),'FIELD_NC')
    strfile='LOPS_WW3-GLOB-30M_'+T1str+'.nc'
    ds=xr.open_dataset(os.path.join(path_year,strfile))
    ds1=ds.sel(time=slice(date1-np.timedelta64(90,'m'), date1+np.timedelta64(90,'m')))
    return ds1
    
# ---  plot 2D model for one date 
def plot_2D_model_Hs(ax,date1,path=None,limx=(-180,180),limy=(-80,80),norm=None):
    if path == None:
        path='/home/ref-ww3/GLOBMULTI_ERA5_GLOBCUR_01/GLOB-30M/'
    ds=read_Hs_model(date1,path=path)
    if norm==None:
        swhNorm = mcolors.Normalize(vmin=5, vmax=12)    
    ds1=ds.sel(longitude=slice(limx[0],limx[1]),latitude=slice(limy[0],limy[1]))
    
    ax,g1=init_map_cartopy(ax,limx=limx,limy=limy)
    
    im=ax.pcolormesh(ds1.longitude,ds1.latitude,np.squeeze(ds1.hs),norm=norm)
    return im,g1

def extract_Welch_tiles_1D(A,nxtile):
    # nxtiles : nb of pixels by tiles
    nxA = len(A)
    #     nxtile = int(nxA//Ntiles)
    Ntiles = int(nxA//nxtile)
    shx = int(nxtile//2)
    Ntiles_overlap = 2*Ntiles-1

    extract= np.zeros((Ntiles_overlap,nxtile))
    for windows in range(Ntiles_overlap):
        if windows < Ntiles:
            i1 = int(windows + 1)
            extract[windows,:]=A[nxtile*(i1-1):nxtile*i1]
        elif windows >= Ntiles:
            i1 = int((windows-Ntiles)+ 1)
            extract[windows,:]=A[nxtile*(i1-1)+shx:nxtile*i1+shx]
    
    return extract, Ntiles_overlap

