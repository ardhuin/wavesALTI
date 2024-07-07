# wavesALTI
Forward (surface simulation, radar waveform simulation) and inverse modelling (waveform retracking) tools for looking at ocean wind-generated waves in altimetry data and the sensitivity of different effects. Note that the forward modelling tools are relatively basic and assume geometrical optics and (so far) neglect many effects that would be taken into account in a proper end-to-end simulator (see RSSS by Nouguier et al. 2019). In contrast the inverse modelling (retracking) is identical to the tools used for actual data processing, for example as part of the ESA-funded Sea State CCI project.    

Main contributors: Marcello Passaro, Fabrice Ardhuin and Marine De Carlo

The WHALES algorithm is defined here: 
Passaro, M., & Algorithm Development Team. (2021). Algorithm theoretical basis document (atbd), sea state climate change initiative (Tech. Rep.). European Space Agency. Retrieved from https://climate.esa.int/media/documents/Sea_State_cci_ATBD_v3.0-signed.pdf

At present this tools package focuses on LRM (Delay-Only) altimetry. It was used for the following papers: 

https://doi.org/10.1038/s41467-021-23982-4 Global coastal attenuation of wind-waves observed with radar altimetry.

https://doi.org/10.1029/2023JC019740   Wave groups and small scale variability of wave heights observed by altimeters

https://doi.org/10.1029/2023JC020832   Along-Track Resolution and Uncertainty of Altimeter-Derived Wave Height and Sea Level



This package aims at unifiying the original WHALES code developed by M. Passaro with the altimeter simple simulations and retracking tools developed by F. Ardhuin for the analysis of wave group effects. The current status is that we have 3 flavours of the 
retrackers, and we are slowly converging to only two, and maybe one:  

1)- WHALES_withRangeAndEpoch.py           : this is the retracker code used for the ESA SeaState CCI project, look at python_WHALES_launcher.py for an example of how to use it. It works on NetCDF GDR data. 

2)- altimetry_waveforms_functions.py      : this is a simplified and generalized retracker, designed to test many options (maximum likelihood vs least squares ...). Works with SWIM netCDF data or simulated spectra. See Figure2_retracking_CFOSAT.ipynb for an example. 

3)- altimetry_waveforms_functions_new.py  : this one should soon replace the previous one. It allows range-dependent weights, as in the WHALES retracker. Works with SWIM netCDF data or simulated spectra. See Figure2_retracking_CFOSAT_WHALES.ipynb for an example. 


Ongoing work: 
- adapting 3) to SWOT Poseidon 3C data
- updating all notebooks to get rid of 2) 
- updating 1) to allow retracking of SWIM and SWOT data. 
- comparing 1) and 3) on the same data to make sure all the steps of the WHALES algorithm are well undersood. (See Passaro, ATBD, also Passaro et al., in prep). 


## Installation 
git clone --quiet https://github.com/ardhuin/wavesALTI

## Running the WHALES retracker 
Here is an example, using SARAL/AltiKa data, from the command line: it reads the file with waveforms specified by the -i option and dumps the retracked result as NetCDF files in the output directory specified by -o . 

python python_WHALES_launcher.py -m saral -i /home/ardhuin/PUBLI/2023_groups/DATA/SRL_GPS_2PfP001_0641_20130405_141055_20130405_150113.CNES.nc  -o OUTPUT


## Various notebooks 
Figure2_retracking_CFOSAT.ipynb     : this generates the plots of Figure 2 and D1 in De Carlo & Ardhuin (2024), with the retracking of some SWIM waveforms

Figure5_perturbed_waveforms.ipynb   : generates Figure 5 of De Carlo & Ardhuin (2024), showing waveforms with wave group perturbations

Figure6_J_functions.ipynb 
and 
Figure6_J_WHALES.ipynb              : generates  Figure 6 f De Carlo & Ardhuin (2024), showing J filter functions. 
