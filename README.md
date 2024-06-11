# wavesALTI
Forward (surface simulation, radar waveform simulation) and inverse modelling (waveform retracking) tools for looking at ocean wind-generated waves in altimetry data and the sensitivity of different effects. Note that the forward modelling tools are relatively basic and assume geometrical optics and (so far) neglect many effects that would be taken into account in a proper end-to-end simulator (see RSSS by Nouguier et al. 2019). In contrast the inverse modelling (retracking) is identical to the tools used for actual data processing, for example as part of the ESA-funded Sea State CCI project.    

Main contributors: Marcello Passaro, Fabrice Ardhuin and Marine De Carlo

At present this tools package focuses on LRM (Delay-Only) altimetry. It was used for the following papers: 
http://dx.doi.org/10.13140/RG.2.2.32296.17925



## Installation 
git clone --quiet https://github.com/ardhuin/wavesALTI
