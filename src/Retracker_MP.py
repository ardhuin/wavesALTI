#!/bin/python

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import traceback

from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.integrate import quad


# from scipy.integrate import simps

# from Database import *
# from Functions import *

class Retracker_MP:
    debug = 1

    ' Retracker configuration '
    waveform = None
    refbin = None
    binsize = None
    uralt = None
    threshold = None
    scale = 0
    mission = None
    hsat = None
    tau = None
    theta0 = None
    ref_range = None
    option = None
    factor = None
    weights_type = None
    costfunction = None
    weight_outsub = 1.
    sqrtn = np.sqrt(90.)
    nominal_tracking_gate = 31
    total_gate_number = None
    # weights = None #Weights used for the estimation of SWH
    # weights_flag = None #Flag that identifies start and end of the leading edge in the weight distribution

    ' Retracker results '
    model_x_m = None
    model_x = None
    model_y = None

    model = None
    range = None
    leading_edge = None
    error = None
    b1 = None
    b2 = None
    b3 = None
    b4 = None
    b5 = None

    # ADDITIONAL RETRACKER OUTPUT FROM ALES    
    gate1 = 0  # MP added initial gate of the leading edge
    gate2 = 0  # MP added end gate of the leading edge
    gate3 = 0
    noise = 0.0
    Wt_all_yang = None  # MP added normalised fitted waveform - first pass
    Wt_all_LESfive = None  # MP added normalised fitted waveform - second pass
    D = None  # MP added Normalised real waveform
    Epoch = None  # MP added Epoch (m)
    SWH = None  # MP added SWH (m)
    Amplitude = None  # MP added normalised amplitude
    Norm_Amplitude = None
    uncssh = None  # MP added Orbit Altitude - Retracked Range
    classifier = None
    interpolator = None
    Sigma = None  # MP added Rising Time of the leading edge
    smooth = 0 # MD added smoothing of waveform before detection of leading edge 

    ' speed of light [m/s] '
    c = 299792458

    def __init__(self, config):
        if 'waveform' in config:
            self.waveform = config['waveform']
            for i in range(0, len(self.waveform)):
                self.waveform[i] = float(self.waveform[i])

        if 'waveform_last' in config:
            self.waveform_last = config['waveform_last']
            for i in range(0, len(self.waveform_last)):
                self.waveform_last[i] = float(self.waveform_last[i])
        if 'waveform_current' in config:
            self.waveform_current = config['waveform_current']
            for i in range(0, len(self.waveform_current)):
                self.waveform_current[i] = float(self.waveform_current[i])
        if 'waveform_next' in config:
            self.waveform_next = config['waveform_next']
            for i in range(0, len(self.waveform_next)):
                self.waveform_next[i] = float(self.waveform_next[i])

        if 'refbin' in config:
            self.refbin = config['refbin']
        if 'binsize' in config:
            self.binsize = config['binsize']
        if 'uralt' in config:
            self.uralt = config['uralt']
        if 'threshold' in config:
            if float(config['threshold']) < 0.0 or float(
                    config['threshold']) > 100.0:
                print("Error: Threshold must be between 0 and 100%")
                sys.exit(0)
            self.threshold = float(config['threshold']) / 100.0
        if 'scale' in config and config['scale'] == 1:
            self.scale = 1

        ' improved threshold '
        if 'ref_range' in config:
            self.ref_range = config['ref_range']
        if 'option' in config:
            self.option = config['option']

        if 'factor' in config:
            self.factor = config['factor']

        # MP added
        ' ales_rev'
        if 'mission' in config:
            self.mission = config['mission']
        if 'hsat' in config:
            self.hsat = config['hsat']
        if 'theta0' in config:
            self.theta0 = config['theta0']
        if 'xi' in config:
            self.xi = config['xi']
        if 'doppler' in config:
            self.doppler = config['doppler']
        if 'Epoch' in config:
            self.Epoch = config['Epoch']
        if 'SWH' in config:
            self.SWH = config['SWH']
        if 'Amplitude_initial' in config:
            self.Amplitude_initial = config['Amplitude_initial']
        if 'classifier' in config:
            self.classifier = config['classifier']
        if 'weights' in config:
            self.weights = config['weights']
        if 'weights_flag' in config:
            self.weights_flag = config['weights_flag']

            # end MP addition
        # FA added
        ' CCI_rev'
        if 'tau' in config:
            self.tau = config['tau']
        if 'Theta' in config:
            self.Theta = config['Theta']
        if 'SigmaP' in config:
            self.SigmaP = config['SigmaP']
        if 'total_gate_number' in config:
            self.total_gate_number = config['total_gate_number']
        if 'nominal_tracking_gate' in config:
            self.ominal_tracking_gate = config['nominal_tracking_gate']
        if 'weights_type' in config:
            self.weights_type = config['weights_type']
        if 'costfunction' in config:
            self.costfunction = config['costfunction']
        if 'smooth' in config:
            self.smooth = config['smooth']

        ' debug flag '
        if 'debug' in config:
            self.debug = int(config['debug'])

        if self.__class__.__name__ == "Threshold" and self.threshold == None:
            print("Error: Threshold retracker needs argument 'threshold' [0, "
                  "..., 100]")
            sys.exit(0)
        if self.__class__.__name__ == "ImprovedThreshold" and self.threshold == None:
            print("Error: ImprovedThreshold retracker needs argument "
                  "'threshold' [0, ..., 100]")
            sys.exit(0)

    def plot(self):

        x_data = []
        y_data = []
        for i in range(0, len(self.waveform)):
            x_data.append(i)
            y_data.append(self.waveform[i])
        plt.plot(x_data, y_data, 'b.-')

        x_model = []
        y_model = []
        if self.model != None:
            for i in range(0, len(self.model)):
                x_model.append(i)
                y_model.append(self.model[i])
        plt.plot(x_model, y_model, 'r.-')

        if self.model_x != None and self.model_y != None:
            plt.plot(self.model_x, self.model_y, 'r-')

        x_leading_edge = []
        y_leading_edge = []
        if self.leading_edge != None:
            x_leading_edge.append(self.leading_edge)
            y_leading_edge.append(0)

            x_leading_edge.append(self.leading_edge)
            y_leading_edge.append(max(self.waveform))
        plt.plot(x_leading_edge, y_leading_edge, 'g.-')

        plt.ylabel('Power')
        plt.xlabel('Bin')

        plt.show()

    def scale_waveform(self, waveform):

        max_val = max(waveform)
        if max_val == 0:
            return waveform
        for i in range(0, len(waveform)):
            waveform[i] = waveform[i] / max_val
        return waveform

    def stat_waveform(self):
        info = {}
        info['max_power'] = None
        info['pos_max_power'] = None

        if self.waveform != None:
            info['max_power'] = max(self.waveform)
            for i in range(0, len(self.waveform)):
                if self.waveform[i] == info['max_power']:
                    info['pos_max_power'] = i
                    break
        return info

    def get_model(self):
        print("No model available for this retracker")
