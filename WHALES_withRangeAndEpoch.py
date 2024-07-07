# -*- coding: utf-8 -*-
"""
Created on 11 February 2019

@author: passaro

WHALES Retracker corresponding to the WHALES version of the 4.12.2018, but with Epoch and Range in Output

Modification history:
	2023-01-10: corrections for SARAL by Marine De Carlo 
        2024-06-30: adaptation for wavesALTI package by F. Ardhuin 


"""


from Retracker_MP    import *
from waveform_models import waveform_brown_LS,waveform_brown_ML
import scipy
from scipy import stats
import cmath #handling of complex square root
from scipy import optimize
from scipy.optimize import minimize
from scipy import special
from scipy import signal
from math import erf
import math
import numpy as np
import matplotlib.pyplot as plt
#from akima_interpolate import interpolate as akima
' WHALES Retracker '



class WHALES_withRangeAndEpoch(Retracker_MP):
    
    def __init__(self,config):
        Retracker_MP.__init__(self, config)
        self.retrack_MP()


    def NM_fit(self,xdata,ydata,Zeta,tau,Theta,SigmaP,altitude,initial_conditions,mission,weights,weightflag,modelcost) :
        #Nelder-Mead fit of a waveform 
    
        #IT NEEDS:
            #1) xdata in NANOSECONDS (ex. xdata=[0*tau:1:127*tau])
            #2) ydata (waveform coefficient) in normalized units
            #3) Off-nadir angle in radians
    
        #OUTPUT:
            #1) x: vector of the estimated parameters
            #2) Wt: fitted waveform
            #3) exitflag: exit flag to check convergence
            #4) Err: Fitting Error on the portion of waveform given to the function
            #5) SWH: Significant Wave Height
        
        # WARNING: ZETA is xi converted to radians 
        
        incognita=initial_conditions
        
        c=3.0*(10**8) #Light speed
        H=altitude
        Ri=6378.1363*(10**3) #Earth radius

        Gamma=0.5 * (1/math.log(2))*np.sin(Theta)*np.sin(Theta) # antenna beamwidth parameter
     
        b_xi = np.cos (2*Zeta) - ((np.sin(2*Zeta))**2)/Gamma
        a=( (4/Gamma)*(c/H) * 1/(1+H/Ri))
        c_xi=b_xi* ( (4/Gamma)*(c/H) * 1/(1+H/Ri))
    
        a=a/1000000000 #/ns
        c_xi=c_xi/1000000000 #1/ns
        if modelcost == 'brown_LS':
           xopt = minimize(waveform_brown_LS, incognita, args=((ydata,Gamma,Zeta,xdata,SigmaP,c_xi,weights,weightflag),) ,method='Nelder-Mead',options={'disp': False})
        elif modelcost == 'brown_ML':
           xopt = minimize(waveform_brown_ML, incognita, args=((ydata,Gamma,Zeta,xdata,SigmaP,c_xi,weights,weightflag),) ,method='Nelder-Mead',options={'disp': False})

        
        if xopt.success == True:
            exitflag=1 #xopt[4]
        else:
            exitflag=0
        
        #Calculation of the fitted waveform
        x=xopt.x #This are the parameters estimated in minimised

        SWH_squared=( - SigmaP**2 + x[1]**2 )
        if SWH_squared>=0 :    
            SWH=cmath.sqrt( np.abs(- SigmaP**2 + x[1]**2)  ).real * (2*0.3)   # factor 2*0.3 converts sigma_time in nanosecond to 4*sigma in meters
        else:
            SWH=- cmath.sqrt( np.abs(- SigmaP**2 + x[1]**2) ).real * (2*0.3)
        SigmaS=SWH/(2*0.3) 
        Sigma=cmath.sqrt(SigmaS**2+SigmaP**2).real
        t=xdata-x[0]
        v=c_xi*(t-c_xi*(Sigma**2)/2)
        u=np.divide( (t-c_xi*(Sigma**2)) , (np.sqrt(2)*Sigma) )
        A=x[2]/2*np.exp((-4/Gamma)*(np.sin(Zeta))**2)
        Wt=A*np.exp(-v)*(1+scipy.special.erf(u))
        
    
    
        if np.size(ydata)>0: 
            Err=np.sqrt( 1./np.size(ydata) * np.sum( (ydata-Wt)**2 ))  #Fitting Error
        else:
            Err=np.nan
        
        return x, Wt, exitflag, Err, SWH
    
    
    def Conversion_NMbrown(self,x,mission) :
        
        # This function converts the Epoch estimated by NM_fit into an Epoch referred to the nominal point of the mission
        # In input it takes x, the vector of parameters estimated by NM_fit            
        
        c=0.3 # Light speed divided by a factor 10^-9
        
        # THIS IS MISSION DEPENDENT!!!
        if mission.lower() == 'envisat':
            tau=3.125 #gate spacing in ns
            nominal_tracking_gate=45
        if mission.lower() == 'saral' or mission.lower() == 'saral_igdr':
            tau=3.125*320/480 #gate spacing in ns
            nominal_tracking_gate=51            
        if mission.lower() == 'jason2' or mission.lower() == 'jason1' or mission.lower() == 'jason3': 
            tau=3.125
            nominal_tracking_gate=31
        if mission.lower() == 'ers2_r' or mission.lower() == 'ers2_r_2cm': 
            tau=3.03
            nominal_tracking_gate=33
        if mission.lower() == 'cs2_lrm':
            tau=3.125 #gate spacing in ns
            nominal_tracking_gate=64            
        
        t0=x[0] # ns
        Sigma=x[1]
        Au=x[2]
        
        #============== Calculate Epoch from t0=========================
        Epoch=t0 - nominal_tracking_gate*tau;  
        Epoch=Epoch*(c/2)  #m conversion from ns to meters(1gate=0.468m->c*tau/2)
        
        return(Epoch,t0,Sigma,Au)


    def akima_interpolate(self,x, y, x_new, axis=-1, out=None):
        """Return interpolated data using Akima's method.

        This Python implementation is inspired by the Matlab(r) code by
        N. Shamsundar. It lacks certain capabilities of the C implementation
        such as the output array argument and interpolation along an axis of a
        multidimensional data array.

        Parameters
        ----------
        x : array like
        1D array of monotonically increasing real values.
        y : array like
        N-D array of real values. y's length along the interpolation
        axis must be equal to the length of x.
        x_new : array like
        New independent variables.
        axis : int
        Specifies axis of y along which to interpolate. Interpolation
        defaults to last axis of y.
        out : array
        Optional array to receive results. Dimension at axis must equal
        length of x.

        Examples
        --------
        >>> interpolate([0, 1, 2], [0, 0, 1], [0.5, 1.5])
        array([-0.125,  0.375])
        >>> x = np.sort(np.random.random(10) * 10)
        >>> y = np.random.normal(0.0, 0.1, size=len(x))
        >>> z = interpolate(x, y, x)
        >>> np.allclose(y, z)
        True
        >>> x = x[:10]
        >>> y = np.reshape(y, (10, -1))
        >>> z = np.reshape(y, (10, -1))
        >>> interpolate(x, y, x, axis=0, out=z)
        >>> np.allclose(y, z)
        True

        """
        x = np.array(x, dtype=np.float64, copy=True)
        y = np.array(y, dtype=np.float64, copy=True)
        xi = np.array(x_new, dtype=np.float64, copy=True)

        if axis != -1 or out is not None or y.ndim != 1:
            raise NotImplementedError("implemented in C extension module")

        if x.ndim != 1 or xi.ndim != 1:
            raise ValueError("x-arrays must be one dimensional")

        n = len(x)
        if n < 3:
            raise ValueError("array too small")
        if n != y.shape[axis]:
            raise ValueError("size of x-array must match data shape")

        dx = np.diff(x)
        if any(dx <= 0.0):
            raise ValueError("x-axis not valid")

        if any(xi < x[0]) or any(xi > x[-1]):
            raise ValueError("interpolation x-axis out of bounds")

        m = np.diff(y) / dx
        mm = 2.0 * m[0] - m[1]
        mmm = 2.0 * mm - m[0]
        mp = 2.0 * m[n - 2] - m[n - 3]
        mpp = 2.0 * mp - m[n - 2]

        m1 = np.concatenate(([mmm], [mm], m, [mp], [mpp]))

        dm = np.abs(np.diff(m1))
        f1 = dm[2:n + 2]
        f2 = dm[0:n]
        f12 = f1 + f2

        ids = np.nonzero(f12 > 1e-9 * np.max(f12))[0]
        b = m1[1:n + 1]

        b[ids] = (f1[ids] * m1[ids + 1] + f2[ids] * m1[ids + 2]) / f12[ids]
        c = (3.0 * m - 2.0 * b[0:n - 1] - b[1:n]) / dx
        d = (b[0:n - 1] + b[1:n] - 2.0 * m) / dx ** 2

        bins = np.digitize(xi, x)
        bins = np.minimum(bins, n - 1) - 1
        bb = bins[0:len(xi)]
        wj = xi - x[bb]

        return ((wj * d[bb] + c[bb]) * wj + b[bb]) * wj + y[bb]


    
    def retrack_MP(self):

        mission = self.mission
        waveform = np.array(self.waveform)
        
            
        # IN THE FOLLOWING LINES; THE SPECIFIC CHARACTERISTICS OF EACH MISSIONS ARE DEFINED
        # NOTE THAT:
        # Waveforms are not oversampled, because Jason has been tested with the addition of 
        # weights, whose distribution might change if we oversample the waveform.             
            
        if mission.lower() == 'envisat':
                index_originalbins=np.arange(0,127,1) #Gate index of the waveform samples
                total_gate_number=128                
                noisegates=np.arange(4,10); #gates used to estimate Thermal Noise
                tau=3.125 #gate width in nanoseconds
                startgate=4 #First gate to be considered in the retracking window
                ALEScoeff0=2.45 #experimental values for SWH. it is the constant term in the definition of the number of gates to be considered in the retracking
                                #after the middle of the leading edge
                ALEScoeff1=4.05 #This is the slope of the WHALES relationship between tolerance of precision and width of the subwaveform   
                Err_tolerance_vector=0.3; #Tolerance on the (normalised) fitting error of the waveform. It can be used, for example,
                                                        #to retrack the same waveform in a different way if fitting performances are not satisfactory

        elif mission.lower() == 'saral' or mission.lower() == 'saral_igdr':
                index_originalbins=np.arange(0,127,1) #Gate index of the waveform samples
                total_gate_number=128                
                noisegates=10+np.arange(4,10); #gates used to estimate Thermal Noise # changed by Marine De Carlo
                tau=3.125*320/480 #gate width in nanoseconds
                startgate=4 #First gate to be considered in the retracking window
                ALEScoeff0=2.94 #experimental values for SWH. it is the constant term in the definition of the number of gates to be considered in the retracking
                                #after the middle of the leading edge
                ALEScoeff1=3.56 #This is the slope of the WHALES relationship between tolerance of precision and width of the subwaveform   
                Err_tolerance_vector=0.3; #Tolerance on the (normalised) fitting error of the waveform. It can be used, for example,
                                                        #to retrack the same waveform in a different way if fitting performances are not satisfactory              

   
        elif mission.lower() == 'jason2' or mission.lower() == 'jason1' or mission.lower() == 'jason3': 
                index_originalbins=np.arange(0,103,1) #Gate index of the waveform samples
                total_gate_number=104                
                noisegates=np.arange(0,6); #gates used to estimate Thermal Noise
                tau=3.125 #gate width in nanoseconds
                startgate=1 #First gate to be considered in the retracking window
                ALEScoeff0=3.89 #experimental values for SWH. it is the constant term in the definition of the number of gates to be considered in the retracking
                                #after the middle of the leading edge
                ALEScoeff1=3.86 #This is the slope of the WHALES relationship between tolerance of precision and width of the subwaveform   
                Err_tolerance_vector=0.3; #Tolerance on the (normalised) fitting error of the waveform. It can be used, for example,
                                                        #to retrack the same waveform in a different way if fitting performances are not satisfactory

        elif mission.lower() == 'cs2_lrm' :
                index_originalbins=np.arange(0,127,1) #Gate index of the waveform samples
                total_gate_number=128                
                noisegates=np.arange(4,10); #gates used to estimate Thermal Noise
                tau=3.125 #gate width in nanoseconds
                startgate=4 #First gate to be considered in the retracking window
                ALEScoeff0=3.68 #experimental values for SWH. it is the constant term in the definition of the number of gates to be considered in the retracking
                                #after the middle of the leading edge
                ALEScoeff1=3.36 #This is the slope of the WHALES relationship between tolerance of precision and width of the subwaveform   
                Err_tolerance_vector=0.3; #Tolerance on the (normalised) fitting error of the waveform. It can be used, for example,
                                                        #to retrack the same waveform in a different way if fitting performances are not satisfactory                 


        elif mission.lower() == 'ers2_r': 
                print("Mission not yet supported")
                sys.exit(0)
                
        elif mission.lower() == 'ers2_r_2cm':
                print("Mission not yet supported")
                sys.exit(0)

        else:
                print("unknown mission")
                sys.exit(0)
                
                
        # INITIAL DEFINITION OF EXIT VARIABLES

        self.Epoch=np.nan
        self.SWH=np.nan
        self.SWH_yang=np.nan
        self.Amplitude=np.nan
        self.Error = np.nan                  
        self.range = np.nan                     
        self.uncssh = np.nan #Uncorrected Sea Surface Height: not computed 
        Epoch_LESfive=np.nan                
                
                
                
            # STEP 1: NOISE ESTIMATION                 
        estimated_noise = np.nanmean(waveform[noisegates])
        C = waveform - estimated_noise;
            
            # STEP 2: WAVEFORM NORMALISATION
        igoodsample=C>np.max([5, 2*estimated_noise]) #5 and 2 are arbitrary factors here
        normalize_factor=1.3*np.nanmedian(C[igoodsample]);        
        D=C/normalize_factor
            
            # STEP 3: DEFINING RETRACKING ABSCISSA (xdata)
#        if mission.lower() == 'envisat':
#            xdata=np.arange(0,total_gate_number*tau,tau)
#        elif mission.lower() == 'jason2' or mission.lower() == 'ers2_r' or mission.lower() == 'ers2_r_2cm':
        xdata=np.empty([total_gate_number,])*np.nan
        xdata[index_originalbins]=np.arange(0,((np.size(index_originalbins)-1)*tau)+tau,tau)
        iii=1            
        for iii in np.arange(1,total_gate_number,2) :
            if iii==total_gate_number-1 :
                xdata[iii]=xdata[iii-1]+(xdata[iii-1]-xdata[iii-2])
            else:
                xdata[iii]=(xdata[iii+1]+xdata[iii-1])/2
                  
            
        
            # STEP 3.2: OTHER DEFINITIONS
        self.Wt_all_yang= np.empty(total_gate_number)*np.nan       
        self.Wt_all_LESfive= np.empty(total_gate_number)*np.nan  
        
            # STEP 4: FIND LEADING EDGE: For explanations see Passaro et al. 2014 and 2018
        edgestart=1
        edgeend=1
        
        # --- Changed by Marine De Carlo 
        # wv=D[index_originalbins] # old version of 'wv'
        wv0=D[index_originalbins]
        kernel_size = 3
        kernel = np.ones(kernel_size) / kernel_size
        wv = np.convolve(wv0, kernel, mode='same')
        
        Dwv=np.diff(wv)
        i = 4 #Gate where the search starts (avoids wrap-up of ERS for example)
        
        while i<=np.size(wv)-5 :
            #In order to be the leading edge, it doesn't have to go below
            #the 1% of its maximum in the space of few gates
            if Dwv[i]>0.01 and wv[i+1]>0.1 and wv[i+2]>0.1 and wv[i+3]>0.1 and wv[i+4]>0.1:
                edgestart=i # FOUND THE START OF THE LEADING EDGE
                
                while i<=np.size(Dwv)-4 :
                    #if the slope is negative for more than one gate, then the trailing edge
                    #is starting
                        if Dwv[i]<0 :
                            if Dwv[i+1]>0 and Dwv[i+2]>0 and Dwv[i+3]>0 : #if th following gates are still growing, it's only a perturbation of the leading edge
                                i=i+1
                            else :
                                edgeend=i # FOUND THE END OF THE LEADING EDGE                           
                                break
                        else :
                            i=i+1
                break
            else :
                i=i+1
        

        gate2=index_originalbins[edgeend+1]  #Gate2 is the end of the leading edge on the interpolated waveform. One gate of tolerance is added for numerical reasons (for example if a leading-edge-only retracking is attempted)
        gate1=index_originalbins[edgestart]  #Gate1 is the start of the leading edge on the waveform

        self.gate2=gate2
        self.gate1=gate1

        if np.isnan(D[gate2])==0:

            # STEP 5: RETRACKING WITH THE ALES RETRACKER
            
            x1_LESfive=np.empty(3)*np.nan  #Initialisation of vector of the estimated parameters (LESfive corresponds to the second retracking)
            x1_yang=np.empty(3)*np.nan     #Initialisation of vector of the estimated parameters (yang corresponds to the first retracking, restricted to the leading edge)
            Err_yang=1
    
            # STEP 5.1: FIRST PASS INITIAL CONDITIONS
            x_initial=(edgestart-1)*tau;
            sigma_initial=(edgeend-edgestart)*tau/(2*np.sqrt(2))
            ampl_initial=2*np.mean(D[gate1:gate2])
            
            # STEP 5.2: FIRST PASS RETRACKING
            tol_Err_yang=Err_tolerance_vector
            x1_yang[2]=0 #initialising to start the while cycle
            growingdue=0 #variation of the number of gates considered
            
            while Err_yang>tol_Err_yang and (gate2+growingdue)<total_gate_number and sigma_initial<100 :
            #while cycle: check on the Fitting Error, check that the number of available gates is not being exceeding, check that initial condition has been defined as a number (sigma_inital<100, could be removed)    
                
                weightflag=0  #No weights are used in the first (leading-edge only) retracker
                this_weights=np.ones(np.shape(xdata))
                
                x1_yang, Wt_yang, exitflag_yang, Err, SWH =\
                        self.NM_fit( xdata[startgate:gate2+growingdue+1] , D[startgate:gate2+growingdue+1],\
                        self.xi*math.pi/180,self.tau,self.Theta,self.SigmaP,self.hsat,\
                        np.array([x_initial, sigma_initial, ampl_initial]),mission,this_weights[startgate:gate2+growingdue+1],weightflag,modelcost='brown_ML')


                if gate2>gate1+1 and gate1>0 and np.size(Wt_yang)>1 and gate1>startgate and gate2>startgate+1 :
                            # Fitting Error computed on the leading edge
                            Err_yang=np.sqrt( 1/float(gate2-gate1-1) * np.sum( (D[gate1+1:gate2]-Wt_yang[gate1-(startgate-1):gate2-startgate])**2 ))
                            
                self.Wt_all_yang[startgate:gate2+growingdue+1]=Wt_yang
                #If the fitted waveforms only contains zeroes,
                #it's not the good one...
                if np.sum(Wt_yang==0)==np.size(Wt_yang):
                    Err_yang=1
                
                
                if exitflag_yang==0: 
                #if convergence is not reached, we might need more gates (growingdue+2) 
                    growingdue=growingdue+2
                else: #else then we can stop the while cycle
                        break
                    

            self.D=D  #D is the normalised original waveform
    
                    
                      
            del(growingdue)    
                    
            if np.isnan(x1_yang[2])==0 :
                Epoch_yang,Tau_yang,Sigma_yang,Au_yang=self.Conversion_NMbrown(x1_yang,mission)
                SWH_yang=SWH #SWH after the first retracking 
                # self.SWH_yang=SWH_yang #SWH after the first retracking              
                Sigma0_yang=10*np.log10(Au_yang*normalize_factor) #De-normalise the amplitude and save it as db 
            else:
                    Epoch_yang=np.nan 
                    Tau_yang=np.nan
                    SWH_yang=np.nan
                    # self.SWH_yang=np.nan
                    Au_yang=np.nan
                    Sigma0_yang=np.nan 
                    Sigma_yang=np.nan
                             
                             
                             
                             
                             
                             
             # STEP 5,3  WHALES EQUATION TO SET SECOND-PASS WINDOW: depending on the first estimation of SWH, a new width of the subwaveform is chosen, similarly 
                         #what is done in the ALES retracker (see Passaro et al. 2014 for the methodology). "stopgate" is the limit of the new subwaveform
             
            SWH_yang=np.real(SWH_yang)
            if SWH_yang<0:
                SWH_yang=0 # THE WHALES ALGORITHM IS DERIVED ONLY FOR POSITIVE SSB
            if SWH_yang<15:
                gateafterLE=np.ceil(ALEScoeff0+ALEScoeff1*np.abs(SWH_yang))                  
                stopgate=np.ceil(x1_yang[0]/tau)+gateafterLE            
            else: #we consider 15 m as the maximum possible SWH value
                gateafterLE=np.ceil(ALEScoeff0+ALEScoeff1*15)                   
                stopgate=np.ceil(x1_yang[0]/tau)+gateafterLE
            if stopgate<=total_gate_number: #if we need more than the actual number of gates, we use the full waveform
                pass
            else:
                stopgate=total_gate_number-1 #-1 because then we have the condition "(stopgate+growingdue)<total_gate_number"
             
            if xdata[-1]<stopgate*tau : #if the computed stopgate is bigger than the actual length of the vector
                                        #(can happen for very high SWH), then stopgate will be the end of the waveform
                stopgate=np.size(xdata)-1
            else: # FIND ON THE OVERSAMPLED WAVEFORM THE POSITION CORRESPONDING TO THE COMPUTED STOPGATE
                stopgate=np.nonzero(xdata==stopgate*tau)[0]-1; #-1 because the first element of xdata is 0 and not 3.125
            
            
            #we don't want to stop before the end of the
            #leading edge
            if stopgate>=gate2:
                pass
            else:
                stopgate=gate2
                
            if np.isscalar(stopgate) == False: #Addition to check that the output of stopgate is a scalar
                stopgate=stopgate[0]

            
            
            
    
    
                   
             # STEP 5.4 SECOND PASS RETRACKING                  
            exitflag_LESfive=0
            Err_LESfive=1  
            tol_Err_LESfive=Err_tolerance_vector
                    
            if Err_yang<1: #To make it faster: If the first pass was not able to retrack the leading edge, it's useless to go on
                    if Err_LESfive>tol_Err_LESfive:
                        growingdue=0 #variation of the number of gates considered
                        x1_LESfive[1]=0 #initializing to start the while cycle
                        while Err_LESfive>tol_Err_LESfive and (stopgate+growingdue)<total_gate_number and sigma_initial<100 : #first while cycle: check on the Fitting Error

                            weightflag=1
                            
                            #In WHALES, for each value of SWH there is a corresponding set of weights, which are defined only from the start to the end of the 
                            #leading edge of the synthetic waveforms generated in the Montecarlo simulation
                            
                            #In the following lines,the closest value of SWH is searched in the table, considering the exit of the first pass
                            weigths_SWH_vector=np.arange(0,10.5,0.5)
                            select_weights=np.argmin(  np.abs(np.abs(SWH_yang)-weigths_SWH_vector)  )
                                #Select the right line of weights
                            weights_select=self.weights[select_weights,:]
                            index_nanweights=np.where(np.isnan(weights_select))[0]
                            weights_select[index_nanweights]=1 #Transform the NaNs of the weight vector in ones
                            index_startweight=np.where(self.weights_flag[select_weights,:]==1)[0] #identify the start and the end of the leading edge in the weight vector
                            index_endweight=np.where(self.weights_flag[select_weights,:]==2)[0]
                            index_startweight=index_startweight[0] #convert array to index
                            index_endweight=index_endweight[0] #convert array to index
                            
                                #Now prepare a weight vector considering the start of the leading edge in this waveform
                            this_weights=np.ones(np.shape(xdata))
                            if gate1+index_endweight-index_startweight<total_gate_number :
                                this_weights[gate1:gate1+(index_endweight-index_startweight)]=weights_select[index_startweight:index_endweight]
                                this_weights=1./this_weights
                            
                            # Launche the second retracking process
                            x1_LESfive, Wt_LESfive, exitflag_LESfive, Err, SWH =self.NM_fit( xdata[startgate:stopgate+growingdue+1] , D[startgate:stopgate+growingdue+1],self.xi*math.pi/180,self.tau,self.Theta,self.SigmaP,self.hsat,np.array([x_initial, sigma_initial, ampl_initial]),mission,this_weights[startgate:stopgate+growingdue+1],weightflag,modelcost='brown_LS')
                            
                                     
                            self.Wt_all_LESfive[startgate:stopgate+growingdue+1]=Wt_LESfive  #This is the fitted subwaveform
        
                                
                            if gate2>gate1+1 and gate1>0 and np.size(Wt_LESfive)>1 and gate1>startgate and gate2>startgate+1 :
                                 #Fitting error on the leading edge
                                Err_LESfive=np.sqrt( 1/float(gate2-gate1-1) * np.sum( (D[gate1+1:gate2]-Wt_LESfive[gate1-(startgate-1):gate2-startgate])**2 ))
                                

                                
                                
                            if exitflag_LESfive==0 and tol_Err_LESfive==Err_tolerance_vector : #if convergence is not reached, we might need more gates (growingdue+1), otherwise we prepare to terminate the while cycle
        
                                growingdue=growingdue+2

                                    
                                    
                            else: # then we can stop the while cycle
                                growingdue=total_gate_number-gate2;
                                
                                
                                
                                 
                            #END OF THE SECOND WHILE CYCLE

            stopgate=stopgate-1  #This is not used anymore
                
                
#Bug with ML ...
#            x1_LESfive, Wt_LESfive, exitflag_LESfive, Err, SWH =self.NM_fit( xdata[startgate:stopgate+growingdue+1] , D[startgate:stopgate+growingdue+1],self.xi*math.pi/   180,self.tau,self.Theta,self.SigmaP,self.hsat,np.array([x_initial, sigma_initial, ampl_initial]),mission,this_weights[startgate:stopgate+growingdue+1],weightflag,modelcost='brown_ML')
                
                
                
            if np.isnan(x1_LESfive[2])==0 :
                Epoch_LESfive,Tau_LESfive,Sigma_LESfive,Au_LESfive=self.Conversion_NMbrown(x1_LESfive,mission)
                SWH_LESfive=SWH
    
                Sigma0_LESfive=10*np.log10(Au_LESfive*normalize_factor)  #db
            else :
                Epoch_LESfive=np.nan 
                Tau_LESfive=np.nan
                SWH_LESfive=np.nan
                Au_LESfive=np.nan
                Sigma0_LESfive=np.nan 
                Sigma_LESfive=np.nan
            
            
            


        # STEP 5.5 FINAL OUTPUT

#
        if np.isnan(Epoch_LESfive)==0:
            
            self.Epoch=Epoch_LESfive
            self.SWH=SWH_LESfive  #in m
            self.SWH_yang=SWH_yang
            self.Sigma=Sigma_LESfive #Rising time of the leading edge in ns
            self.Amplitude=Sigma0_LESfive#  Instead of the normalised amplitude(Au_LESfive), we are outputting the backscatter coefficient (before corrections)
            #self.Norm_Amplitude=Au_LESfive
            #self.leading_edge = np.nan
            self.Error = Err_LESfive  #Fitting Error on the leading edge
            
            self.range = self.uralt  + self.Epoch   #Range in m             
               
            #self.uncssh = self.hsat - self.range  #Can be removed
        else:
            self.range = np.nan
            self.uncssh = np.nan            
            self.SWH=np.nan 
            self.SWH_yang=np.nan
            self.Sigma=np.nan 
            self.Amplitude=np.nan 
            self.Norm_Amplitude=np.nan 
            #self.leading_edge = np.nan
            self.Error = np.nan 


        self.model = self.Wt_all_LESfive.copy()*normalize_factor  #Fitted Waveform
            
        #return
#        plt.figure()
#        plt.plot(self.D)
#        plt.plot(self.Wt_all_yang)
#        
#        plt.figure()
#        plt.plot(self.D)        
#        plt.plot(self.Wt_all_LESfive,'r')
#        
        
       

        
        
        
        
        
        
        
        
        
        

        
        
