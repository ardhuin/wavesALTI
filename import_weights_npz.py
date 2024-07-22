# -*- coding: utf-8 -*-
"""


@author: Marcello Passaro
"""

import pickle
import numpy as np
import pickle

import pickle

def import_weights_pkl(pickle_file_path):
    try:
        with open(pickle_file_path, 'rb') as file:
            SWH_values = pickle.load(file)
            residual_std = pickle.load(file)
            flag_edges = pickle.load(file)
            #print('examples :',SWH_values[0:10],np.shape(flag_edges),'WHERE:',np.where(flag_edges[:,10]==1)[0])
            return SWH_values, np.transpose(residual_std), np.transpose(flag_edges)
    except FileNotFoundError:
        print(f"File {pickle_file_path} not found.")
    except Exception as e:
        print(f"An error occurred while loading variables from {pickle_file_path}: {e}")
    return None, None, None


def import_weights_npz(pickle_file_path):
    try:
        with open(pickle_file_path, 'rb') as file:
            data = pickle.load(file)
            print(f"Loaded data: {data}")  # Debugging line to print the loaded data
            # Ensure data is a dictionary and contains the key 'residual_std'
            if isinstance(data, dict):
                SWH_values = data.get('SWH_values')
                residual_std = data.get('residual_std')
                flag_edges = data.get('flag_edges')  # Assuming this is another key
                if residual_std is None:
                    raise KeyError("The key 'residual_std' is not present in the pickle file.")
                if flag_edges is None:
                    raise KeyError("The key 'flag_edges' is not present in the pickle file.")
                print('coucou:', len(residual_std), len(data))
                return SWH_values,np.transpose(residual_std), np.transpose(flag_edges)
            else:
                raise TypeError("Data loaded from pickle file is not a dictionary.")
    except FileNotFoundError:
        print(f"File {pickle_file_path} not found.")
    except pickle.UnpicklingError:
        print("Error unpickling the file.")
    except KeyError as e:
        print(f"KeyError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None, None, None


 

#def import_weights_npz(my_path_weights)     :
#    data = np.load(my_path_weights,allow_pickle=True)
#    for k in data.keys():
#        print(k+' = data["'+k+'"]')
#        exec(k+' = data["'+k+'"]')
#        exec('print("cou:","'+k+'",len('+k+'))')
#    print('coucou:',len(data))
#    print('coucou:',len(residual_std),len(data))
#    return residual_std, flag_edges
    

def import_weights_mat(my_path_weights)     :
    import h5py  
    print('COUCOU:',my_path_weights)
    mat_weights = h5py.File(my_path_weights,'r')
    residual_std=np.transpose(mat_weights['residual_tot']) 
    flag_edges=np.transpose(mat_weights['flag_edges'])
    
    return residual_std, flag_edges
