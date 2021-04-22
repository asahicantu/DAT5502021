## Creating Functions to load and save pickle data
import pickle
import os
def dump_pickle(file_path,data):
    print(f'Saving pickle file into {file_path}')
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
    except Exception as e:
        print(f'Error while dumping {e}')

def load_pickle(file_path):
    print(f'Loading pickle file from {file_path}')
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)   
    except:
        return None
    