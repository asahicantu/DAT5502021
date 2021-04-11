## Creating Functions to load and save pickle data
import pickle
import os
def dump_pickle(file_name,data):
    path = os.path.join('Data','Out','Pickle',file_name)
    print(path)
    with open(path, 'wb') as file:
        pickle.dump(data, file)

def load_pickle(file_name):
    path = os.path.join('Data','Out','Pickle',file_name)
    print(path)
    with open(path, 'rb') as file:
        return pickle.load(file)   