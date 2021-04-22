from sklearn.model_selection import train_test_split
import numpy as np

def split_in_sequences(data, length, keep_short_batch=True):
    out = []
    for i in range(int(np.floor(data.shape[0]/length))):
        out.append(data[i*length:(i+1)*length])

    if keep_short_batch:
        out.append(data[int(np.floor(data.shape[0]/length))*length:]) # append last batch (possibly shorter than length)
    return out 

def merge_sequences(data):
    return np.concatenate(data, axis=0)

def train_test_validation_split(X, y,train_size = 0.6,test_size = 0.2,validation_size = 0.2):
    X = np.array(X)
    y = np.array(y)
    X_batched = split_in_sequences(X, length=80)
    y_batched = split_in_sequences(y, length=80)
    
    X_train, X_test, y_train, y_test = train_test_split(X_batched, y_batched, random_state=42, test_size=test_size + validation_size)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test,random_state=42, test_size=validation_size/(test_size + validation_size))
    X = {'train':X_train, 'test':X_test, 'valid':X_valid}
    Y = {'train':y_train,'test':y_test,'valid':y_valid}
    for key in X.keys():
        X[key] = merge_sequences(X[key])
        Y[key] = merge_sequences(Y[key])
    return X,Y
    