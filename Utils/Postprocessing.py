import numpy as np

def smooth_predictions(predictions, window_length):
    
    smoothed_pred = np.zeros(predictions.shape)
    
    for i in range(predictions.shape[0] - window_length + 1):
        win = predictions[i : i + window_length]
        smoothed_pred[i] = np.round(np.average(win, axis=0))
    
    # smooth remaining values
    win = predictions[-window_length:]
    for i in range(predictions.shape[0] - window_length + 1, predictions.shape[0]):    
        smoothed_pred[i] = np.round(np.average(win, axis=0))
            
    return smoothed_pred

