import numpy as np
import tensorflow as tf

'''
Input predictions and a threshold value for a key to be pressed
Returns an array of 0s and 1s corresponding to the predicted keys
Threshold should be between 0 and 1
'''
def predict_keys(predictions, threshold=0.5):
    return tf.math.floor(predictions+1-threshold)

'''
Input predictions and labels
Returns the confusion matrix of key pressing being predicted correctly 
'''
def get_confusion_matrix(predictions, groud_truth):
    return tf.math.confusion_matrix(predict_keys(tf.reshape(predictions,[-1]), 0.5), tf.reshape(groud_truth, [-1]))

