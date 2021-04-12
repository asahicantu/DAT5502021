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


def get_metric(predictions, groud_truth, metric='accuracy'):
    predictions = predict_keys(tf.reshape(predictions,[-1]))
    labels = tf.reshape(groud_truth, [-1])
    if metric=='accuracy':
        ac = tf.keras.metrics.Accuracy()
        ac.update_state(predictions, labels)
        return ac.result().numpy()
    elif metric=='precision':
        pr = tf.keras.metrics.Precision()
        pr.update_state(predictions, labels)
        return pr.result().numpy()
    elif metric=='recall':
        re = tf.keras.metrics.Recall()
        re.update_state(predictions, labels)
        return re.result().numpy()
    else:
        print("unknown metric")