import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve

'''
Input predictions and a threshold value for a key to be pressed
Returns an array of 0s and 1s corresponding to the predicted keys
Threshold should be between 0 and 1
'''
def predict_keys(predictions, threshold=0.5):
    return tf.math.floor(predictions+1.0-threshold)

'''
Input predictions and labels
Returns the confusion matrix of key pressing being predicted correctly 
'''
def get_confusion_matrix(predictions, groud_truth):
    return tf.math.confusion_matrix(predict_keys(tf.reshape(predictions,[-1]), 0.5), tf.reshape(groud_truth, [-1]))


def get_metric(predictions, ground_truth, metric='accuracy', threshold=0.5):
    predictions = predict_keys(tf.reshape(predictions,[-1]), threshold=threshold)
    labels = tf.reshape(ground_truth, [-1])
    
    if metric=='accuracy':
        return accuracy_score(y_true=labels, y_pred=predictions)
    elif metric=='precision':
        return precision_score(y_true=labels, y_pred=predictions)
    elif metric=='recall':
        return recall_score(y_true=labels, y_pred=predictions)
    elif metric=='F1':
        return f1_score(y_true=labels, y_pred=predictions)
    else:
        print("unknown metric")

def plot_metric(model, metric):
    train_metrics = model.history[metric]
    val_metrics = model.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()
    
def plot_roc_curve(predictions, ground_truth):
    fpr, tpr, thresholds = roc_curve(tf.reshape(ground_truth,[-1]), tf.reshape(predictions,[-1]))
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1])
    plt.title("ROC Curve")
    plt.ylabel("True Positives")
    plt.xlabel("False Positives")
    plt.show()