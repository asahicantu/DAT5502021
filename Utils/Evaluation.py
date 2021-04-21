import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import os

'''
Input predictions and a threshold value for a key to be pressed
Returns an array of 0s and 1s corresponding to the predicted keys
Threshold should be between 0 and 1
'''
def predict_keys(predictions, threshold=0.5):
    if np.max(predictions) > 1: # normalize for faster inference
        return tf.math.floor((predictions/np.max(predictions)) + 1.0 - (threshold/np.max(predictions)))
    else:        
        return tf.math.floor(predictions+1.0-threshold)

'''
Input predictions and labels
Returns the confusion matrix of key pressing being predicted correctly 
'''
def get_confusion_matrix(predictions, groud_truth, threshold=0.5):
    return tf.math.confusion_matrix(predict_keys(tf.reshape(predictions,[-1]), threshold=threshold), tf.reshape(groud_truth, [-1]))


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
    print("Area under curve:", auc(fpr, tpr))
    plt.show()


'''
Input: One data label
Output: Indices of keys played
'''
def get_keys(label):
    keys = []
    for i in range(label.shape[0]):
        if label[i] == 1:
            keys.append(i)
    return keys


def plot_label_hist(labels):
    hist = labels[0]
    for i in labels[0:]:
        hist += i
    plt.hist(range(0, 88), weights=hist/np.sum(hist), bins=88)
    plt.xlabel('Key')
    plt.ylabel('Frequency')
    plt.title('Relative Key Frequencies')
    plt.show()


def load_models(model_dir, model_type):
    models = {}
    for model_file in os.listdir(model_dir):
        if model_file.find(model_type+'.h5') >= 0:  # check if model has type
            models[model_file] = tf.keras.models.load_model(model_dir + '/' + model_file)

    return models
