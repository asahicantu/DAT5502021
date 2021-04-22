import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve

from Utils import TrainTestValid
from Utils.Models import Accuracy
from Utils.TrainTestValid import split_in_sequences


def predict_keys(predictions, threshold=0.5):
    """
    Input predictions and a threshold value for a key to be pressed
    Returns an array of 0s and 1s corresponding to the predicted keys
    Threshold should be between 0 and 1
    """
    if np.max(predictions) > 1: # normalize for faster inference
        return tf.math.floor((predictions/np.max(predictions)) + 1.0 - (threshold/np.max(predictions)))
    else:        
        return tf.math.floor(predictions+1.0-threshold)


def get_confusion_matrix(predictions, groud_truth, threshold=0.5):
    """ Input predictions and labels
        Returns the confusion matrix of key pressing being predicted correctly """
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


def get_keys(label):
    """
    Input: One data label
    Output: Indices of keys played
    """
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


def load_models(model_dir, model_type= None):
    custom_objects = {
        'AccuracyHistory':Accuracy.AccuracyHistory,
        'root_mse':Accuracy.root_mse,
        'r2_coeff_determination':Accuracy.r2_coeff_determination
    }
    models = {}
    model_files = [x for x in os.listdir(model_dir) if x.endswith('.h5')]
    for model_file in model_files:
        try:
            print(f'loading model {model_file}...')
            if  model_type is None or model_type in  model_file:  # check if model has type
                models[model_file] = tf.keras.models.load_model(model_dir + '/' + model_file,custom_objects=custom_objects)
        except Exception as e:
            print(f'Failed to load model {model_file}, exception message is: {e}')
            
    return models


def is_difference_significant(pred_1, pred_2, threshold1, threshold2, labels, alpha=0.05):
    """null hypothesis: there no statistically significant difference between the mean accuracy of both classifiers
    at the given significance level (default 5%)"""
    acc_1 = []
    acc_2 = []
    for idx, label in enumerate(labels):
        acc_1.append(get_metric(pred_1[idx], label, metric='accuracy', threshold=threshold1))
        acc_2.append(get_metric(pred_2[idx], label, metric='accuracy', threshold=threshold2))

    stat, p = ttest_rel(acc_1, acc_2)
    if p > alpha:
        return False
    else:
        return True


def determine_opt_threshold(predictions, labels):
    """Determine the optimal threshold value according to F1 score"""
    precision, recall, thresholds = precision_recall_curve(tf.reshape(labels, [-1]), tf.reshape(predictions, [-1]))
    best_threshold_index = 0
    best_f1 = 0
    for idx in range(len(thresholds)):
        f1 = 2*recall[idx]*precision[idx]/(recall[idx]+precision[idx])
        if f1 >= best_f1:
            best_f1 = f1
            best_threshold_index = idx
    return thresholds[best_threshold_index]


def get_data_dict(features, labels):
    X_mel, y = TrainTestValid.train_test_validation_split(features['mel'],
                                                          labels)  # note that the splitting uses a fixed seed and the labels are the same
    X_mfcc, _ = TrainTestValid.train_test_validation_split(features['mfcc'], labels)
    X_cqt, _ = TrainTestValid.train_test_validation_split(features['cqt'], labels)

    X_MLP = {}
    X_MLP['mel'] = format_input_dict(X_mel, 'MLP')
    X_MLP['mfcc'] = format_input_dict(X_mfcc, 'MLP')
    X_MLP['cqt'] = format_input_dict(X_cqt, 'MLP')

    X_LSTM = {}
    X_LSTM['mel'] = format_input_dict(X_mel, 'LSTM')
    X_LSTM['mfcc'] = format_input_dict(X_mfcc, 'LSTM')
    X_LSTM['cqt'] = format_input_dict(X_cqt, 'LSTM')

    X_LSTM_M2M = {}
    X_LSTM_M2M['mel'] = format_input_dict(X_mel, 'LSTM_M2M')
    X_LSTM_M2M['mfcc'] = format_input_dict(X_mfcc, 'LSTM_M2M')
    X_LSTM_M2M['cqt'] = format_input_dict(X_cqt, 'LSTM_M2M')

    X = {}
    X['MLP'] = X_MLP
    X['MLP_1H'] = X_MLP
    X['LSTM'] = X_LSTM
    X['LSTM_M2M'] = X_LSTM_M2M

    return X, y


def format_input_dict(x_dict, model_type):
    """Formats input dictionary according to model type specific properties.
    Assumes that dictionary content is numpy array"""
    out = {}
    if model_type == 'LSTM':
        out['train'] = x_dict['train'].swapaxes(1, 2)
        out['test'] = x_dict['test'].swapaxes(1, 2)
        out['valid'] = x_dict['valid'].swapaxes(1, 2)
    elif model_type == 'MLP' or model_type == 'MLP_1H' or model_type == 'LSTM_M2M':
        out['train'] = np.array([x.flatten() for x in x_dict['train']])
        out['test'] = np.array([x.flatten() for x in x_dict['test']])
        out['valid'] = np.array([x.flatten() for x in x_dict['valid']])
        if model_type == 'LSTM_M2M':
            out['train'] = np.array(split_in_sequences(out['train'], 10, keep_short_batch=False))
            out['test'] = np.array(split_in_sequences(out['test'], 10, keep_short_batch=False))
            out['valid'] = np.array(split_in_sequences(out['valid'], 10, keep_short_batch=False))
    return out


def group_labels_LSTM_M2M(label_dict):
    out = {}
    for key in label_dict.keys():
        out[key] = np.array(split_in_sequences(label_dict[key], 10, keep_short_batch=False))
    return out
