# This takes no arguments on the command line. It
# loads the trained model(s), dataset splits,
# and does the evaluation.
#
# output is the open-world performance

import tensorflow
from tensorflow import keras
import pickle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc, mean_squared_error
from sklearn.calibration import calibration_curve
import numpy
import matplotlib.pyplot as plt

pr_curve_data = {}

class TemperatureScalingLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TemperatureScalingLayer, self).__init__(**kwargs)
        self.temperature = tensorflow.Variable(initial_value = 2.0, trainable = True, dtype=tensorflow.float32)

    def call(self, inputs):
        return inputs / self.temperature

def get_pr_curve_val(model, representation, protocol):
    with open('../3_open_world_baseline/' + representation + '_open_world_' + protocol + '_splits.pkl', 'rb') as handle:
        splits = pickle.load(handle)
    preds = model.predict(splits['x_val'], verbose = 2)
    scores = []
    for i in range(len(preds)):
        # Find the max softmax probability for any monitored
        # class. This will be fairly low if the argmax was 60 or
        # if the model was torn between two monitored classes.
        # We're implying the the probability of 60 is 1.0 - this.
        scores.append(max(preds[i][:60]))
    true_labels = numpy.argmax(splits['y_val'], axis = 1)
    true_binary = (true_labels < 60)
    return precision_recall_curve(true_binary, scores)

def evaluate_model(model, representation, protocol, approach):
    with open('../3_open_world_baseline/' + representation + '_open_world_' + protocol + '_splits.pkl', 'rb') as handle:
        splits = pickle.load(handle)
    for size in ['64000']:
        print(representation, protocol, approach)
        preds = model.predict(splits['x_test_' + size], verbose = 2)
        preds_labels = numpy.argmax(preds, axis = 1)
        scores = []
        for i in range(len(preds)):
            # Find the max softmax probability for any monitored
            # class. This will be fairly low if the argmax was 60 or
            # if the model was torn between two monitored classes.
            # We're implying the the probability of 60 is 1.0 - this.
            scores.append(max(preds[i][:60]))
        true_labels = numpy.argmax(splits['y_test_' + size], axis = 1)
        true_binary = (true_labels < 60)
        precisions, recalls, thresholds = precision_recall_curve(true_binary, scores)
        # add these to a global data structure to plot later
        pr_curve_data[approach] = (precisions, recalls, thresholds)

        pr_auc = auc(recalls, precisions)
        print('PR AUC', pr_auc)
               
        # plot and save a calibration curve figure
        prob_true, prob_pred = calibration_curve(true_binary, scores, n_bins=20, strategy='uniform')
        bin_edges = numpy.linspace(0, 1, 21)
        bin_width = numpy.diff(bin_edges)
        bin_centers = bin_edges[:-1] + bin_width / 2
        bin_counts = numpy.histogram(scores, bins=bin_edges)[0]
        plt.figure(figsize=(16, 12))
        plt.bar(bin_centers, prob_true, width = bin_width, align = 'center', alpha = 0.5, edgecolor='b', label='Calibration Curve')
        for i, count in enumerate(bin_counts):
            plt.text(bin_centers[i], prob_true[i], f' {count}', verticalalignment= 'bottom', horizontalalignment = 'center')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        plt.xlabel('Predicted Probability of Monitored and Number of Instances', fontsize = 32)
        plt.ylabel('True Monitored Frequency', fontsize = 32)
        plt.title('Calibration Curve (' + approach + ' - ' + protocol + ')', fontsize = 32)
        plt.legend(fontsize = 20)
        plt.savefig(approach + '_cal_curve_' + protocol + '.png', dpi=300)

for protocol in ['https', 'tor']:
    for approach in ['baseline', 'temp_scaling']: # I'll add more approaches to this list as I build them
        for representation in ['dschuster16', 'schuster8']:
            try:
                model = keras.models.load_model(representation + '_' + protocol + '_' + approach + '_model.h5',
                                                custom_objects = {'TemperatureScalingLayer': TemperatureScalingLayer})
                print('found ' + representation + '_' + protocol + '_' + approach + '_model.h5')
            except Exception as e:
                print(e)
                continue
            evaluate_model(model, representation, protocol, approach)

    # create and save the P-R curve figure
    colors = plt.cm.viridis(numpy.linspace(0, 1, len(pr_curve_data)))
    plt.figure(figsize=(16, 12))
    for (label, (precisions, recalls, _)), color in zip(pr_curve_data.items(), colors):
        plt.plot(recalls, precisions, label=label, color=color)
    plt.xlabel('Recall', fontsize = 32)
    plt.ylabel('Precision', fontsize = 32)
    protocol_string = 'HTTPS' if protocol == 'https' else 'Tor'
    plt.title('Precision-Recall Curve (' + protocol_string + ' 64k Test Set)', fontsize = 32)
    plt.legend(loc = 'lower left', fontsize = 20, title = 'Approach', title_fontsize = 20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.xlim(0.5, 1)
    plt.ylim(0.5, 1)
    plt.grid(True)
    plt.savefig('enhanced_pr_curve_' + protocol + '.png', dpi=300)
    pr_curve_data = {}
