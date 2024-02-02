# This takes no arguments on the command line. It
# loads the trained model(s), dataset splits,
# and does the evaluation.
#
# output is the open-world performance

from tensorflow import keras
import pickle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc, mean_squared_error
from sklearn.calibration import calibration_curve
import numpy
import matplotlib.pyplot as plt

pr_curve_data = {}

def get_pr_curve_val(model, representation, protocol):
    with open(representation + '_open_world_' + protocol + '_splits.pkl.original', 'rb') as handle:
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

# This gives us the threshold where the F1 is at the
# maximum over the validation set
def get_threshold_f1(precisions, recalls, thresholds):
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    max_f1_index = numpy.argmax(f1_scores)
    threshold_f1 = thresholds[max_f1_index]
    print('max F1 score on the validation set and threshold are', f1_scores[max_f1_index], threshold_f1)
    return threshold_f1

# This was my original idea to tune for zero false positives on the validation set
def get_threshold_zero_fp(model, representation, protocol):
    with open(representation + '_open_world_' + protocol + '_splits.pkl.original', 'rb') as handle:
        splits = pickle.load(handle)
    preds = model.predict(splits['x_val'], verbose = 2)
    preds_labels = numpy.argmax(preds, axis = 1)
    scores = []
    for i in range(len(preds)):
        # Find the max softmax probability for any monitored
        # class. This will be fairly low if the argmax was 60 or
        # if the model was torn between two monitored classes.
        # We're implying the the probability of 60 is 1.0 - this.
        scores.append(max(preds[i][:60]))
    true_labels = numpy.argmax(splits['y_val'], axis = 1)
    true_binary = (true_labels < 60)

    max_false_positive_prob = 0
    max_false_positive_visit = ''
    max_false_positive_label = 0
    for i in range(len(preds)):
        if true_binary[i] == False:
            if scores[i] > max_false_positive_prob:
                max_false_positive_prob = scores[i]
                max_false_positive_visit = splits['visits_val'].iloc[i]
                max_false_positive_label = preds_labels[i]
                print('new highest probability for a validation set false positive:', str(max_false_positive_prob))
    # the visit lets us trace this all the way back to the original dataset to see the raw .pcap and screenshots
    print('visit and predicted label:', max_false_positive_visit, max_false_positive_label)
    return max_false_positive_prob

# This gives us the threshold where recall nearest to 0.9
#
# in the HTTPS case, this sacrifices some recall for a
# better P and FPR compared to tuning for the max F1 score
#
# in the Tor case this guards against the problem
# where recall falls off a cliff due to one
# exceptionally high confidence false positive
def get_threshold_90_recall(precisions, recalls, thresholds):
    closest_recall_index = numpy.argmin(numpy.abs(recalls - 0.9))
    if closest_recall_index == len(thresholds):
        threshold_90_recall = thresholds[-1]
    else:
        threshold_90_recall = thresholds[closest_recall_index]
    print('aiming for a recall of 0.9 on the validation set: recall, precision, and threshold are', recalls[closest_recall_index], precisions[closest_recall_index], threshold_90_recall)
    return threshold_90_recall

def evaluate_model(model, representation, protocol, t_f1, t_zero, t_90):
    with open(representation + '_open_world_' + protocol + '_splits.pkl.original', 'rb') as handle:
        splits = pickle.load(handle)
    for size in ['1000', '2000', '4000', '8000', '16000', '32000', '64000']:
        print(representation, protocol, size)
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
        pr_curve_data[size[:-3] + 'k'] = (precisions, recalls, thresholds)

        # adjust True / False predictions based on the given threshold
        for threshold in [t_zero, t_90]:
            preds_binary = []
            for i in range(len(scores)):
                # lookup the probability for the predicted class for
                # this instance, and set the label to False if it's
                # less than or equal to the highest probability that
                # we found for a false positive in the validation set
                if scores[i] >= threshold:
                    preds_binary.append(True)
                else:
                    preds_binary.append(False)
            print(representation, protocol, size, 'threshold of', str(threshold))
            conf_matrix = confusion_matrix(true_binary, preds_binary)
            FP = conf_matrix.sum(axis=0) - numpy.diag(conf_matrix)  
            FN = conf_matrix.sum(axis=1) - numpy.diag(conf_matrix)
            TP = numpy.diag(conf_matrix)
            TN = conf_matrix.sum() - (FP + FN + TP)
            P = TP/(TP+FP)
            R = TP/(TP+FN)
            F1 = TP / (TP + ((FN + FP) / 2))
            FPR = FP/(FP+TN)
            print('Precision:', str(P), 'Recall:', str(R), 'F1:', str(F1), 'FPR:', str(FPR), 'False Positives:', str(FP))
            if size == '64000':
                false_positives = set()
                for i in range(len(preds)):
                    if scores[i] >= threshold and true_binary[i] == False:
                        false_positives.add((splits['visits_test_64000'].iloc[i], preds_labels[i]))
                print('false positive visits and predicted labels:', false_positives)
                fp_rates, tp_rates, roc_thresholds = roc_curve(true_binary, scores)
                pr_auc = auc(recalls, precisions)
                print('PR AUC', pr_auc)
                roc_auc = auc(fp_rates, tp_rates)
                print('AUROC', roc_auc)
               
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
                protocol_string = 'HTTPS' if protocol == 'https' else 'Tor'
                plt.title('Calibration Curve (Baseline Model - ' + protocol_string + ')', fontsize = 32)
                plt.legend(fontsize = 20)
                plt.savefig('baseline_cal_curve_' + protocol + '.png', dpi=300)

#for representation in ['dschuster16', 'schuster16', 'dschuster8', 'schuster8', 'schuster4', 'schuster2', 'hayden', 'rahman', 'sirinam_vf', 'sirinam_wf']:
for representation in ['dschuster16', 'schuster8']:
    for protocol in ['https', 'tor']:
            try:
                model = keras.models.load_model(representation + '_open_world_' + protocol + '_model.h5.original')
                print('found', representation + '_open_world_' + protocol + '_model.h5.original')
            except:
                print('did not find', representation + '_open_world_' + protocol + '_model.h5.original')
                continue
            print('getting thresholds over the validations set for', representation, protocol)
            precisions, recalls, thresholds = get_pr_curve_val(model, representation, protocol)
            t_f1 = get_threshold_f1(precisions, recalls, thresholds)
            t_zero = get_threshold_zero_fp(model, representation, protocol)
            t_90 = get_threshold_90_recall(precisions, recalls, thresholds)
            evaluate_model(model, representation, protocol, t_f1, t_zero, t_90)

            # create and save the P-R curve figure
            colors = plt.cm.viridis(numpy.linspace(0, 1, len(pr_curve_data)))
            plt.figure(figsize=(16, 12))
            for (label, (precisions, recalls, _)), color in zip(pr_curve_data.items(), colors):
                plt.plot(recalls, precisions, label=label, color=color)
            plt.xlabel('Recall', fontsize = 32)
            plt.ylabel('Precision', fontsize = 32)
            protocol_string = 'HTTPS' if protocol == 'https' else 'Tor'
            plt.title('Precision-Recall Curve (' + protocol_string + ')', fontsize = 32)
            plt.legend(loc = 'lower left', fontsize = 20, title = 'World Size', title_fontsize = 20)
            plt.tick_params(axis='both', which='major', labelsize=20)
            plt.xlim(0.8, 1)
            plt.ylim(0.8, 1)
            plt.grid(True)
            plt.savefig('baseline_pr_curve_' + protocol + '.png', dpi=300)
            pr_curve_data = {}
