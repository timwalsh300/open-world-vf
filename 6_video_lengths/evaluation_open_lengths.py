# This takes one argument for the protocol,
# either 'https' or 'tor', and another for
# the specified traffic flow length in seconds.
# It loads the trained model(s), dataset splits,
# and does the evaluation.
#
# Outputs are the open-world performance as the
# average precision-recall curve and various
# metrics over 20 trials on the 64k test set
# with traffic flows truncated to the specified length.

import torch
import mymodels_torch
import numpy
from sklearn.metrics import precision_recall_curve, auc
from sklearn.calibration import calibration_curve
from scipy.interpolate import interp1d
import pickle
import sys

protocol = sys.argv[1]

length = int(sys.argv[2])

INPUT_SHAPES = {'schuster8': (1, length * 8),
                'dschuster16': (2, length * 16)}

BASELINE_HYPERPARAMETERS = {'schuster8_tor': {'filters': 256, 'kernel': 8, 'conv_stride': 1, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 7.191906601911815e-05, 'batch_size': 128},
                            'dschuster16_https': {'filters': 256, 'kernel': 4, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.4, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.8, 'lr': 0.0005153393428807454, 'batch_size': 64}}

common_recall_levels = numpy.linspace(0, 1, 500)

# this stores the Bayesian model average after
# doing inference with mc_samples, so that we can
# use the same Bayesian model average to report
# either MSP or uncertainty (entropy) as the decision
# function or score
sscd_preds_avg = {}

def get_scores(test_loader, protocol, representation, approach, trial):
    if (approach == 'baseline'):
        model = mymodels_torch.DFNetTunable(INPUT_SHAPES[representation],
                                            61,
                                            BASELINE_HYPERPARAMETERS[representation + '_' + protocol])
        model.load_state_dict(torch.load('models/' + representation + '_' + protocol + '_' + approach + '_model_' + str(length) + '_' + str(trial) + '.pt'))
        model.to(device)
        model.eval()
        logits_batches = []
        for x_test, y_test in test_loader:
            with torch.no_grad():
                logits_batches.append(model(x_test.to(device), training = False).to('cpu'))
        logits_concatenated = torch.cat(logits_batches, dim = 0)
        preds = torch.softmax(logits_concatenated, dim=1).detach().numpy()
        scores = []
        for i in range(len(preds)):
            # Find the max softmax probability for any monitored
            # class. This will be fairly low if the argmax was 60 or
            # if the model was torn between two monitored classes.
            # We're implying the the probability of 60 is 1.0 - this.
            scores.append(max(preds[i][:60]))
        if trial == 0:
            try:
                # print('ECE', check_calibration(scores, test_loader, approach, protocol))
                pass
            except:
                pass
        return preds, scores

    # this loads a trained Spike and Slab Dropout model with Concrete Dropout layers before
    # using maximum softmax probability to rank predictions when they are for any monitored class
    #
    # we combine this with the Standard Model (Background Class), Standard Model + Mixup,
    # and monitored only + NOTA methods of training data augmentation
    elif (approach == 'sscd' or
         approach == 'sscd_mixup' or
         approach == 'sscd_nota' or
         approach == 'sscd_mixup_nota'):
        model = mymodels_torch.DFNetTunableSSCD(INPUT_SHAPES[representation],
                                                61,
                                                BASELINE_HYPERPARAMETERS[representation + '_' + protocol])
        model.load_state_dict(torch.load('models/' + representation + '_' + protocol + '_' + approach + '_model_' + str(length) + '_' + str(trial) + '.pt'))
        preds_avg, scores = get_bayesian_msp(test_loader, model, 61)
        if trial == 0:
            try:
                # print('ECE', check_calibration(scores, test_loader, approach, protocol))
                pass
            except:
                pass
        sscd_preds_avg[approach + str(len(test_loader)) + str(trial)] = preds_avg
        return preds_avg, scores

def get_bayesian_msp(test_loader, model, classes):
        model.to(device)
        model.eval()
        mc_samples = 10
        preds = numpy.zeros([mc_samples, len(test_loader.dataset), classes])
        for i in range(mc_samples):
            logits_batches = []
            for x_test, y_test in test_loader:
                with torch.no_grad():
                    logits_batches.append(model(x_test.to(device), training = True).to('cpu'))
            logits_concatenated = torch.cat(logits_batches, dim = 0)
            preds[i] = torch.softmax(logits_concatenated, dim=1).detach().numpy()
        preds_avg = preds.mean(axis = 0)
        scores = []
        for i in range(len(preds_avg)):
            scores.append(max(preds_avg[i][:60]))
        return preds_avg, scores

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for representation in ['dschuster16', 'schuster8']:
    #for protocol in ['https', 'tor']:
        try:
            val_tensors = torch.load('../4_open_world_enhancements/' + representation + '_' + protocol + '_val_tensors.pt')
            # truncate the lengths down to the specified number of time steps
            val_tensors = (val_tensors[0][:, :, :INPUT_SHAPES[representation][1]], val_tensors[1])
            val_dataset = torch.utils.data.TensorDataset(*val_tensors)
            val_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=128,
                                                     shuffle=False)
            y_val_batches = []
            for _, y_val in val_loader:
                y_val_batches.append(y_val)
            y_val_concatenated = torch.cat(y_val_batches, dim = 0)
            y_val_np = y_val_concatenated.numpy()
            true_labels_val = numpy.argmax(y_val_np, axis = 1)
            true_binary_val = (true_labels_val < 60)

            test_tensors = torch.load('../4_open_world_enhancements/' + representation + '_' + protocol + '_test_tensors.pt')
            # truncate the lengths down to the specified number of time steps
            test_tensors = (test_tensors[0][:, :, :INPUT_SHAPES[representation][1]], test_tensors[1])
            test_dataset = torch.utils.data.TensorDataset(*test_tensors)
            test_loader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=128,
                                                      shuffle=False)
            y_test_batches = []
            for _, y_test in test_loader:
                y_test_batches.append(y_test)
            y_test_concatenated = torch.cat(y_test_batches, dim = 0)
            y_test_np = y_test_concatenated.numpy()
            true_labels = numpy.argmax(y_test_np, axis = 1)
            true_binary = (true_labels < 60)
            with open('../3_open_world_baseline/' + representation + '_open_world_' + protocol + '_splits.pkl', 'rb') as handle:
                splits = pickle.load(handle)
        except Exception as e:
            print(e)
            continue

        for approach in ['baseline', 'sscd_mixup']:
            trial_scores = []
            trial_best_case_recalls = []
            trial_t_50_FPRs = []
            trial_t_75_FPRs = []
            trial_accuracies = []
            for trial in range(20):
                print('Getting scores for', protocol, approach, length, '... Trial', trial)
                
                # find t_50 on the validation set for this model
                preds_val, scores_val = get_scores(val_loader, protocol, representation, approach, trial)
                precisions_val, recalls_val, thresholds_val = precision_recall_curve(true_binary_val, scores_val)
                t_50_index = numpy.argmin(numpy.abs(recalls_val - 0.5))
                if t_50_index == len(thresholds_val):
                    t_50 = thresholds_val[-1]
                else:
                    t_50 = thresholds_val[t_50_index]
                print('... t_50 validation recall, precision, and threshold are', recalls_val[t_50_index], precisions_val[t_50_index], t_50)
                
                # find t_75 on the validation set for this model
                t_75_index = numpy.argmin(numpy.abs(recalls_val - 0.75))
                if t_75_index == len(thresholds_val):
                    t_75 = thresholds_val[-1]
                else:
                    t_75 = thresholds_val[t_75_index]
                print('... t_75 validation recall, precision, and threshold are', recalls_val[t_75_index], precisions_val[t_75_index], t_75)
                
                # run the model against the test set
                preds, scores = get_scores(test_loader, protocol, representation, approach, trial)
                trial_scores.append(scores)
                precisions, recalls, thresholds = precision_recall_curve(true_binary, scores)
                best_case_recall = recalls[precisions >= 1.0].max()
                print('... Recall at precision of 1.0:', best_case_recall)
                trial_best_case_recalls.append(best_case_recall)
                preds_labels = numpy.argmax(preds, axis = 1)
                total_within_monitored = 0
                correct_within_monitored = 0
                for i in range(len(preds_labels)):
                    if preds_labels[i] < 60 and true_labels[i] < 60:
                        total_within_monitored += 1
                        if preds_labels[i] == true_labels[i]:
                            correct_within_monitored += 1
                print('... Accuracy within monitored:', correct_within_monitored / total_within_monitored)
                trial_accuracies.append(correct_within_monitored / total_within_monitored)
                preds_binary = []
                for i in range(len(scores)):
                    if scores[i] >= t_50:
                        preds_binary.append(True)
                    else:
                        preds_binary.append(False)
                FP_visits = []
                for i in range(len(preds_binary)):
                    if preds_binary[i] and not true_binary[i]:
                       FP_visits.append(splits['visits_test_64000'].iloc[i])
                trial_t_50_FPRs.append(len(FP_visits) / 64000)
                print('... False positives at t_50:', len(FP_visits))
                if len(FP_visits) <= 10:
                    print(FP_visits)
                    
                preds_binary = []
                for i in range(len(scores)):
                    if scores[i] >= t_75:
                        preds_binary.append(True)
                    else:
                        preds_binary.append(False)
                FP_visits = []
                for i in range(len(preds_binary)):
                    if preds_binary[i] and not true_binary[i]:
                       FP_visits.append(splits['visits_test_64000'].iloc[i])
                trial_t_75_FPRs.append(len(FP_visits) / 64000)
                print('... False positives at t_75:', len(FP_visits))
                if len(FP_visits) <= 10:
                    print(FP_visits)

            print('Mean recall at precision of 1.0:', numpy.mean(trial_best_case_recalls), 'StdDev: ', numpy.std(trial_best_case_recalls))
            print('Mean accuracy within monitored:', numpy.mean(trial_accuracies), 'StdDev: ', numpy.std(trial_accuracies))
            print('Mean FPR at t_50:', numpy.mean(trial_t_50_FPRs), 'StdDev: ', numpy.std(trial_t_50_FPRs))
            print('Mean FPR at t_75:', numpy.mean(trial_t_75_FPRs), 'StdDev: ', numpy.std(trial_t_75_FPRs))
            interpolated_precisions = []
            for trial_score in trial_scores:
                trial_precision, trial_recall, _ = precision_recall_curve(true_binary, trial_score)
                interpolate_precision = interp1d(trial_recall, trial_precision, bounds_error=False, fill_value=(trial_precision[0], 0))
                interpolated_precision = interpolate_precision(common_recall_levels)
                interpolated_precisions.append(interpolated_precision)
            mean_precisions = numpy.mean(interpolated_precisions, axis=0)
            pr_auc = auc(common_recall_levels, mean_precisions)
            print('Average PR-AUC:', pr_auc)
            print('-------------------------\n')
