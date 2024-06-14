# This takes no arguments on the command line. It
# loads the trained model(s), dataset splits,
# and does the evaluation.
#
# Outputs are the open-world performance as the
# average precision-recall curve and various
# metrics over ten trials on the 64k test set

import torch
import mymodels_torch
import numpy
from sklearn.metrics import precision_recall_curve, auc
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pickle

INPUT_SHAPES = {'schuster8': (1, 1920),
                'dschuster16': (2, 3840)}

# we manually copy and paste these hyperparameters from the output of search_open.py
BASELINE_HYPERPARAMETERS = {'schuster8_tor': {'filters': 256, 'kernel': 8, 'conv_stride': 1, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 7.191906601911815e-05, 'batch_size': 128},
                            'dschuster16_https': {'filters': 256, 'kernel': 4, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.4, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.8, 'lr': 0.0005153393428807454, 'batch_size': 64}}

pr_curve_data = {}
    
common_recall_levels = numpy.linspace(0, 1, 500)

# this plots and saves a reliability diagram and then
# returns the calculated expected calibration error
def check_calibration(scores, test_loader, approach, protocol):
    y_test_batches = []
    for _, y_test in test_loader:
        y_test_batches.append(y_test)
    y_test_concatenated = torch.cat(y_test_batches, dim = 0)
    y_test_np = y_test_concatenated.numpy()
    true_labels = numpy.argmax(y_test_np, axis = 1)
    true_binary = (true_labels < 60)
    prob_true, prob_pred = calibration_curve(true_binary,
                                             scores,
                                             n_bins = 10,
                                             strategy = 'uniform')

    # Plot and save a reliability diagram
    bin_edges = numpy.linspace(0, 1, 11)
    bin_width = numpy.diff(bin_edges)
    bin_centers = bin_edges[:-1] + bin_width / 2
    bin_counts = numpy.histogram(scores, bins=bin_edges)[0]
    plt.figure(figsize=(16, 12))
    plt.bar(bin_centers, prob_true, width = bin_width, align = 'center', alpha = 0.5, edgecolor='b', label=approach)
    for i, count in enumerate(bin_counts):
        plt.text(bin_centers[i], prob_true[i], f' {count}', verticalalignment= 'bottom', horizontalalignment = 'center')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Predicted Probability Bins and Number of Instances', fontsize = 32)
    plt.ylabel('True Positive Frequency', fontsize = 32)
    protocol_string = 'HTTPS' if protocol == 'https' else 'Tor'
    plt.title('Reliability Diagram (' + protocol_string + ' 64k Test Set)', fontsize = 32)
    plt.legend(fontsize = 20)
    plt.savefig('reliability_' + protocol + '_' + approach + '.png', dpi=300)

    # Calculate the absolute difference between prob_true and prob_pred
    abs_diff = numpy.abs(prob_true - prob_pred)
    # Calculate the weights for each bin as the proportion of samples in each bin
    weights = bin_counts / numpy.sum(bin_counts)
    # Calculate the ECE as the weighted average of the absolute differences
    return numpy.sum(weights * abs_diff)

def get_scores(test_loader, protocol, representation, approach, trial):
    # this is the MLE model that uses Max Softmax Probability or
    # Softmax Thresholding for OSR
    #
    # we combine this with the Standard Model (Background Class), Standard Model + Mixup,
    # and monitored only + NOTA methods of training data augmentation
    if (approach == 'baseline' or
       approach == 'baseline_mixup' or
       approach == 'baseline_nota'):
        model = mymodels_torch.DFNetTunable(INPUT_SHAPES[representation],
                                            61,
                                            BASELINE_HYPERPARAMETERS[representation + '_' + protocol])
        model.load_state_dict(torch.load(representation + '_' + protocol + '_' + approach + '_model' + str(trial) + '.pt'))
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
        #print('ECE', check_calibration(scores, test_loader, approach, protocol))
        return preds, scores

    # this is just a 60-way classification model relying only on MSP
    # or Softmax Thresholding for OSR, with no unmonitored training data
    # or output class, using a deterministic model
    if (approach == 'baseline_monitored'):
        model = mymodels_torch.DFNetTunable(INPUT_SHAPES[representation],
                                            60,
                                            BASELINE_HYPERPARAMETERS[representation + '_' + protocol])
        model.load_state_dict(torch.load(representation + '_' + protocol + '_' + approach + '_model' + str(trial) + '.pt'))
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
        #print('ECE', check_calibration(scores, test_loader, approach, protocol))
        return preds, scores

    # this loads a trained Spike and Slab Dropout model with Concrete Dropout layers before
    # using maximum softmax probability to rank predictions when they are for any monitored class
    #
    # we combine this with the Standard Model (Background Class), Standard Model + Mixup,
    # and monitored only + NOTA methods of training data augmentation
    elif (approach == 'sscd' or
         approach == 'sscd_mixup' or
         approach == 'sscd_nota'):
        model = mymodels_torch.DFNetTunableSSCD(INPUT_SHAPES[representation],
                                            61,
                                            BASELINE_HYPERPARAMETERS[representation + '_' + protocol])
        model.load_state_dict(torch.load(representation + '_' + protocol + '_' + approach + '_model' + str(trial) + '.pt'))
        preds, scores = get_bayesian_msp(test_loader, model, 61)
        #print('ECE', check_calibration(scores, test_loader, approach, protocol))
        return preds, scores

    # this loads a trained Spike and Slab Dropout model with Concrete Dropout layers before
    # using epistemic uncertainty to rank predictions when they are for any monitored class
    #
    # the intuition is that the epistemic uncertainty should be higher when we have
    # a false positive than when we have a true positive
    #
    # ranking both monitored and unmonitored predictions by epistemic uncertainty is
    # problematic because uncertainty will be low on both ends, so low epistemic
    # uncertainty doesn't mean that an instance is obviously monitored; it could be
    # obviously unmonitored too, so we only rank monitored predictions
    elif (approach == 'sscd_epistemic' or
         approach == 'sscd_mixup_epistemic' or
         approach == 'sscd_nota_epistemic'):
        model = mymodels_torch.DFNetTunableSSCD(INPUT_SHAPES[representation],
                                            61,
                                            BASELINE_HYPERPARAMETERS[representation + '_' + protocol])
        model.load_state_dict(torch.load(representation + '_' + protocol + '_' + approach[:-10] + '_model' + str(trial) + '.pt'))
        return get_epistemic_uncertainty(test_loader, model)

    # this is just a 60-way classification model relying only on MSP
    # or Softmax Thresholding for OSR, with no unmonitored training data
    # or output class, using a trained Spike and Slab Dropout model with
    # Concrete Dropout and mixup data augmentation
    if (approach == 'sscd_mixup_monitored'):
        model = mymodels_torch.DFNetTunableSSCD(INPUT_SHAPES[representation],
                                            60,
                                            BASELINE_HYPERPARAMETERS[representation + '_' + protocol])
        model.load_state_dict(torch.load(representation + '_' + protocol + '_' + approach + '_model' + str(trial) + '.pt'))
        preds, scores = get_bayesian_msp(test_loader, model, 60)
        #print('ECE', check_calibration(scores, test_loader, approach, protocol))
        return preds, scores

    # this loads a trained baseline model and OpenGAN discriminator and returns
    # the predictions and scores from the discriminator
    elif (approach == 'opengan'):
        model = mymodels_torch.DFNetTunable(INPUT_SHAPES[representation],
                                            61,
                                            BASELINE_HYPERPARAMETERS[representation + '_' + protocol])
        model.load_state_dict(torch.load(representation + '_' + protocol + '_baseline_model' + str(trial) + '.pt'))
        model.to(device)
        model.eval()
        discriminator = mymodels_torch.DiscriminatorDFNet_fea(INPUT_SHAPES[representation],
                                                              61,
                                                              BASELINE_HYPERPARAMETERS[representation + '_' + protocol])
        discriminator.load_state_dict(torch.load(representation + '_' + protocol + '_opengan_model' + str(trial) + '.pt'))
        discriminator.to(device)
        discriminator.eval()
        logits_batches = []
        for x_test, y_test in test_loader:
            with torch.no_grad():
                features = model.extract_features_flattened(x_test.to(device))
                logits_batches.append(discriminator(features, training = False).to('cpu'))
        logits_concatenated = torch.cat(logits_batches, dim = 0)
        preds = torch.softmax(logits_concatenated, dim=1).detach().numpy()
        scores = []
        for i in range(len(preds)):
            scores.append(max(preds[i][:60]))
        return preds, scores

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

def get_epistemic_uncertainty(test_loader, model):
        model.to(device)
        model.eval()
        mc_samples = 10
        preds = numpy.zeros([mc_samples, len(test_loader.dataset), 61])
        entropies = numpy.zeros([mc_samples, len(test_loader.dataset)])
        for i in range(mc_samples):
            logits_batches = []
            for x_test, y_test in test_loader:
                with torch.no_grad():
                    logits_batches.append(model(x_test.to(device), training = True).to('cpu'))
            logits_concatenated = torch.cat(logits_batches, dim = 0)
            preds[i] = torch.softmax(logits_concatenated, dim=1).detach().numpy()
            entropies[i] = -1 * numpy.sum(preds[i] * numpy.log(preds[i] + 1e-9), axis = 1)
        preds_avg = preds.mean(axis = 0)
        # compute the total uncertainty for each test instance
        total_uncertainties = -1 * numpy.sum(preds_avg * numpy.log(preds_avg + 1e-9), axis = 1)
        # compute aleatoric uncertainty for each test instance
        aleatoric_uncertainties = numpy.mean(entropies, axis = 0)
        # compute the epistemic uncertainty for each test instance
        epistemic_uncertainties = total_uncertainties - aleatoric_uncertainties
        # we negate the epistemic uncertainties because precision_recall_curve() expects
        # a higher score to be associated with a higher confidence prediction for the
        # positive class, but a high epistemic uncertainty is a low confidence prediction
        scores = [-eu for eu in epistemic_uncertainties]
        # now the min of scores is the highest uncertainty
        highest_eu = min(scores)
        for i in range(len(test_loader.dataset)):
            # if the prediction was unmonitored, we rank it as low as possible without
            # even considering the uncertainty
            if numpy.argmax(preds_avg[i]) == 60:
                scores[i] = highest_eu
        #print('... lowest epistemic uncertainty for a positive prediction', max(scores), numpy.argmax(scores))
        #print('... highest epistemic uncertainty for a positive prediction', min(scores), numpy.argmin(scores))
        return preds_avg, scores

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for protocol in ['https', 'tor']:
    for representation in ['dschuster16', 'schuster8']:
        try:
            val_tensors = torch.load(representation + '_' + protocol + '_val_tensors.pt')
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

            test_tensors = torch.load(representation + '_' + protocol + '_test_tensors.pt')
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
            continue

        #with open('pr_curve_data_' + protocol + '.pkl', 'rb') as handle:
        #    pr_curve_data = pickle.load(handle)
        # These approaches are the top competitors with the baseline
        for approach in ['baseline', 'baseline_mixup', 'baseline_nota', 'opengan',
                         'sscd', 'sscd_epistemic', 'sscd_mixup', 'sscd_mixup_epistemic', 'sscd_nota', 'sscd_nota_epistemic']:
        # These approaches are the competitors with monitored-only deterministic MSP
        #for approach in ['baseline_monitored', 'opengan']:
            trial_scores = []
            trial_best_case_recalls = []
            trial_t_50_FPRs = []
            trial_t_75_FPRs = []
            trial_accuracies = []
            for trial in range(10):
                print('Getting scores for', protocol, approach, '... Trial', trial)
                
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
            # add these to a global data structure to plot later
            pr_curve_data[approach] = mean_precisions
            pr_auc = auc(common_recall_levels, mean_precisions)
            print('Average PR-AUC:', pr_auc)
            print('-------------------------\n')
        
        # save this for future runs, so we don't have to do
        # ten trials for every approach every time we add
        # a new approach
        with open('pr_curve_data_' + protocol + '.pkl', 'wb') as handle:
            pickle.dump(pr_curve_data, handle)

        # create and save the P-R curve figure for top competitors with the baseline
        # determ  msp-std               msp-std-mix              msp-std-nota               opengan
        # bayes   msp-std    epi-std    msp-std-mix  epi-std-mix msp-std-nota  epi-std-nota            
        colors = ['#000000',            '#ff0000',               '#00cc00',                 '#ff9933',
                  '#000000', '#000000', '#ff0000',   '#ff0000',  '#00cc00',    '#00cc00']
        lines =  ['-',                  '-',                     '-',                       '-',
                  '-.',      ':',       '-.',        ':',        '-.',         ':']
        # create and save the P-R curve figure for competitors with monitored-only MSP
        # determ  msp-mon    opengan-mon        
        #colors = ['#000000', '#ff9933']
        #lines =  ['-',       '-']
        num_styles = len(lines)
        num_colors = len(colors)
        plt.figure(figsize=(16, 12))
        for i, (label, mean_precisions) in enumerate(pr_curve_data.items()):
            color = colors[i % num_colors]
            line_style = lines[i % num_styles]
            plt.plot(common_recall_levels, mean_precisions, label=label, color=color, linestyle=line_style)
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
        #plt.savefig('enhanced_pr_curve_' + protocol + '_monitored.png', dpi=300)
        pr_curve_data = {}
        print('-------------------------\n')
