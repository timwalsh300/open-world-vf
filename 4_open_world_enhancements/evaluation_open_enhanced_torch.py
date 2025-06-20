# This takes argument for protocol to evaluate. It
# loads the trained model(s), dataset splits,
# and does the evaluation.
#
# Outputs are the open-world performance as the
# average precision-recall curve and various
# metrics over 20 trials on the 64k test set

import torch
import mymodels_torch
import numpy
from sklearn.metrics import precision_recall_curve, auc
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import pickle
import sys

INPUT_SHAPES = {'schuster8': (1, 1920),
                'dschuster16': (2, 3840)}

# we manually copy and paste these hyperparameters from the output of search_open.py
BASELINE_HYPERPARAMETERS = {'schuster8_tor': {'filters': 256, 'kernel': 8, 'conv_stride': 1, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 7.191906601911815e-05, 'batch_size': 128},
                            'dschuster16_https': {'filters': 256, 'kernel': 4, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.4, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.8, 'lr': 0.0005153393428807454, 'batch_size': 64}}

pr_curve_data = {}

common_recall_levels = numpy.linspace(0, 1, 500)

PLOT_MSP_DIST = False
msp_dist_data = {}

PLOT_CSSR_DIST = False

# this stores the Bayesian model average after
# doing inference with mc_samples, so that we can
# use the same Bayesian model average to report
# either MSP or uncertainty (entropy) as the decision
# function or score
sscd_preds_avg = {}

# for the CSSR approach, this creates and saves a bar chart for each
# autoencorder's mean, normalized reconstruction error for monitored
# and unmonitored instances
#
# it also creates and saves a scatter plot showing the individual scores
# for each instance, per class
def plot_classwise_scores(preds, scores, true_binary_labels, num_classes, protocol):
    positive_scores_by_class = [[] for _ in range(num_classes)]
    negative_scores_by_class = [[] for _ in range(num_classes)]
    predicted_classes = numpy.argmax(preds, axis=1)
    
    for i, predicted_class in enumerate(predicted_classes):
        if true_binary_labels[i]:  # monitored (positive) instance
            positive_scores_by_class[predicted_class].append(-scores[i])
        else:  # unmonitored (negative) instance
            negative_scores_by_class[predicted_class].append(-scores[i])

    mean_positive_scores = [numpy.mean(scores) if scores else 0 for scores in positive_scores_by_class]
    mean_negative_scores = [numpy.mean(scores) if scores else 0 for scores in negative_scores_by_class]
    
    x = numpy.arange(num_classes)
    width = 0.35
    protocol_string = 'HTTPS' if protocol == 'https' else 'Tor'
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, mean_positive_scores, width, label='Monitored', color='blue')
    ax.bar(x + width/2, mean_negative_scores, width, label='Unmonitored', color='black')
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Mean Score')
    ax.set_title('Mean Scores by Predicted Class (' + protocol_string + ' Test Set)')
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(x, rotation=90)
    fig.tight_layout()
    plt.savefig('cssr_test_means_' + protocol + '.png', dpi=300)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(num_classes):
        # Plot monitored scores
        if positive_scores_by_class[i]:
            ax.scatter([i - width/2] * len(positive_scores_by_class[i]), positive_scores_by_class[i], 
                       color='blue', s=30, alpha=0.2, label='Monitored' if i == 0 else "")
        
        # Plot unmonitored scores with reduced size/alpha to handle imbalance
        if negative_scores_by_class[i]:
            ax.scatter([i + width/2] * len(negative_scores_by_class[i]), negative_scores_by_class[i], 
                       color='black', s=10, alpha=0.2, label='Unmonitored' if i == 0 else "")
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Score')
    ax.set_title('Scores by Predicted Class (' + protocol_string + ' Test Set)')
    ax.set_xticks(x)
    ax.set_xticklabels(x, rotation=90)
    ax.legend()
    fig.tight_layout()
    plt.savefig('cssr_test_instances_' + protocol + '.png', dpi=300)

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
       approach == 'baseline_nota' or
       approach == 'baseline_mixup_nota'):
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
        if trial == 0:
            try:
                # print('ECE', check_calibration(scores, test_loader, approach, protocol))
                pass
            except:
                pass
        return preds, scores

    # this is just a 60-way classification model relying only on MSP
    # or Softmax Thresholding for OSR, with no unmonitored training data
    # or output class, using a deterministic model
    elif (approach == 'baseline_monitored'):
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
        if trial == 0:
            try:
                # print('ECE', check_calibration(scores, test_loader, approach, protocol))
                pass
            except:
                pass
        return preds, scores

    # this is mostly copied from the baseline approach but adds
    # some lines to introduce the trained temp scaling layer produced by
    # train_temp_scaling_torch.py
    elif 'temp_scaling' in approach:
        if 'monitored' in approach:
            model = mymodels_torch.DFNetTunable(INPUT_SHAPES[representation],
                                                60,
                                                BASELINE_HYPERPARAMETERS[representation + '_' + protocol])
            model.load_state_dict(torch.load(representation + '_' + protocol + '_baseline_monitored_model' + str(trial) + '.pt'))
        else:
            model = mymodels_torch.DFNetTunable(INPUT_SHAPES[representation],
                                                61,
                                                BASELINE_HYPERPARAMETERS[representation + '_' + protocol])
            model.load_state_dict(torch.load(representation + '_' + protocol + '_baseline_model' + str(trial) + '.pt'))
        model.to(device)
        model.eval()
        temp_model = mymodels_torch.TemperatureScaling(float(approach[-3:]))
        temp_model.to(device)
        temp_model.eval()
        logits_batches = []
        for x_test, y_test in test_loader:
            with torch.no_grad():
                logits = model(x_test.to(device), training = False)
                logits_batches.append(temp_model(logits).to('cpu'))
        logits_concatenated = torch.cat(logits_batches, dim = 0)
        preds = torch.softmax(logits_concatenated, dim=1).detach().numpy()
        scores = []
        for i in range(len(preds)):
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
        model.load_state_dict(torch.load(representation + '_' + protocol + '_' + approach + '_model' + str(trial) + '.pt'))
        preds_avg, scores = get_bayesian_msp(test_loader, model, 61)
        if trial == 0:
            try:
                # print('ECE', check_calibration(scores, test_loader, approach, protocol))
                pass
            except:
                pass
        sscd_preds_avg[approach + str(len(test_loader)) + str(trial)] = preds_avg
        return preds_avg, scores

    # this loads a trained Spike and Slab Dropout model with Concrete Dropout layers before
    # using total uncertainty to rank predictions
    #
    # total uncertainty should correlate with MSP... a high MSP would be low entropy, thus
    # low total uncertainty, and vice versa... it includes epistemic uncertainty as one
    # component which we hypothesized would be higher for instances of unmonitored classes
    # due to a lack of training data, i.e. they are unknown at training time 
    elif (approach == 'sscd_uncertainty' or
         approach == 'sscd_mixup_uncertainty' or
         approach == 'sscd_nota_uncertainty' or
         approach == 'sscd_mixup_nota_uncertainty'):
        if approach[:-12] + str(len(test_loader)) in sscd_preds_avg:
            preds_avg = sscd_preds_avg[approach[:-12] + str(len(test_loader)) + str(trial)]
            del sscd_preds_avg[approach[:-12] + str(len(test_loader)) + str(trial)]
        else:
            model = mymodels_torch.DFNetTunableSSCD(INPUT_SHAPES[representation],
                                                    61,
                                                    BASELINE_HYPERPARAMETERS[representation + '_' + protocol])
            model.load_state_dict(torch.load(representation + '_' + protocol + '_' + approach[:-12] + '_model' + str(trial) + '.pt'))
            preds_avg, _ = get_bayesian_msp(test_loader, model, 61)
        scores = []
        for i in range(len(preds_avg)):
            # compute the total uncertainty for each test instance as the entropy
            # of the predicted probabilities for only the monitored classes 0-59
            #
            # we distribute the predicted probability mass for the unmonitored
            # class across the monitored classes to ensure that a highly confident
            # prediction of unmonitored doesn't yield near-zero entropy
            # across the monitored classes, which would be indistinguishable
            # from a highly confident prediction for a monitored class
            unmon_prob = preds_avg[i][60]
            distributed_prob = unmon_prob / 60
            adjusted_probs = preds_avg[i][:60] + distributed_prob
            adjusted_probs = numpy.where(adjusted_probs == 0, 1e-9, adjusted_probs)
            # we negate the entropy because precision_recall_curve() expects
            # a higher score to be associated with a higher confidence prediction
            # for the positive class, but a high uncertainty is a low confidence
            # prediction...
            negative_entropy = numpy.sum(adjusted_probs * numpy.log(adjusted_probs))
            scores.append(negative_entropy)
        return preds_avg, scores

    # this loads a trained baseline model and OpenGAN discriminator and returns
    # the predictions and scores from the discriminator
    elif (approach == 'opengan' or
          approach == 'opengan_mixup' or
          approach == 'opengan_monitored'):
        model = mymodels_torch.DFNetTunable(INPUT_SHAPES[representation],
                                            61,
                                            BASELINE_HYPERPARAMETERS[representation + '_' + protocol])
        model.load_state_dict(torch.load(representation + '_' + protocol + '_baseline_model' + str(trial) + '.pt'))
        model.to(device)
        model.eval()
        discriminator = mymodels_torch.DiscriminatorDFNet_fea(INPUT_SHAPES[representation],
                                                              61,
                                                              BASELINE_HYPERPARAMETERS[representation + '_' + protocol])
        discriminator.load_state_dict(torch.load(representation + '_' + protocol + '_' + approach + '_model' + str(trial) + '.pt'))
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

    # this loads a baseline model trained only on the monitored set
    # to extract feature maps and get closed-set predictions
    #
    # then loads the trained CSSRClassifier to get the reconstruction 
    # errors produced by the autoencoders for the predicted classes
    #
    # the negative class-specific reconstruction errors (normalized by 
    # class-specific means and feature magnitudes) are the scores for
    # the binary monitored vs. unmonitored task
    elif (approach == 'cssr'):
        backbone = mymodels_torch.DFNetTunable(INPUT_SHAPES[representation],
                                               60,
                                               BASELINE_HYPERPARAMETERS[representation + '_' + protocol])
        backbone.load_state_dict(torch.load(representation + '_' + protocol + '_baseline_monitored_model' + str(trial) + '.pt'))
        backbone.to(device)
        backbone.eval()
        classifier = mymodels_torch.CSSRClassifier(in_channels=256, num_classes=60, hidden_layers=[], latent_channels=32, gamma=0.1)
        classifier.load_state_dict(torch.load(representation + '_' + protocol + '_baseline_cssr_model' + str(trial) + '.pt'))
        classifier.to(device)
        classifier.eval()
        mean_reconstruction_errors = torch.load('train_baseline_cssr_means_' + protocol + '.pt')
        logits_batches = []
        scores = []
        for x_test, y_test in test_loader:
            with torch.no_grad():
                x_test = x_test.to(device)
                x_features = backbone.extract_features(x_test.to(device))
                logits = backbone(x_test, training = False)
                logits_batches.append(logits.to('cpu'))
                predicted_classes = torch.argmax(logits, dim=1)
                for i in range(x_features.size(0)):
                    predicted_class = predicted_classes[i].item()
                    autoencoder = classifier.class_aes[predicted_class]
                    instance_features = x_features[i:i+1]
                    rc = autoencoder(instance_features)
                    # make it negative, so lower reconstruction errors correspond to higher scores
                    reconstruction_error = -torch.norm(rc - instance_features, p=1, dim=1, keepdim=True).mean().item()
                    feature_magnitude = torch.norm(instance_features, p=1, dim=1, keepdim=True).mean().item()
                    class_mean = mean_reconstruction_errors[(protocol, representation)][predicted_class]
                    scores.append(reconstruction_error / (feature_magnitude ** 2) / class_mean)
        logits_concatenated = torch.cat(logits_batches, dim = 0)
        preds = torch.softmax(logits_concatenated, dim=1).detach().numpy()
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
            print('len(test_loader) is', len(test_loader))
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
        
        # open this to get the data saved from past runs so that we don't
        # need to run ten trials for every approach every time we add
        # a new approach, but we can run an approach again and overwrite
        # whatever was already saved for it
        #with open('pr_curve_data_' + protocol + '.pkl', 'rb') as handle:
        #    pr_curve_data = pickle.load(handle)
        
        # These approaches are the top competitors with the baseline
        for approach in ['baseline', 'baseline_mixup', 'baseline_nota', 'baseline_mixup_nota', 'opengan', 'opengan_mixup',
                         'sscd', 'sscd_uncertainty', 'sscd_mixup', 'sscd_mixup_uncertainty', 'sscd_nota', 'sscd_nota_uncertainty',
                         'sscd_mixup_nota', 'sscd_mixup_nota_uncertainty']:
        #for approach in ['temp_scaling_001', 'temp_scaling_002', 'temp_scaling_004', 'temp_scaling_008', 'temp_scaling_016', 'temp_scaling_032', 'temp_scaling_064', 'temp_scaling_128', 'temp_scaling_256']:
        #for approach in ['temp_scaling_0.9', 'temp_scaling_0.8', 'temp_scaling_0.7', 'temp_scaling_0.6', 'temp_scaling_0.5', 'temp_scaling_0.4', 'temp_scaling_0.3', 'temp_scaling_0.2', 'temp_scaling_0.1']:
        #for approach in ['baseline', 'temp_scaling_016', 'baseline_monitored', 'temp_scaling_monitored_016']:
        # These approaches are the competitors with monitored-only deterministic MSP
        #for approach in ['baseline_monitored', 'opengan_monitored', 'cssr']:
        #for approach in []:
            trial_scores = []
            trial_best_case_recalls = []
            trial_t_50_FPRs = []
            trial_t_60_FPRs = []
            trial_t_70_FPRs = []
            trial_t_80_FPRs = []
            trial_t_90_FPRs = []
            trial_accuracies = []
            for trial in range(20):
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
                
                # find t_60 on the validation set for this model
                t_60_index = numpy.argmin(numpy.abs(recalls_val - 0.60))
                if t_60_index == len(thresholds_val):
                    t_60 = thresholds_val[-1]
                else:
                    t_60 = thresholds_val[t_60_index]
                print('... t_60 validation recall, precision, and threshold are', recalls_val[t_60_index], precisions_val[t_60_index], t_60)

                # find t_70 on the validation set for this model
                t_70_index = numpy.argmin(numpy.abs(recalls_val - 0.70))
                if t_70_index == len(thresholds_val):
                    t_70 = thresholds_val[-1]
                else:
                    t_70 = thresholds_val[t_70_index]
                print('... t_70 validation recall, precision, and threshold are', recalls_val[t_70_index], precisions_val[t_70_index], t_70)

                # find t_80 on the validation set for this model
                t_80_index = numpy.argmin(numpy.abs(recalls_val - 0.80))
                if t_80_index == len(thresholds_val):
                    t_80 = thresholds_val[-1]
                else:
                    t_80 = thresholds_val[t_80_index]
                print('... t_80 validation recall, precision, and threshold are', recalls_val[t_80_index], precisions_val[t_80_index], t_80)

                # find t_90 on the validation set for this model
                t_90_index = numpy.argmin(numpy.abs(recalls_val - 0.90))
                if t_90_index == len(thresholds_val):
                    t_90 = thresholds_val[-1]
                else:
                    t_90 = thresholds_val[t_90_index]
                print('... t_90 validation recall, precision, and threshold are', recalls_val[t_90_index], precisions_val[t_90_index], t_90)
                
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
                    if scores[i] >= t_60:
                        preds_binary.append(True)
                    else:
                        preds_binary.append(False)
                FP_visits = []
                for i in range(len(preds_binary)):
                    if preds_binary[i] and not true_binary[i]:
                       FP_visits.append(splits['visits_test_64000'].iloc[i])
                trial_t_60_FPRs.append(len(FP_visits) / 64000)
                print('... False positives at t_60:', len(FP_visits))
                if len(FP_visits) <= 10:
                    print(FP_visits)
                    
                preds_binary = []
                for i in range(len(scores)):
                    if scores[i] >= t_70:
                        preds_binary.append(True)
                    else:
                        preds_binary.append(False)
                FP_visits = []
                for i in range(len(preds_binary)):
                    if preds_binary[i] and not true_binary[i]:
                       FP_visits.append(splits['visits_test_64000'].iloc[i])
                trial_t_70_FPRs.append(len(FP_visits) / 64000)
                print('... False positives at t_70:', len(FP_visits))
                if len(FP_visits) <= 10:
                    print(FP_visits)
                    
                preds_binary = []
                for i in range(len(scores)):
                    if scores[i] >= t_80:
                        preds_binary.append(True)
                    else:
                        preds_binary.append(False)
                FP_visits = []
                for i in range(len(preds_binary)):
                    if preds_binary[i] and not true_binary[i]:
                       FP_visits.append(splits['visits_test_64000'].iloc[i])
                trial_t_80_FPRs.append(len(FP_visits) / 64000)
                print('... False positives at t_80:', len(FP_visits))
                if len(FP_visits) <= 10:
                    print(FP_visits)
            
                preds_binary = []
                for i in range(len(scores)):
                    if scores[i] >= t_90:
                        preds_binary.append(True)
                    else:
                        preds_binary.append(False)
                FP_visits = []
                for i in range(len(preds_binary)):
                    if preds_binary[i] and not true_binary[i]:
                       FP_visits.append(splits['visits_test_64000'].iloc[i])
                trial_t_90_FPRs.append(len(FP_visits) / 64000)
                print('... False positives at t_90:', len(FP_visits))
                if len(FP_visits) <= 10:
                    print(FP_visits)
            
            if PLOT_MSP_DIST:
                if trial == 0:
                    scores_np = numpy.array(scores)
                    labels_np = numpy.array(true_binary)
                    scores_TP = scores_np[labels_np == 1]
                    scores_TN = scores_np[labels_np == 0]
                    msp_dist_data[approach] = (scores_TP, scores_TN)
                    
            if PLOT_CSSR_DIST:
                if trial == 0:
                    plot_classwise_scores(preds, scores, true_binary, num_classes=60, protocol=protocol)

            print('Mean recall at precision of 1.0:', numpy.mean(trial_best_case_recalls), 'StdDev: ', numpy.std(trial_best_case_recalls))
            print('Mean accuracy within monitored:', numpy.mean(trial_accuracies), 'StdDev: ', numpy.std(trial_accuracies))
            print('Mean FPR at t_50:', numpy.mean(trial_t_50_FPRs), 'StdDev: ', numpy.std(trial_t_50_FPRs))
            print('Mean FPR at t_60:', numpy.mean(trial_t_60_FPRs), 'StdDev: ', numpy.std(trial_t_60_FPRs))
            print('Mean FPR at t_70:', numpy.mean(trial_t_70_FPRs), 'StdDev: ', numpy.std(trial_t_70_FPRs))
            print('Mean FPR at t_80:', numpy.mean(trial_t_80_FPRs), 'StdDev: ', numpy.std(trial_t_80_FPRs))
            print('Mean FPR at t_90:', numpy.mean(trial_t_90_FPRs), 'StdDev: ', numpy.std(trial_t_90_FPRs))
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
            
        # this creates figures to show the separation between the
        # MSP distributions for monitored and unmonitored instances
        # before and after temperature scaling
        if PLOT_MSP_DIST:
            protocol_string = 'HTTPS' if protocol == 'https' else 'Tor'
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
            sns.kdeplot(msp_dist_data['baseline'][0], color="blue", label="Monitored", linewidth=3)
            sns.kdeplot(msp_dist_data['baseline'][1], color="black", label="Unmonitored", linewidth=3)
            plt.title('1AB ' + protocol_string)
            plt.xlabel('MSP')
            plt.xlim(0.0, 1.0)  # Set the x-axis limits to [0.0, 1.0]
            plt.ylabel('Density')
            plt.legend()

            plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
            sns.kdeplot(msp_dist_data['temp_scaling_016'][0], color="blue", label="Monitored", linewidth=3)
            sns.kdeplot(msp_dist_data['temp_scaling_016'][1], color="black", label="Unmonitored", linewidth=3)
            plt.title('2AB (T = 16) ' + protocol_string)
            plt.xlabel('MSP')
            plt.xlim(0.0, 1.0)  # Set the x-axis limits to [0.0, 1.0]
            plt.ylabel('Density')
            plt.legend()
            plt.savefig('temp_scaling_msp_dist_' + protocol + '.png', dpi=300)
            
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
            sns.kdeplot(msp_dist_data['baseline_monitored'][0], color="blue", label="Monitored", linewidth=3)
            sns.kdeplot(msp_dist_data['baseline_monitored'][1], color="black", label="Unmonitored", linewidth=3)
            plt.title('1A ' + protocol_string)
            plt.xlabel('MSP')
            plt.xlim(0.0, 1.0)  # Set the x-axis limits to [0.0, 1.0]
            plt.ylabel('Density')
            plt.legend()

            plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
            sns.kdeplot(msp_dist_data['temp_scaling_monitored_016'][0], color="blue", label="Monitored", linewidth=3)
            sns.kdeplot(msp_dist_data['temp_scaling_monitored_016'][1], color="black", label="Unmonitored", linewidth=3)
            plt.title('2A (T = 16) ' + protocol_string)
            plt.xlabel('MSP')
            plt.xlim(0.0, 1.0)  # Set the x-axis limits to [0.0, 1.0]
            plt.ylabel('Density')
            plt.legend()
            plt.savefig('temp_scaling_monitored_msp_dist_' + protocol + '.png', dpi=300)

        # save this for future runs, so we don't have to do
        # 20 trials for every approach every time we add
        # a new approach, or want to plot a slightly different
        # curve figure
        #with open('pr_curve_data_' + protocol + '_20_trials.pkl', 'wb') as handle:
        #    pickle.dump(pr_curve_data, handle)

        # create and save the P-R curve figure for top competitors with the baseline...
        plt.figure(figsize=(16, 18))
        # colors correspond to decision functions 1-6 and lines correspond to data A-E
        displays = {'baseline': ('MSP (1AB)', '#000000', '-'),
                    'baseline_mixup': ('MSP + mixup (1ABC)', '#000000', ':'),
                    'opengan': ('OpenGAN (5ABE)', '#ff0000', '-'),
                    'opengan_mixup': ('OpenGAN + mixup (5ABCE)', '#ff0000', ':'),
                    'baseline_nota': ('MSP + NOTA (1ABD)', '#000000', '--'),
                    'baseline_mixup_nota': ('MSP + mixup + NOTA (1ABCD)', '#000000', '-.'),
                    'sscd': ('Bayes. MSP (3AB)', '#0066ff', '-'),
                    'sscd_uncertainty': ('Bayes. Unc. (4AB)', '#00cc00', '-'),
                    'sscd_mixup': ('Bayes. MSP + mixup (3ABC)', '#0066ff', ':'),
                    'sscd_mixup_uncertainty': ('Bayes. Unc. + mixup (4ABC)', '#00cc00', ':'),
                    'sscd_nota': ('Bayes. MSP + NOTA', '#0066ff', '--'),
                    'sscd_nota_uncertainty': ('Bayes. Unc. + NOTA', '#00cc00', '--'),
                    'sscd_mixup_nota': ('Bayes. MSP + mixup + NOTA (3ABCD)', '#0066ff', '-.'),
                    'sscd_mixup_nota_uncertainty': ('Bayes. Unc. + mixup + NOTA (4ABCD)', '#00cc00', '-.')}
        for approach, mean_precisions in pr_curve_data.items():
            plt.plot(common_recall_levels,
                     mean_precisions,
                     label=displays[approach][0],
                     color=displays[approach][1],
                     linestyle=displays[approach][2],
                     linewidth = 4)
        plt.xlabel('Recall', fontsize = 32)
        plt.ylabel('Precision', fontsize = 32)
        protocol_string = 'HTTPS' if protocol == 'https' else 'Tor'
        plt.title('Precision-Recall Curve (' + protocol_string + ' 64k Test Set)', fontsize = 32)
        if protocol == 'https':
            plt.legend(loc = 'lower left', fontsize = 20, title = 'Approach', title_fontsize = 20)
        elif protocol == 'tor':
            plt.legend(loc = 'upper right', fontsize = 20, title = 'Approach', title_fontsize = 20)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.xlim(0.25, 1)
        plt.ylim(0.98, 1)
        plt.grid(True)
        plt.savefig('enhanced_pr_curve_' + protocol + '.png', dpi=300)
        pr_curve_data = {}
        print('-------------------------\n')
