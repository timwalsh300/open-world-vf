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
from scipy.interpolate import interp1d
import pickle

INPUT_SHAPES = {'schuster8': (1, 1920),
                'dschuster16': (2, 3840)}

BASELINE_HYPERPARAMETERS = {'schuster8_tor': {'filters': 256, 'kernel': 8, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.1, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'relu', 'fc_dropout': 0.4, 'lr': 0.00014940538775594852, 'batch_size': 128},
                            'dschuster16_https': {'filters': 256, 'kernel': 4, 'conv_stride': 1, 'pool': 16, 'pool_stride': 4, 'conv_dropout': 0.2, 'fc_neurons': 256, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.6, 'lr': 6.0487458111998665e-05, 'batch_size': 32}}
    
common_recall_levels = numpy.linspace(0, 1, 500)

def get_scores(test_loader, protocol, representation, approach, training_region, trial):
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
        model.load_state_dict(torch.load('models/' + representation + '_' + protocol + '_vimeo_' + training_region + '_model'  + str(trial) + '.pt'))
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
         approach == 'sscd_nota'):
        model = mymodels_torch.DFNetTunableSSCD(INPUT_SHAPES[representation],
                                            61,
                                            BASELINE_HYPERPARAMETERS[representation + '_' + protocol])
        model.load_state_dict(torch.load(representation + '_' + protocol + '_' + approach + '_model' + str(trial) + '.pt'))
        preds, scores = get_bayesian_msp(test_loader, model, 61)
        if trial == 0:
            try:
                # print('ECE', check_calibration(scores, test_loader, approach, protocol))
                pass
            except:
                pass
        return preds, scores

    # this loads a trained Spike and Slab Dropout model with Concrete Dropout layers before
    # using total uncertainty to rank predictions
    #
    # total uncertainty should correlate with MSP... a high MSP would be low entropy, thus
    # low total uncertainty, and vice versa... it includes epistemic uncertainty as one
    # component which we hypothesized would be higher for instances of unmonitored classes
    # due to a lack of training data, i.e. they are unknown at training time 
    elif (approach == 'sscd_uncertainty' or
         approach == 'sscd_mixup_uncertainty' or
         approach == 'sscd_nota_uncertainty'):
        model = mymodels_torch.DFNetTunableSSCD(INPUT_SHAPES[representation],
                                            61,
                                            BASELINE_HYPERPARAMETERS[representation + '_' + protocol])
        model.load_state_dict(torch.load(representation + '_' + protocol + '_' + approach[:-12] + '_model' + str(trial) + '.pt'))
        return get_uncertainty(test_loader, model)

    # this loads a trained baseline model and OpenGAN discriminator and returns
    # the predictions and scores from the discriminator
    elif (approach == 'opengan' or
          approach == 'opengan_mixup' or
          approach == 'opengan_monitored'):
        model = mymodels_torch.DFNetTunable(INPUT_SHAPES[representation],
                                            60,
                                            BASELINE_HYPERPARAMETERS[representation + '_' + protocol])
        model.load_state_dict(torch.load('models/' + representation + '_' + protocol + '_vimeo_' + training_region + '_model'  + str(trial) + '.pt'))
        model.to(device)
        model.eval()
        discriminator = mymodels_torch.DiscriminatorDFNet_fea(INPUT_SHAPES[representation],
                                                              61,
                                                              BASELINE_HYPERPARAMETERS[representation + '_' + protocol])
        discriminator.load_state_dict(torch.load('models/' + representation + '_' + protocol + '_' + training_region + '_opengan_model' + str(trial) + '.pt'))
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

def get_uncertainty(test_loader, model):
        model.to(device)
        model.eval()
        mc_samples = 10
        preds = numpy.zeros([mc_samples, len(test_loader.dataset), 61])
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

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for protocol in ['https', 'tor']:
  for representation in ['dschuster16', 'schuster8']:
#    for training_region in ['africa', 'brazil', 'frankfurt', 'london', 'oregon', 'seoul', 'stockholm', 'sydney', 'uae', 'virginia']:
    for training_region in ['oregon']:
        try:
            val_tensors = torch.load('../4_open_world_enhancements/' + representation + '_' + protocol + '_val_tensors.pt')
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
        
        # These approaches are the top competitors with the baseline
        for approach in ['baseline_monitored', 'opengan_monitored']:
            trial_scores = []
            trial_best_case_recalls = []
            trial_t_50_FPRs = []
            trial_t_75_FPRs = []
            trial_accuracies = []
            for trial in range(10):
                print('Getting scores for', protocol, approach, training_region, '... Trial', trial)
                
                # find t_50 on the validation set for this model
                preds_val, scores_val = get_scores(val_loader, protocol, representation, approach, training_region, trial)
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
                preds, scores = get_scores(test_loader, protocol, representation, approach, training_region, trial)
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
