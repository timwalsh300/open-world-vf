import torch
import mymodels_torch
import torch.nn.functional as F
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

INPUT_SHAPES = {'schuster8': (1, 1920),
                'dschuster16': (2, 3840)}

# we manually copy and paste these hyperparameters from the output of search_open.py
BASELINE_HYPERPARAMETERS = {'schuster8_tor': {'filters': 256, 'kernel': 8, 'conv_stride': 1, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 7.191906601911815e-05, 'batch_size': 128},
                            'dschuster16_https': {'filters': 256, 'kernel': 4, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.4, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.8, 'lr': 0.0005153393428807454, 'batch_size': 64}}

def evaluate_reconstruction_errors_and_stats(classifier, backbone, loader, device):
    # Initialize lists to store reconstruction errors for each class
    reconstruction_errors = [[] for _ in range(classifier.num_classes)]
    reconstruction_errors_fea_mag = [[] for _ in range(classifier.num_classes)]
    
    classifier.eval()
    backbone.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x_features = backbone.extract_features(x)
            y_indices = torch.argmax(y, dim=1)
            
            for i in range(x_features.size(0)):
                true_class = y_indices[i].item()
                autoencoder = classifier.class_aes[true_class]
                instance_features = x_features[i:i+1]
                rc = autoencoder(instance_features)
                reconstruction_error = torch.norm(rc - instance_features, p=1, dim=1, keepdim=True).mean().item()
                
                # Store reconstruction error without normalization
                reconstruction_errors[true_class].append(reconstruction_error)
                
                # Normalize by feature magnitude
                feature_magnitude = torch.norm(instance_features, p=1, dim=1, keepdim=True).mean().item()
                #feature_magnitude = torch.abs(instance_features).mean().item()
                reconstruction_error_fea_mag = reconstruction_error / (feature_magnitude ** 2)
                reconstruction_errors_fea_mag[true_class].append(reconstruction_error_fea_mag)

    # Compute mean reconstruction errors for each class from the training set errors
    mean_reconstruction_errors = [numpy.mean(errors) for errors in reconstruction_errors]
    mean_reconstruction_errors_fea_mag = [numpy.mean(errors) for errors in reconstruction_errors_fea_mag]
    
    return mean_reconstruction_errors, mean_reconstruction_errors_fea_mag

# Function to plot the reconstruction errors
def plot_reconstruction_errors(mean_reconstruction_errors, mean_reconstruction_errors_fea_mag, protocol):
    num_classes = len(mean_reconstruction_errors)
    
    x = numpy.arange(num_classes)
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.bar(x - width/2, mean_reconstruction_errors, width, label='No Normalization', color='black')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Reconstruction Error (No Normalization)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x, rotation=90)

    ax2 = ax1.twinx()
    ax2.bar(x + width/2, mean_reconstruction_errors_fea_mag, width, label='Feature Magnitude Normalization', color='gray')
    ax2.set_ylabel('Reconstruction Error (Feature Magnitude Normalization)', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.legend(loc='upper right')
    
    fig.tight_layout()
    plt.subplots_adjust(top=0.85) 
    plt.title('Mean Reconstruction Errors by Class (' + protocol + ')')
    plt.savefig('train_baseline_cssr_' + protocol + '.png', dpi=300)
    plt.show()

def evaluate_cssr_with_normalized_errors(classifier, backbone, loader, device, mean_reconstruction_errors):
    classifier.eval()
    backbone.eval()
    
    logits_batches = []
    scores_batches = []
    scores_fea_mag_batches = []
    scores_class_mean_batches = []
    scores_class_mean_fea_mag_batches = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x_features = backbone.extract_features(x)
            logits = backbone(x, training = False)
            logits_batches.append(logits.to('cpu'))
            
            predicted_classes = torch.argmax(logits, dim=1)
            scores = []
            scores_fea_mag = []
            scores_class_mean = []
            scores_class_mean_fea_mag = []
            
            for i in range(x_features.size(0)):
                predicted_class = predicted_classes[i].item()
                autoencoder = classifier.class_aes[predicted_class]
                instance_features = x_features[i:i+1]
                rc = autoencoder(instance_features)
                reconstruction_error = -torch.norm(rc - instance_features, p=1, dim=1, keepdim=True).mean().item()
                
                # Normalize by feature magnitude
                feature_magnitude = torch.norm(instance_features, p=1, dim=1, keepdim=True).mean().item()
                #feature_magnitude = torch.abs(instance_features).mean().item()
                reconstruction_error_fea_mag = reconstruction_error / (feature_magnitude ** 2)
                
                class_mean = mean_reconstruction_errors[predicted_class]
                
                # Compute various scores
                scores.append(reconstruction_error)
                scores_fea_mag.append(reconstruction_error_fea_mag)
                scores_class_mean.append(reconstruction_error / class_mean)
                scores_class_mean_fea_mag.append(reconstruction_error_fea_mag / class_mean)
            
            scores_batches.append(torch.tensor(scores).unsqueeze(1).to('cpu'))
            scores_fea_mag_batches.append(torch.tensor(scores_fea_mag).unsqueeze(1).to('cpu'))
            scores_class_mean_batches.append(torch.tensor(scores_class_mean).unsqueeze(1).to('cpu'))
            scores_class_mean_fea_mag_batches.append(torch.tensor(scores_class_mean_fea_mag).unsqueeze(1).to('cpu'))
    
    logits_concatenated = torch.cat(logits_batches, dim=0)
    preds = torch.softmax(logits_concatenated, dim=1).detach().numpy()
    scores = torch.cat(scores_batches, dim=0).detach().numpy()
    scores_fea_mag = torch.cat(scores_fea_mag_batches, dim=0).detach().numpy()
    scores_class_mean = torch.cat(scores_class_mean_batches, dim=0).detach().numpy()
    scores_class_mean_fea_mag = torch.cat(scores_class_mean_fea_mag_batches, dim=0).detach().numpy()
    
    return preds, scores, scores_fea_mag, scores_class_mean, scores_class_mean_fea_mag

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trial = 0
mean_reconstruction_errors_dict = {}
for protocol in ['https', 'tor']:
    for representation in ['dschuster16', 'schuster8']:
        try:
            train_tensors = torch.load(representation + '_' + protocol + '_train_mon_tensors.pt')
            val_tensors = torch.load(representation + '_' + protocol + '_val_tensors.pt')
        except:
            continue
        train_dataset = torch.utils.data.TensorDataset(*train_tensors)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=32,
                                                   shuffle=False)
        val_dataset = torch.utils.data.TensorDataset(*val_tensors)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
        y_val_batches = []
        for _, y_val in val_loader:
            y_val_batches.append(y_val)
        y_val_concatenated = torch.cat(y_val_batches, dim = 0)
        y_val_np = y_val_concatenated.numpy()
        true_labels_val = numpy.argmax(y_val_np, axis = 1)
        true_binary_val = (true_labels_val < 60)
        
        backbone = mymodels_torch.DFNetTunable(INPUT_SHAPES[representation], 60, BASELINE_HYPERPARAMETERS[representation + '_' + protocol])
        backbone.load_state_dict(torch.load(representation + '_' + protocol + '_baseline_monitored_model' + str(trial) + '.pt'))
        backbone.to(device)
        classifier = mymodels_torch.CSSRClassifier(in_channels=256, num_classes=60, hidden_layers=[], latent_channels=32, gamma=0.1)
        classifier.load_state_dict(torch.load(representation + '_' + protocol + '_baseline_cssr_model' + str(trial) + '.pt'))
        classifier.to(device)
        

        mean_reconstruction_errors, mean_reconstruction_errors_fea_mag = evaluate_reconstruction_errors_and_stats(classifier,
                                                                                                                 backbone,
                                                                                                                 train_loader,
                                                                                                                 device)
        plot_reconstruction_errors(mean_reconstruction_errors, mean_reconstruction_errors_fea_mag, protocol)
        mean_reconstruction_errors_dict[(protocol, representation)] = mean_reconstruction_errors
        torch.save(mean_reconstruction_errors_dict, 'train_baseline_cssr_means_' + protocol + '.pt')
        # use the following line during evaluation to load the dictionary of class means learned from training
        # mean_reconstruction_errors_dict = torch.load('train_baseline_cssr_means_' + protocol + '.pt')
        
        preds, scores_no_norm, scores_with_norm, scores_class_mean, scores_class_mean_norm = evaluate_cssr_with_normalized_errors(classifier,
                                                                                                                                backbone,
                                                                                                                                val_loader,
                                                                                                                                device,
                                                                                                                                mean_reconstruction_errors)
                                                                                                                                
        
        precisions_val, recalls_val, thresholds_val = precision_recall_curve(true_binary_val, scores_no_norm)
        print(protocol, 'PR-AUC reconstruction error', auc(recalls_val, precisions_val))
        precisions_val, recalls_val, thresholds_val = precision_recall_curve(true_binary_val, scores_with_norm)
        print(protocol, 'PR-AUC reconstruction error / feature magnitude', auc(recalls_val, precisions_val))
        precisions_val, recalls_val, thresholds_val = precision_recall_curve(true_binary_val, scores_class_mean)
        print(protocol, 'PR-AUC reconstruction error / class mean', auc(recalls_val, precisions_val))
        precisions_val, recalls_val, thresholds_val = precision_recall_curve(true_binary_val, scores_class_mean_norm)
        print(protocol, 'PR-AUC reconstruction error / class mean / feature magnitude', auc(recalls_val, precisions_val))

