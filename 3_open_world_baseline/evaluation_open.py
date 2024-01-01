import mymodels
from tensorflow import keras
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import numpy

# This takes no arguments on the command line. It just 
# loads the right model, hyperparameters, dataset splits, 
# and does the training and evaluation.
#
# output is the open-world performance

BASE_PATH = '/home/timothy.walsh/VF/3_open_world_baseline/'
DATA_POINTS = {'sirinam_wf': 5000,
               'sirinam_vf': 25000,
               'rahman': 25000,
               'hayden': 25000,
               'schuster2': 480,
               'schuster4': 960,
               'schuster8': 1920,
               'schuster16': 3840}
# we manually copy and paste these hyperparameters from the output of search_open.py
BEST_HYPERPARAMETERS = {'schuster8_https': {'filters': 128, 'kernel': 16, 'conv_stride': 1, 'pool': 16, 'pool_stride': 2, 'conv_dropout': 0.2, 'fc_neurons': 512, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.2, 'lr': 8.523214976539055e-05, 'batch_size': 128},
                        'schuster8_tor': {'filters': 64, 'kernel': 16, 'conv_stride': 1, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.6, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'elu', 'fc_dropout': 0.4, 'lr': 0.00026199601672768355, 'batch_size': 64}
                        }

def create_model(representation, protocol):
    if representation == 'sirinam_wf':
        model = mymodels.DFNet.build((DATA_POINTS[representation], 1), 61)
        lr = 0.002
    else:
        model = mymodels.DFNetTunable.build((DATA_POINTS[representation], 1), 61,
                                            BEST_HYPERPARAMETERS[representation + '_' + protocol])
        lr = BEST_HYPERPARAMETERS[representation + '_' + protocol]['lr']
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = lr),
                  loss = 'categorical_crossentropy',
                  metrics = [keras.metrics.Precision(class_id = 60),
                             keras.metrics.Recall(class_id = 60)])
    return model

def train_model(representation, protocol):
    with open(BASE_PATH + representation + '_open_world_' + protocol + '.pkl', 'rb') as handle:
        splits = pickle.load(handle)
    model = create_model(representation, protocol)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience = 10,
                                                   verbose = 1,
                                                   mode = 'min',
                                                   restore_best_weights = True)
    batch_size = 128 if representation == 'sirinam_wf' else BEST_HYPERPARAMETERS[representation + '_' + protocol]['batch_size']
    history = model.fit(splits['x_train'],
                        splits['y_train'],
                        epochs = 120,
                        callbacks = [early_stopping],
                        batch_size = batch_size,
                        validation_data = (splits['x_val'], splits['y_val']),
                        verbose = 2)
    return model

def get_threshold(model, representation, protocol):
    with open(BASE_PATH + representation + '_open_world_' + protocol + '.pkl', 'rb') as handle:
        splits = pickle.load(handle)
    preds = model.predict(splits['x_val'], verbose = 2)
    preds_labels = numpy.argmax(preds, axis = 1)
    preds_binary = (preds_labels < 60)
    true_labels = numpy.argmax(splits['y_val'], axis = 1)
    true_binary = (true_labels < 60)
    max_false_positive_prob = 0
    for i in range(len(preds_binary)):
        if preds_binary[i] != true_binary[i]:
            if preds[i, preds_labels[i]] > max_false_positive_prob:
                max_false_positive_prob = preds[i, preds_labels[i]]
                print('new highest probability for a validation set false positive:', str(max_false_positive_prob))
    return max_false_positive_prob

def evaluate_model(model, representation, protocol, threshold):
    with open(BASE_PATH + representation + '_open_world_' + protocol + '.pkl', 'rb') as handle:
        splits = pickle.load(handle)
    for size in ['1000', '2000', '4000', '8000', '16000', '32000', '64000']:
        print(representation, protocol, size)
        preds = model.predict(splits['x_test_' + size], verbose = 2)
        preds_labels = numpy.argmax(preds, axis = 1)
        preds_binary = (preds_labels < 60)
        true_labels = numpy.argmax(splits['y_test_' + size], axis = 1)
        true_binary = (true_labels < 60)

        print(representation, protocol, size, 'threshold of 0')
        conf_matrix = confusion_matrix(true_binary, preds_binary)
        FP = conf_matrix.sum(axis=0) - numpy.diag(conf_matrix)  
        FN = conf_matrix.sum(axis=1) - numpy.diag(conf_matrix)
        TP = numpy.diag(conf_matrix)
        TN = conf_matrix.sum() - (FP + FN + TP)
        # Precision
        P = TP/(TP+FP)
        # Recall, or true positive rate
        R = TP/(TP+FN)
        # False positive rate
        FPR = FP/(FP+TN)
        print('Precision:', str(P), 'Recall:', str(R), 'FPR:', str(FPR), 'False Positives:', str(FP))

        # adjust True / False predictions based on the given threshold
        for i in range(len(preds_binary)):
            # for every True prediction
            if preds_binary[i]:
                # lookup the probability for the predicted class for
                # this instance, and set the label to False if it's
                # less than or equal to the highest probability that
                # we found for a false positive in the validation set
                if preds[i, preds_labels[i]] <= threshold:
                    preds_binary[i] = False
        print(representation, protocol, size, 'threshold of', str(threshold))
        conf_matrix = confusion_matrix(true_binary, preds_binary)
        FP = conf_matrix.sum(axis=0) - numpy.diag(conf_matrix)  
        FN = conf_matrix.sum(axis=1) - numpy.diag(conf_matrix)
        TP = numpy.diag(conf_matrix)
        TN = conf_matrix.sum() - (FP + FN + TP)
        # Precision
        P = TP/(TP+FP)
        # Recall, or true positive rate
        R = TP/(TP+FN)
        # False positive rate
        FPR = FP/(FP+TN)
        print('Precision:', str(P), 'Recall:', str(R), 'FPR:', str(FPR), 'False Positives:', str(FP))

#for representation in ['dschuster16', 'schuster16', 'dschuster8', 'schuster8', 'schuster4', 'schuster2', 'hayden', 'rahman', 'sirinam_vf', 'sirinam_wf']:
for representation in ['schuster8']:
    for protocol in ['https', 'tor']:
        try:
            try:
                model = keras.models.load_model(representation + '_open_world_' + protocol + '_model.h5')
                print('found a trained model to load')
            except:
                print('did not find a trained model to load, starting to train one now...')
                model = train_model(representation, protocol)
#                model.save(representation + '_open_world_' + protocol + '_model.h5')
            threshold = get_threshold(model, representation, protocol)
            print('max probability for a false positive on the validation set:', str(threshold))
            evaluate_model(model, representation, protocol, threshold)
        except Exception as e:
            print(representation, protocol, e)
