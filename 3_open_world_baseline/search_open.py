import mymodels
from ray import tune, train
from ray.tune.schedulers import AsyncHyperBandScheduler
from tensorflow import keras
import pickle
import sys

# This takes two arguments on the command line to load the
# desired dataset splits...
#
# 1: sirinam_vf, rahman, hayden, schuster2, schuster4, schuster8
# 2: tor or https
#
# output is the best hyperparameters

BASE_PATH = '/home/timothy.walsh/VF/3_open_world_baseline/'
REPRESENTATION = sys.argv[1]
PROTOCOL = sys.argv[2]
DATA_POINTS = {'sirinam_vf': 25000,
               'rahman': 25000,
               'hayden': 25000,
               'schuster2': 480,
               'schuster4': 960,
               'schuster8': 1920}

def create_model(hyperparameters):
    model = mymodels.DFNetTunable.build((DATA_POINTS[REPRESENTATION], 1), 61, hyperparameters)
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = hyperparameters['lr']),
                  loss='categorical_crossentropy',
                  metrics=[keras.metrics.Precision(class_id=60), keras.metrics.Recall(class_id=60)])
    return model

def train_model(config):
    with open(BASE_PATH + REPRESENTATION + '_open_world_' + PROTOCOL + '.pkl', 'rb') as handle:
        splits = pickle.load(handle)
    model = create_model(config)
    history = model.fit(splits['x_train'],
                        splits['y_train'],
                        epochs = 30,
                        batch_size = config['batch_size'],
                        validation_data = (splits['x_val'], splits['y_val']))
    val_loss = history.history['val_loss'][-1]
    return {'val_loss': val_loss}

def tune_model():
    sched = AsyncHyperBandScheduler(max_t = 30)
    tuner = tune.Tuner(tune.with_resources(train_model, resources = {'cpu': 2, 'gpu': 1}),
                       tune_config = tune.TuneConfig(metric = 'val_loss',
                                                     mode = 'min',
                                                     scheduler = sched,
                                                     num_samples = 80),
                       run_config = train.RunConfig(),
                       param_space = {'filters': tune.choice([32, 64, 128, 256]),
                                      'kernel': tune.choice([4, 8, 16, 32]),
                                      'conv_stride': tune.choice([1, 2, 4, 8, 16]),
                                      'pool': tune.choice([2, 4, 8, 16]),
                                      'pool_stride': tune.choice([1, 2, 4, 8, 16]),
                                      'conv_dropout': tune.choice([0.1, 0.2, 0.4, 0.6, 0.8]),
                                      'fc_neurons': tune.choice([128, 256, 512, 1024]),
                                      'fc_init': tune.choice(['glorot_uniform', 'he_normal']),
                                      'fc_activation': tune.choice(['relu', 'elu']),
                                      'fc_dropout': tune.choice([0.1, 0.2, 0.4, 0.6, 0.8]),
                                      'lr': tune.loguniform(1e-5, 1e-1),
                                      'batch_size': tune.choice([32, 64, 128])})
    results = tuner.fit()
    print('Best hyperparameters found were:', results.get_best_result().config)

tune_model()
