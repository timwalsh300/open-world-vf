import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import pickle
from tensorflow.keras.optimizers import Adam
import numpy
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

class TemperatureScalingLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(TemperatureScalingLayer, self).__init__(**kwargs)
        self.temperature = tensorflow.Variable(initial_value = 2.0, trainable = True, dtype=tensorflow.float32)

    def call(self, inputs):
        return inputs / self.temperature

for representation in ['dschuster16', 'schuster8']:
    for protocol in ['https', 'tor']:
        try:
            base_model = keras.models.load_model('../3_open_world_baseline/' + representation + '_open_world_' + protocol + '_model.h5')
            print('found', representation + '_open_world_' + protocol + '_model.h5')
            print(base_model.summary())
        except Exception as e:
            print(e)
            continue

        # select all but the final softmax layer from the baseline model
        logits_model = Model(inputs = base_model.inputs, outputs = base_model.layers[-2].output)

        # define the temp scaling calibration model; just a TemperatureScalingLayer and a new softmax layer
        inputs = keras.Input(shape=(61,))
        temperature_scaled_logits = TemperatureScalingLayer()(inputs)
        outputs = layers.Activation('softmax')(temperature_scaled_logits)
        calibration_model = Model(inputs, outputs)

        # compile and fit the temp scaling calibration model; effectively learning the best T
        calibration_model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'categorical_crossentropy')
        with open('../3_open_world_baseline/' + representation + '_open_world_' + protocol + '_splits.pkl', 'rb') as handle:
            splits = pickle.load(handle)
        x_val_logits = logits_model.predict(splits['x_val'])
        # calibration_model.fit(x_val_logits, splits['y_val'], epochs = 10, batch_size = 32)
        final_temperature_value = calibration_model.layers[1].temperature.numpy()
        print("Final Temperature Value (T):", final_temperature_value)

        # combine the logits model and the temp scaling calibration model into one for later use
        inputs = keras.Input(shape=logits_model.input_shape[1:])
        logits = logits_model(inputs)
        outputs = calibration_model(logits)
        unified_model = Model(inputs, outputs)
        unified_model.save(representation + '_' + protocol + '_temp_scaling_model.h5')

        # plot and save a calibration curve figure
        preds = unified_model.predict(splits['x_val'], verbose = 2)
        scores = []
        for i in range(len(preds)):
            # Find the max softmax probability for any monitored
            # class. This will be fairly low if the argmax was 60 or
            # if the model was torn between two monitored classes.
            # We're implying the the probability of 60 is 1.0 - this.
            scores.append(max(preds[i][:60]))
        true_labels = numpy.argmax(splits['y_val'], axis = 1)
        true_binary = (true_labels < 60)
        prob_true, prob_pred = calibration_curve(true_binary, scores, n_bins=10, strategy='uniform')
        bin_edges = numpy.linspace(0, 1, 11)
        bin_width = numpy.diff(bin_edges)
        bin_centers = bin_edges[:-1] + bin_width / 2
        bin_counts = numpy.histogram(scores, bins=bin_edges)[0]
        plt.figure(figsize=(16, 12))
        print(len(bin_centers), len(prob_true), len(bin_width))
        plt.bar(bin_centers, prob_true, width = bin_width, align = 'center', alpha = 0.5, edgecolor='b', label='Calibration Curve')
        for i, count in enumerate(bin_counts):
            plt.text(bin_centers[i], prob_true[i], f' {count}', verticalalignment= 'bottom', horizontalalignment = 'center')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        plt.xlabel('Predicted Probability of Monitored and Number of Instances', fontsize = 32)
        plt.ylabel('True Monitored Frequency', fontsize = 32)
        plt.title('Calibration Curve (val set ' + protocol + ')', fontsize = 32)
        plt.legend(fontsize = 20)
        plt.savefig('cal_curve_val_set_' + protocol + '.png', dpi=300)
