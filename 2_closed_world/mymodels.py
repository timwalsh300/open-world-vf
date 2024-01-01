from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, ELU, MaxPooling1D, Dropout
from tensorflow.keras.layers import Activation, Flatten, Dense

# This reproduces the original Beauty and the Burst model architecture and
# hyperparameters from the work of Schuster et al. The input_shape should be
# (960, 1) for 1/4 second periods over 4 minutes, but it could be different.
# We should probably use MinMaxScaler on the input. Doesn't work yet.
class BeautyNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        model.add(Conv1D(32, kernel_size=16, activation='relu', input_shape=input_shape))
        model.add(Conv1D(32, kernel_size=16, activation='relu'))
        model.add(Conv1D(32, kernel_size=16, activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=6))
        model.add(Dropout(0.7))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))
        return model

# This is the original Deep Fingerprinting model architecture and hyperparameters
# from the work of Sirinam et al. The input shape should be (5000, 1) for website
# or sub-page fingerprinting based on just a few seconds of traffic.
class DFNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        filter_num = ['None',32,64,128,256]
        kernel_size = ['None',8,8,8,8]
        conv_stride_size = ['None',1,1,1,1]
        pool_stride_size = ['None',4,4,4,4]
        pool_size = ['None',8,8,8,8]

        model.add(Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                         input_shape=input_shape,
                         strides=conv_stride_size[1], padding='same',
                         name='block1_conv1'))
        model.add(BatchNormalization(axis=-1))
        model.add(ELU(alpha=1.0, name='block1_adv_act1'))
        model.add(Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                         strides=conv_stride_size[1], padding='same',
                         name='block1_conv2'))
        model.add(BatchNormalization(axis=-1))
        model.add(ELU(alpha=1.0, name='block1_adv_act2'))
        model.add(MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1],
                               padding='same', name='block1_pool'))
        model.add(Dropout(0.1, name='block1_dropout'))

        model.add(Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                         strides=conv_stride_size[2], padding='same',
                         name='block2_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block2_act1'))

        model.add(Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                         strides=conv_stride_size[2], padding='same',
                         name='block2_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block2_act2'))
        model.add(MaxPooling1D(pool_size=pool_size[2], strides=pool_stride_size[3],
                               padding='same', name='block2_pool'))
        model.add(Dropout(0.1, name='block2_dropout'))

        model.add(Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                         strides=conv_stride_size[3], padding='same',
                         name='block3_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block3_act1'))
        model.add(Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                         strides=conv_stride_size[3], padding='same',
                         name='block3_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block3_act2'))
        model.add(MaxPooling1D(pool_size=pool_size[3], strides=pool_stride_size[3],
                               padding='same', name='block3_pool'))
        model.add(Dropout(0.1, name='block3_dropout'))

        model.add(Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                         strides=conv_stride_size[4], padding='same',
                         name='block4_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block4_act1'))
        model.add(Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                         strides=conv_stride_size[4], padding='same',
                         name='block4_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='block4_act2'))
        model.add(MaxPooling1D(pool_size=pool_size[4], strides=pool_stride_size[4],
                               padding='same', name='block4_pool'))
        model.add(Dropout(0.1, name='block4_dropout'))

        model.add(Flatten(name='flatten'))
        model.add(Dense(512, kernel_initializer='glorot_uniform', name='fc1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='fc1_act'))

        model.add(Dropout(0.7, name='fc1_dropout'))

        model.add(Dense(512, kernel_initializer='glorot_uniform', name='fc2'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='fc2_act'))

        model.add(Dropout(0.5, name='fc2_dropout'))

        model.add(Dense(classes, kernel_initializer='glorot_uniform', name='fc3'))
        model.add(Activation('softmax', name="softmax"))
        return model

# This is a tunable version of the Deep Fingerprinting model architecture 
# from the work of Sirinam et al.
class DFNetTunable:
    @staticmethod
    def build(input_shape, classes, hype):
        model = Sequential()
        # block 1
        model.add(Conv1D(filters=hype['filters'], kernel_size=hype['kernel'],
                         input_shape=input_shape,
                         strides=hype['conv_stride'], padding='same',
                         name='block1_conv1'))
        model.add(BatchNormalization(axis=-1))
        model.add(ELU(alpha=1.0, name='block1_adv_act1'))
        model.add(Conv1D(filters=hype['filters'], kernel_size=hype['kernel'],
                         input_shape=input_shape,
                         strides=hype['conv_stride'], padding='same',
                         name='block1_conv2'))
        model.add(BatchNormalization(axis=-1))
        model.add(ELU(alpha=1.0, name='block1_adv_act2'))
        model.add(MaxPooling1D(pool_size=hype['pool'], strides=hype['pool_stride'],
                               padding='same', name='block1_pool'))
        model.add(Dropout(hype['conv_dropout'], name='block1_dropout'))
        
        # block 2
        model.add(Conv1D(filters=hype['filters'], kernel_size=hype['kernel'],
                         input_shape=input_shape,
                         strides=hype['conv_stride'], padding='same',
                         name='block2_conv1'))
        model.add(BatchNormalization(axis=-1))
        model.add(ELU(alpha=1.0, name='block2_adv_act1'))
        model.add(Conv1D(filters=hype['filters'], kernel_size=hype['kernel'],
                         input_shape=input_shape,
                         strides=hype['conv_stride'], padding='same',
                         name='block2_conv2'))
        model.add(BatchNormalization(axis=-1))
        model.add(ELU(alpha=1.0, name='block2_adv_act2'))
        model.add(MaxPooling1D(pool_size=hype['pool'], strides=hype['pool_stride'],
                               padding='same', name='block2_pool'))
        model.add(Dropout(hype['conv_dropout'], name='block2_dropout'))
        
        # block 3
        model.add(Conv1D(filters=hype['filters'], kernel_size=hype['kernel'],
                         input_shape=input_shape,
                         strides=hype['conv_stride'], padding='same',
                         name='block3_conv1'))
        model.add(BatchNormalization(axis=-1))
        model.add(ELU(alpha=1.0, name='block3_adv_act1'))
        model.add(Conv1D(filters=hype['filters'], kernel_size=hype['kernel'],
                         input_shape=input_shape,
                         strides=hype['conv_stride'], padding='same',
                         name='block3_conv2'))
        model.add(BatchNormalization(axis=-1))
        model.add(ELU(alpha=1.0, name='block3_adv_act2'))
        model.add(MaxPooling1D(pool_size=hype['pool'], strides=hype['pool_stride'],
                               padding='same', name='block3_pool'))
        model.add(Dropout(hype['conv_dropout'], name='block3_dropout'))
        
        # block 4
        model.add(Conv1D(filters=hype['filters'], kernel_size=hype['kernel'],
                         input_shape=input_shape,
                         strides=hype['conv_stride'], padding='same',
                         name='block4_conv1'))
        model.add(BatchNormalization(axis=-1))
        model.add(ELU(alpha=1.0, name='block4_adv_act1'))
        model.add(Conv1D(filters=hype['filters'], kernel_size=hype['kernel'],
                         input_shape=input_shape,
                         strides=hype['conv_stride'], padding='same',
                         name='block4_conv2'))
        model.add(BatchNormalization(axis=-1))
        model.add(ELU(alpha=1.0, name='block4_adv_act2'))
        model.add(MaxPooling1D(pool_size=hype['pool'], strides=hype['pool_stride'],
                               padding='same', name='block4_pool'))
        model.add(Dropout(hype['conv_dropout'], name='block4_dropout'))

        model.add(Flatten(name='flatten'))
        model.add(Dense(hype['fc_neurons'], kernel_initializer=hype['fc_init'], name='fc1'))
        model.add(BatchNormalization())
        model.add(Activation(hype['fc_activation'], name='fc1_act'))
        model.add(Dropout(hype['fc_dropout'], name='fc1_dropout'))

        model.add(Dense(hype['fc_neurons'], kernel_initializer=hype['fc_init'], name='fc2'))
        model.add(BatchNormalization())
        model.add(Activation(hype['fc_activation'], name='fc2_act'))
        model.add(Dropout(hype['fc_dropout'], name='fc2_dropout'))

        model.add(Dense(classes, kernel_initializer=hype['fc_init'], name='fc3'))
        model.add(Activation('softmax', name="softmax"))
        return model
