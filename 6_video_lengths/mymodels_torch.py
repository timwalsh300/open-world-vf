import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy
from bayesian_torch.layers import Conv1dFlipout, LinearFlipout

# This redefines DFNet from Sirinam et al. "Deep Fingerprinting"
# in PyTorch and in a way that is suitable for tuning a bunch
# of hyperparameters with Ray Tune
#
# It also enables L2 prior regularization for MAP estimation and
# enables Monte Carlo Dropout with the option to sample during
# inference
class DFNetTunable(nn.Module):
    def __init__(self, input_shape, classes, hyperparameters):
        super(DFNetTunable, self).__init__()
        self.hyperparameters = hyperparameters
        self.conv_padding_left = int(hyperparameters['kernel'] / 2)
        self.pool_padding_left = int(hyperparameters['pool'] / 2)
        self.conv_padding_right = int(hyperparameters['kernel'] / 2) - 1
        self.pool_padding_right = int(hyperparameters['pool'] / 2) - 1
        conv_downsampling = hyperparameters['conv_stride'] ** 8 if hyperparameters['conv_stride'] > 1 else 1
        pool_downsampling = hyperparameters['pool_stride'] ** 4 if hyperparameters['pool_stride'] > 1 else 1

        # Block 1
        self.block1_conv1 = nn.Conv1d(in_channels=input_shape[0], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block1_bn1 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block1_act1 = nn.ELU()
        self.block1_conv2 = nn.Conv1d(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block1_bn2 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block1_act2 = nn.ELU()
        self.block1_pool = nn.MaxPool1d(kernel_size=hyperparameters['pool'], stride=hyperparameters['pool_stride'])

        # Block 2
        self.block2_conv1 = nn.Conv1d(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block2_bn1 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block2_act1 = nn.ELU()
        self.block2_conv2 = nn.Conv1d(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block2_bn2 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block2_act2 = nn.ELU()
        self.block2_pool = nn.MaxPool1d(kernel_size=hyperparameters['pool'], stride=hyperparameters['pool_stride'])

        # Block 3
        self.block3_conv1 = nn.Conv1d(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block3_bn1 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block3_act1 = nn.ELU()
        self.block3_conv2 = nn.Conv1d(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block3_bn2 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block3_act2 = nn.ELU()
        self.block3_pool = nn.MaxPool1d(kernel_size=hyperparameters['pool'], stride=hyperparameters['pool_stride'])

        # Block 4
        self.block4_conv1 = nn.Conv1d(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block4_bn1 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block4_act1 = nn.ELU()
        self.block4_conv2 = nn.Conv1d(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block4_bn2 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block4_act2 = nn.ELU()
        self.block4_pool = nn.MaxPool1d(kernel_size=hyperparameters['pool'], stride=hyperparameters['pool_stride'])

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=hyperparameters['filters'] * math.ceil(input_shape[1] / (conv_downsampling * pool_downsampling)), out_features=hyperparameters['fc_neurons'])
        self.fc1_bn = nn.BatchNorm1d(num_features=hyperparameters['fc_neurons'])
        self.fc1_act = nn.ReLU() if hyperparameters['fc_activation'] == 'relu' else nn.ELU()

        self.fc2 = nn.Linear(in_features=hyperparameters['fc_neurons'], out_features=hyperparameters['fc_neurons'])
        self.fc2_bn = nn.BatchNorm1d(num_features=hyperparameters['fc_neurons'])
        self.fc2_act = nn.ReLU() if hyperparameters['fc_activation'] == 'relu' else nn.ELU()

        self.fc3 = nn.Linear(in_features=hyperparameters['fc_neurons'], out_features=classes)

    # 'training' is True when using the model in the train() mode, and when
    # using it in the eval() mode for Monte Carlo Dropout. When using it
    # in eval() for the baseline approach, it is False
    def forward(self, x, training):
        # Block 1
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block1_conv1(x)
        x = self.block1_bn1(x)
        x = self.block1_act1(x)
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block1_conv2(x)
        x = self.block1_bn2(x)
        x = self.block1_act2(x)
        x = F.pad(x, (self.pool_padding_left, self.pool_padding_right))
        x = F.dropout(self.block1_pool(x),
                                        p = self.hyperparameters['conv_dropout'],
                                        training = training)

        # Block 2
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block2_conv1(x)
        x = self.block2_bn1(x)
        x = self.block2_act1(x)
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block2_conv2(x)
        x = self.block2_bn2(x)
        x = self.block2_act2(x)
        x = F.pad(x, (self.pool_padding_left, self.pool_padding_right))
        x = F.dropout(self.block2_pool(x),
                                        p = self.hyperparameters['conv_dropout'],
                                        training = training)

        # Block 3
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block3_conv1(x)
        x = self.block3_bn1(x)
        x = self.block3_act1(x)
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block3_conv2(x)
        x = self.block3_bn2(x)
        x = self.block3_act2(x)
        x = F.pad(x, (self.pool_padding_left, self.pool_padding_right))
        x = F.dropout(self.block3_pool(x),
                                        p = self.hyperparameters['conv_dropout'],
                                        training = training)

        # Block 4
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block4_conv1(x)
        x = self.block4_bn1(x)
        x = self.block4_act1(x)
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block4_conv2(x)
        x = self.block4_bn2(x)
        x = self.block4_act2(x)
        x = F.pad(x, (self.pool_padding_left, self.pool_padding_right))
        x = F.dropout(self.block4_pool(x),
                                        p = self.hyperparameters['conv_dropout'],
                                        training = training)

        x = self.flatten(x)

        # Fully connected layers
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.dropout(self.fc1_act(x),
                                        p = self.hyperparameters['fc_dropout'],
                                        training = training)

        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.dropout(self.fc2_act(x),
                                        p = self.hyperparameters['fc_dropout'],
                                        training = training)

        x = self.fc3(x)
        return x
        
    def L2reg(self, l2_coeff):
        l2reg_sum = l2_coeff * sum(torch.square(theta).sum() for theta in self.parameters())
        return l2reg_sum

    # This gives us the features from the last convolutional block
    # to use with our OpenGAN discriminators
    #
    # training = false because we only use this with a pre-trained model
    def extract_features(self, x, training = False):
        # Block 1
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block1_conv1(x)
        x = self.block1_bn1(x)
        x = self.block1_act1(x)
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block1_conv2(x)
        x = self.block1_bn2(x)
        x = self.block1_act2(x)
        x = F.pad(x, (self.pool_padding_left, self.pool_padding_right))
        x = F.dropout(self.block1_pool(x),
                                        p = self.hyperparameters['conv_dropout'],
                                        training = training)

        # Block 2
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block2_conv1(x)
        x = self.block2_bn1(x)
        x = self.block2_act1(x)
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block2_conv2(x)
        x = self.block2_bn2(x)
        x = self.block2_act2(x)
        x = F.pad(x, (self.pool_padding_left, self.pool_padding_right))
        x = F.dropout(self.block2_pool(x),
                                        p = self.hyperparameters['conv_dropout'],
                                        training = training)

        # Block 3
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block3_conv1(x)
        x = self.block3_bn1(x)
        x = self.block3_act1(x)
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block3_conv2(x)
        x = self.block3_bn2(x)
        x = self.block3_act2(x)
        x = F.pad(x, (self.pool_padding_left, self.pool_padding_right))
        x = F.dropout(self.block3_pool(x),
                                        p = self.hyperparameters['conv_dropout'],
                                        training = training)

        # Block 4
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block4_conv1(x)
        x = self.block4_bn1(x)
        x = self.block4_act1(x)
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block4_conv2(x)
        x = self.block4_bn2(x)
        x = self.block4_act2(x)
        x = F.pad(x, (self.pool_padding_left, self.pool_padding_right))
        x = F.dropout(self.block4_pool(x),
                                        p = self.hyperparameters['conv_dropout'],
                                        training = training)
        return x
        
    # this turns the 256 feature maps into a 1x1 shape
    # with 256 channels with global average pooling
    # to use with the example code for OpenGAN_fea
    def extract_features_gap(self, x):
        x = self.extract_features(x)
        x = x.unsqueeze(-1)
        x = F.adaptive_avg_pool2d(x, 1)
        return x
        
    # this flattens the 256 feature maps the same way
    # that the baseline architecture does to use our own
    # discriminator architecture for OpenGAN_fea
    def extract_features_flattened(self, x):
        x = self.extract_features(x)
        x = self.flatten(x)
        return x

# This defines a temperature scaling layer that has
# just one parameter, the temperature, to scale
# the logits of another trained model
class TemperatureScaling(torch.nn.Module):
    def __init__(self, t = 1.0):
        super(TemperatureScaling, self).__init__()
        self.temperature = torch.nn.Parameter(torch.full((1,), t))

    def forward(self, logits):
        return logits / self.temperature

class ConcreteDropout(torch.nn.Module):

    """Concrete Dropout.

    Implementation of the Concrete Dropout module as described in the
    'Concrete Dropout' paper: https://arxiv.org/pdf/1705.07832
    """

    def __init__(self,
                 weight_regulariser: float,
                 dropout_regulariser: float,
                 init_min: float = 0.1,
                 init_max: float = 0.1) -> None:

        """Concrete Dropout.

        Parameters
        ----------
        weight_regulariser : float
            Weight regulariser term.
        dropout_regulariser : float
            Dropout regulariser term.
        init_min : float
            Initial min value.
        init_max : float
            Initial max value.
        """

        super().__init__()

        self.weight_regulariser = weight_regulariser
        self.dropout_regulariser = dropout_regulariser

        init_min = numpy.log(init_min) - numpy.log(1.0 - init_min)
        init_max = numpy.log(init_max) - numpy.log(1.0 - init_max)

        self.p_logit = torch.nn.parameter.Parameter(torch.empty(1).uniform_(init_min, init_max))
        self.p = torch.sigmoid(self.p_logit)

        self.regularisation = 0.0

    def forward(self, x: torch.Tensor, layer: torch.nn.Module) -> torch.Tensor:

        """Calculates the forward pass.

        The regularisation term for the layer is calculated and assigned to a
        class attribute - this can later be accessed to evaluate the loss.

        Parameters
        ----------
        x : Tensor
            Input to the Concrete Dropout.
        layer : nn.Module
            Layer for which to calculate the Concrete Dropout.

        Returns
        -------
        Tensor
            Output from the dropout layer.
        """

        output = layer(self._concrete_dropout(x))

        sum_of_squares = 0
        for param in layer.parameters():
            sum_of_squares += torch.sum(torch.pow(param, 2))

        weights_reg = self.weight_regulariser * sum_of_squares / (1.0 - self.p)

        dropout_reg = self.p * torch.log(self.p)
        dropout_reg += (1.0 - self.p) * torch.log(1.0 - self.p)
        dropout_reg *= self.dropout_regulariser * x[0].numel()

        self.regularisation = weights_reg + dropout_reg

        return output

    def _concrete_dropout(self, x: torch.Tensor) -> torch.Tensor:

        """Computes the Concrete Dropout.

        Parameters
        ----------
        x : Tensor
            Input tensor to the Concrete Dropout layer.

        Returns
        -------
        Tensor
            Outputs from Concrete Dropout.
        """

        eps = 1e-7
        tmp = 0.1

        self.p = torch.sigmoid(self.p_logit)
        u_noise = torch.rand_like(x)

        drop_prob = (torch.log(self.p + eps) -
                     torch.log(1 - self.p + eps) +
                     torch.log(u_noise + eps) -
                     torch.log(1 - u_noise + eps))

        drop_prob = torch.sigmoid(drop_prob / tmp)

        random_tensor = 1 - drop_prob
        retain_prob = 1 - self.p

        x = torch.mul(x, random_tensor) / retain_prob

        return x

# This redefines the DFNetTunable model using Spike and Slab Dropout
# with Flipout and Concrete Dropout layers in every block
class DFNetTunableSSCD(nn.Module):
    # w and d are only used for training, so they should provide when instantiating
    # the model before training, but can / should be omitted when instantiating the
    # model to load a state dictionary for testing after training 
    def __init__(self, input_shape, classes, hyperparameters, w = 1, d = 1):
        super(DFNetTunableSSCD, self).__init__()
        self.hyperparameters = hyperparameters      
        self.w = w
        self.d = d
        self.conv_padding_left = int(hyperparameters['kernel'] / 2)
        self.pool_padding_left = int(hyperparameters['pool'] / 2)
        self.conv_padding_right = int(hyperparameters['kernel'] / 2) - 1
        self.pool_padding_right = int(hyperparameters['pool'] / 2) - 1
        conv_downsampling = hyperparameters['conv_stride'] ** 8 if hyperparameters['conv_stride'] > 1 else 1
        pool_downsampling = hyperparameters['pool_stride'] ** 4 if hyperparameters['pool_stride'] > 1 else 1

        # Block 1
        self.block1_conv1 = Conv1dFlipout(in_channels=input_shape[0], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block1_bn1 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block1_act1 = nn.ELU()
        self.block1_conv2 = Conv1dFlipout(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block1_bn2 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block1_act2 = nn.ELU()
        self.block1_pool = nn.MaxPool1d(kernel_size=hyperparameters['pool'], stride=hyperparameters['pool_stride'])
        self.block1_cd = ConcreteDropout(weight_regulariser = self.w, dropout_regulariser = self.d)

        # Block 2
        self.block2_conv1 = Conv1dFlipout(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block2_bn1 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block2_act1 = nn.ELU()
        self.block2_conv2 = Conv1dFlipout(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block2_bn2 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block2_act2 = nn.ELU()
        self.block2_pool = nn.MaxPool1d(kernel_size=hyperparameters['pool'], stride=hyperparameters['pool_stride'])
        self.block2_cd = ConcreteDropout(weight_regulariser = self.w, dropout_regulariser = self.d)

        # Block 3
        self.block3_conv1 = Conv1dFlipout(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block3_bn1 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block3_act1 = nn.ELU()
        self.block3_conv2 = Conv1dFlipout(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block3_bn2 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block3_act2 = nn.ELU()
        self.block3_pool = nn.MaxPool1d(kernel_size=hyperparameters['pool'], stride=hyperparameters['pool_stride'])
        self.block3_cd = ConcreteDropout(weight_regulariser = self.w, dropout_regulariser = self.d)

        # Block 4
        self.block4_conv1 = Conv1dFlipout(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block4_bn1 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block4_act1 = nn.ELU()
        self.block4_conv2 = Conv1dFlipout(in_channels=hyperparameters['filters'], out_channels=hyperparameters['filters'], kernel_size=hyperparameters['kernel'], stride=hyperparameters['conv_stride'])
        self.block4_bn2 = nn.BatchNorm1d(num_features=hyperparameters['filters'])
        self.block4_act2 = nn.ELU()
        self.block4_pool = nn.MaxPool1d(kernel_size=hyperparameters['pool'], stride=hyperparameters['pool_stride'])
        self.block4_cd = ConcreteDropout(weight_regulariser = self.w, dropout_regulariser = self.d)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = LinearFlipout(in_features=hyperparameters['filters'] * math.ceil(input_shape[1] / (conv_downsampling * pool_downsampling)), out_features=hyperparameters['fc_neurons'])
        self.fc1_bn = nn.BatchNorm1d(num_features=hyperparameters['fc_neurons'])
        self.fc1_act = nn.ReLU() if hyperparameters['fc_activation'] == 'relu' else nn.ELU()
        self.fc1_cd = ConcreteDropout(weight_regulariser = self.w, dropout_regulariser = self.d)

        self.fc2 = LinearFlipout(in_features=hyperparameters['fc_neurons'], out_features=hyperparameters['fc_neurons'])
        self.fc2_bn = nn.BatchNorm1d(num_features=hyperparameters['fc_neurons'])
        self.fc2_act = nn.ReLU() if hyperparameters['fc_activation'] == 'relu' else nn.ELU()
        self.fc2_cd = ConcreteDropout(weight_regulariser = self.w, dropout_regulariser = self.d)

        self.fc3 = LinearFlipout(in_features=hyperparameters['fc_neurons'], out_features=classes)

    # training = True at all times; we don't even use this argument; it's
    # only here so that we can re-use the same functions that we previously
    # wrote for Monte Carlo Dropout (which does require this argument)
    def forward(self, x, training = True):
        # Block 1
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block1_conv1(x, return_kl = False)
        x = self.block1_bn1(x)
        x = self.block1_act1(x)
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block1_conv2(x, return_kl = False)
        x = self.block1_bn2(x)
        x = self.block1_act2(x)
        x = F.pad(x, (self.pool_padding_left, self.pool_padding_right))
        x = self.block1_cd(x, self.block1_pool)

        # Block 2
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block2_conv1(x, return_kl = False)
        x = self.block2_bn1(x)
        x = self.block2_act1(x)
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block2_conv2(x, return_kl = False)
        x = self.block2_bn2(x)
        x = self.block2_act2(x)
        x = F.pad(x, (self.pool_padding_left, self.pool_padding_right))
        x = self.block2_cd(x, self.block2_pool)

        # Block 3
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block3_conv1(x, return_kl = False)
        x = self.block3_bn1(x)
        x = self.block3_act1(x)
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block3_conv2(x, return_kl = False)
        x = self.block3_bn2(x)
        x = self.block3_act2(x)
        x = F.pad(x, (self.pool_padding_left, self.pool_padding_right))
        x = self.block3_cd(x, self.block3_pool)

        # Block 4
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block4_conv1(x, return_kl = False)
        x = self.block4_bn1(x)
        x = self.block4_act1(x)
        x = F.pad(x, (self.conv_padding_left, self.conv_padding_right))
        x = self.block4_conv2(x, return_kl = False)
        x = self.block4_bn2(x)
        x = self.block4_act2(x)
        x = F.pad(x, (self.pool_padding_left, self.pool_padding_right))
        x = self.block4_cd(x, self.block4_pool)

        x = self.flatten(x)

        # Fully connected layers
        x = self.fc1(x, return_kl = False)
        x = self.fc1_bn(x)
        x = self.fc1_cd(x, self.fc1_act)

        x = self.fc2(x, return_kl = False)
        x = self.fc2_bn(x)
        x = self.fc2_cd(x, self.fc2_act)

        x = self.fc3(x, return_kl = False)
        return x
    
    def bernoulli_kl_loss(self, representation):
        # our prior for p_drop is 0.5 to maximize uncertainty
        #
        # here are the number of neurons in each CD layer for the
        # schuster8_tor and dschuster16_https models
        neurons = {'schuster8_tor': [480,120,30,8,128,128],
                   'dschuster16_https': [960,240,60,15,1024,1024]}
        
        bernoulli_kl_loss = 0.0
        bernoulli_kl_loss += neurons[representation][0] * (self.block1_cd.p * torch.log(self.block1_cd.p / 0.5) + (1 - self.block1_cd.p) * torch.log((1 - self.block1_cd.p) / 0.5))
        bernoulli_kl_loss += neurons[representation][1] * (self.block2_cd.p * torch.log(self.block2_cd.p / 0.5) + (1 - self.block2_cd.p) * torch.log((1 - self.block2_cd.p) / 0.5))
        bernoulli_kl_loss += neurons[representation][2] * (self.block3_cd.p * torch.log(self.block3_cd.p / 0.5) + (1 - self.block3_cd.p) * torch.log((1 - self.block3_cd.p) / 0.5))
        bernoulli_kl_loss += neurons[representation][3] * (self.block4_cd.p * torch.log(self.block4_cd.p / 0.5) + (1 - self.block4_cd.p) * torch.log((1 - self.block4_cd.p) / 0.5))
        bernoulli_kl_loss += neurons[representation][4] * (self.fc1_cd.p * torch.log(self.fc1_cd.p / 0.5) + (1 - self.fc1_cd.p) * torch.log((1 - self.fc1_cd.p) / 0.5))
        bernoulli_kl_loss += neurons[representation][5] * (self.fc2_cd.p * torch.log(self.fc2_cd.p / 0.5) + (1 - self.fc2_cd.p) * torch.log((1 - self.fc2_cd.p) / 0.5))
        return bernoulli_kl_loss

# This is our adaptation of the generator
# for OpenGAN_fea by Kong et al.
#
# Their Jupyter Notebook example differs from what's described
# in the paper. It is a DCGAN that takes noise inputs of size
# 1x1 with 100 channels, and works depth-wise to increase that
# to the same 1x1 but with nc channels. nc = 512 for block 4 of
# ResNet 18, but nc = 256 for block 4 of our model. The paper
# instead described a generator with fully-connected layers that
# output shape 1x512, which is what you get when you flatten the
# output of ResNet18 block 4.
class GeneratorOpenGAN_fea(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=256):
        super(GeneratorOpenGAN_fea, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        
        self.main = nn.Sequential(
            nn.Conv2d( self.nz, self.ngf * 8, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            nn.Conv2d(self.ngf * 8, self.ngf * 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            nn.Conv2d( self.ngf * 4, self.ngf * 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            nn.Conv2d( self.ngf * 2, self.ngf*4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU(True),
            nn.Conv2d( self.ngf*4, self.nc, 1, 1, 0, bias=True)
        )

    def forward(self, input):
        return self.main(input)

# This is our adaptation of the discriminator
# for OpenGAN_fea by Kong et al.
#
# Their Jupyter Notebook example differs from what's described
# in the paper. It is a CNN that takes inputs of size
# 1x1 with nc channels, and works depth-wise to reduce that
# to the same 1x1 but with 1 channel, or just a single value,
# which is the discriminator's prediction. The paper
# instead described a discriminator with fully-connected layers
# that got progressively smaller to reduce an input of nc
# dimensions to a single value output.
#
# For this to work on the output of our model's block 4, we need
# to use a global avg pooling layer or AdaptiveAvgPool2d(1) to
# reduce the 256 feature maps to a 1x1 shape with 256 channels.
class DiscriminatorOpenGAN_fea(nn.Module):
    def __init__(self, nc=256, ndf=64):
        super(DiscriminatorOpenGAN_fea, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.main = nn.Sequential(
            nn.Conv2d(self.nc, self.ndf*8, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf*8, self.ndf*4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf*4, self.ndf*2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf*2, self.ndf, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# This is our adaptation of the MLP generator for
# OpenGAN_fea described in the paper by Kong et al.
# to pair with our DiscriminatorDFNet
class GeneratorDFNet_fea(nn.Module):
    def __init__(self, input_shape, hyperparameters, nz = 100):
        super(GeneratorDFNet_fea, self).__init__()
        conv_downsampling = hyperparameters['conv_stride'] ** 8 if hyperparameters['conv_stride'] > 1 else 1
        pool_downsampling = hyperparameters['pool_stride'] ** 4 if hyperparameters['pool_stride'] > 1 else 1
        
        self.main = nn.Sequential(
            nn.Linear(nz, nz * 8),
            nn.BatchNorm1d(nz * 8),
            torch.nn.LeakyReLU(),
            nn.Linear(nz * 8, nz * 4),
            nn.BatchNorm1d(nz * 4),
            torch.nn.LeakyReLU(),
            nn.Linear(nz * 4, nz * 2),
            nn.BatchNorm1d(nz * 2),
            torch.nn.LeakyReLU(),
            nn.Linear(nz * 2, nz * 4),
            nn.BatchNorm1d(nz * 4),
            torch.nn.LeakyReLU(),
            nn.Linear(in_features=nz * 4, out_features=hyperparameters['filters'] * math.ceil(input_shape[1] / (conv_downsampling * pool_downsampling))),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# This is our own discriminator architecture
# for OpenGAN_fea by Kong et al.
#
# It uses the same fully-connected architecture
# from our baseline model, so it's a continuation
# of the baseline model after the convolutional
# blocks that get invoked with extract_features()
#
# It outputs a logit per class which we can treat
# as a binary decision by using
# torch.max(torch.softmax(x, dim=1)[:, :60], dim=1)
class DiscriminatorDFNet_fea(nn.Module):
    def __init__(self, input_shape, classes, hyperparameters):
        super(DiscriminatorDFNet_fea, self).__init__()
        self.hyperparameters = hyperparameters
        conv_downsampling = hyperparameters['conv_stride'] ** 8 if hyperparameters['conv_stride'] > 1 else 1
        pool_downsampling = hyperparameters['pool_stride'] ** 4 if hyperparameters['pool_stride'] > 1 else 1
        self.fc1 = nn.Linear(in_features=hyperparameters['filters'] * math.ceil(input_shape[1] / (conv_downsampling * pool_downsampling)), out_features=hyperparameters['fc_neurons'])
        self.fc1_bn = nn.BatchNorm1d(num_features=hyperparameters['fc_neurons'])
        self.fc1_act = nn.ReLU() if hyperparameters['fc_activation'] == 'relu' else nn.ELU()

        self.fc2 = nn.Linear(in_features=hyperparameters['fc_neurons'], out_features=hyperparameters['fc_neurons'])
        self.fc2_bn = nn.BatchNorm1d(num_features=hyperparameters['fc_neurons'])
        self.fc2_act = nn.ReLU() if hyperparameters['fc_activation'] == 'relu' else nn.ELU()

        self.fc3 = nn.Linear(in_features=hyperparameters['fc_neurons'], out_features=classes)

    def forward(self, x, training):
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.dropout(self.fc1_act(x),
                      p = self.hyperparameters['fc_dropout'],
                      training = training)
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.dropout(self.fc2_act(x),
                      p = self.hyperparameters['fc_dropout'],
                      training = training)
        x = self.fc3(x)
        return x

# For use with the OpenGAN_pix approach, in which the GAN
# works in input space instead of feature space,
# this is our adaptation of the generator architecture
# used in GANDaLF by Oh et al. To our knowledge, it is the
# only architecture that has been shown for producing 1D network
# traffic, which seems like a good starting point.
#
# the size of the first dense layer, the reshaping operation
# before the convolutional layers, the number of input channels
# to the first convolutional layer, and the number of ouput
# channels from the final convolutional layer are all calculated
# based on the "input_shape" that the DFNet model (which will be
# the discriminator) expects
class GeneratorGANDaLF(nn.Module):
    def __init__(self, input_shape):
        super(GeneratorGANDaLF, self).__init__()
        self.input_shape = input_shape
        # Noise vector to initial dense layer
        self.dense1 = nn.Linear(100, input_shape[0] * int(input_shape[1] / 16))
        self.bn_dense1 = nn.BatchNorm1d(input_shape[0] * int(input_shape[1] / 16))
        self.dense2 = nn.Linear(input_shape[0] * int(input_shape[1] / 16), input_shape[0] * int(input_shape[1] / 16))
        self.bn_dense2 = nn.BatchNorm1d(input_shape[0] * int(input_shape[1] / 16))
        self.dense3 = nn.Linear(input_shape[0] * int(input_shape[1] / 16), input_shape[0] * int(input_shape[1] / 16))
        self.bn_dense3 = nn.BatchNorm1d(input_shape[0] * int(input_shape[1] / 16))
        
        
        # Define the sequential layers
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv1d(input_shape[0], 512, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(512)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv1d(512, 512, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(512)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Conv1d(512, 256, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(256)

        self.upsample4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = nn.Conv1d(256, 256, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(256)

        self.conv5 = nn.Conv1d(256, 128, kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm1d(128)

        self.conv6 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.bn6 = nn.BatchNorm1d(128)

        self.conv7 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        self.bn7 = nn.BatchNorm1d(64)

        self.conv8 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.bn8 = nn.BatchNorm1d(64)

        self.final_conv = nn.Conv1d(64, self.input_shape[0], kernel_size=5, padding=2)
        self.final_tanh = nn.Tanh()

    def forward(self, x, training):
        x = self.dense1(x)
        x = self.bn_dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        x = self.bn_dense2(x)
        x = F.relu(x)
        x = F.dropout(x, p = 0.5, training = training)
        x = self.dense2(x)
        x = self.bn_dense2(x)
        x = F.relu(x)
        x = F.dropout(x, p = 0.5, training = training)
        x = x.view(-1, self.input_shape[0], int(self.input_shape[1] / 16))
        
        # Applying upsample and convolutional layers
        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.upsample2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p = 0.3, training = training)
        
        x = self.upsample3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p = 0.3, training = training)
        
        x = self.upsample4(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.dropout(x, p = 0.3, training = training)
        
        # Regular convolution layers
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = F.dropout(x, p = 0.3, training = training)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = F.dropout(x, p = 0.3, training = training)
        
        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = F.dropout(x, p = 0.3, training = training)
        
        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)
        x = F.dropout(x, p = 0.3, training = training)
        
        # Final output layer
        x = self.final_conv(x)
        x = self.final_tanh(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, in_channels, hidden_layers, latent_channels):
        super(AutoEncoder, self).__init__()
        self.encode_layers = nn.ModuleList()
        self.decode_layers = nn.ModuleList()

        # Encoder layers
        current_channels = in_channels
        for hidden_channels in hidden_layers:
            self.encode_layers.append(nn.Conv1d(current_channels, hidden_channels, kernel_size=3, padding=1))
            self.encode_layers.append(nn.Tanh())
            current_channels = hidden_channels

        self.latent_layer = nn.Conv1d(current_channels, latent_channels, kernel_size=3, padding=1)

        # Decoder layers
        self.decode_layers.append(nn.Conv1d(latent_channels, current_channels, kernel_size=3, padding=1))
        self.decode_layers.append(nn.Tanh())

        for hidden_channels in reversed(hidden_layers):
            self.decode_layers.append(nn.Conv1d(current_channels, hidden_channels, kernel_size=3, padding=1))
            self.decode_layers.append(nn.Tanh())
            current_channels = hidden_channels

        self.output_layer = nn.Conv1d(current_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        for layer in self.encode_layers:
            x = layer(x)
        latent = self.latent_layer(x)
        x = latent
        for layer in self.decode_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

class CSSRClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_layers, latent_channels, gamma=1.0):
        super(CSSRClassifier, self).__init__()
        self.class_aes = nn.ModuleList([AutoEncoder(in_channels, hidden_layers, latent_channels) for _ in range(num_classes)])
        self.num_classes = num_classes
        self.gamma = gamma

    def ae_error(self, rc, x):
        return torch.norm(rc - x, p=1, dim=1, keepdim=True)

    def forward(self, x):
        cls_ers = []
        for ae in self.class_aes:
            reconstructed = ae(x)
            cls_er = self.ae_error(reconstructed, x)
            cls_ers.append(cls_er * self.gamma)

        cls_ers = torch.cat(cls_ers, dim=1)
        logits = -cls_ers
        pooled_logits = F.adaptive_avg_pool1d(logits, 1).view(x.size(0), self.num_classes)
        return pooled_logits

# this is just a sanity check for the baseline model
if __name__ == '__main__':
    CLASSES = 61
    INPUT_SHAPES = {'schuster8_tor': (1, 20 * 8), 'dschuster16_https': (2, 20 * 16)}
    BEST_HYPERPARAMETERS = {'schuster8_tor': {'filters': 256, 'kernel': 8, 'conv_stride': 1, 'pool': 8, 'pool_stride': 4, 'conv_dropout': 0.1, 'fc_neurons': 128, 'fc_init': 'he_normal', 'fc_activation': 'elu', 'fc_dropout': 0.1, 'lr': 7.191906601911815e-05, 'batch_size': 128},
                            'dschuster16_https': {'filters': 256, 'kernel': 4, 'conv_stride': 2, 'pool': 8, 'pool_stride': 1, 'conv_dropout': 0.4, 'fc_neurons': 1024, 'fc_init': 'glorot_uniform', 'fc_activation': 'relu', 'fc_dropout': 0.8, 'lr': 0.0005153393428807454, 'batch_size': 64}}
    def print_output_size(module, input, output):
        print(f"{module.__class__.__name__}: {output.shape}")

    for name in ['schuster8_tor', 'dschuster16_https']:
        model = DFNetTunable(INPUT_SHAPES[name], CLASSES, BEST_HYPERPARAMETERS[name])
        print('...DFNetTunable', name)
        print(model)
        for layer in model.modules():
            layer.register_forward_hook(print_output_size)
        model.eval()
        print('-----------------------------------')
        dummy_input = torch.rand(BEST_HYPERPARAMETERS[name]['batch_size'], *INPUT_SHAPES[name])
        print('...shape of dummy input to DFNetTunable')
        print(dummy_input.shape)
        dummy_features = model.extract_features_flattened(dummy_input)
        print('...shape of features from DFNetTunable.extract_features_flattened()')
        print(dummy_features.shape)
        discriminator = DiscriminatorDFNet_fea(INPUT_SHAPES[name], CLASSES, BEST_HYPERPARAMETERS[name])
        dummy_output = discriminator(dummy_features, training = False)
        print('...shape of output from DiscriminatorDFNet_fea')
        print(dummy_output.shape)
#        print('-----------------------------------')
#        generator = GeneratorGANDaLF(INPUT_SHAPES[name])
#        noise = torch.randn(BEST_HYPERPARAMETERS[name]['batch_size'], 100)
#        print('...shape of noise going into GeneratorGANDaLF')
#        print(noise.shape)
#        generated_data = generator(noise, training = False)
#        print('...shape of generated data')
#        print(generated_data.shape)
#        output = model(generated_data, training = False)
#        print('...shape of output of DFNet model(generated data)')
#        print(output.shape)
