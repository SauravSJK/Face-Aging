import numpy as np
import tensorflow as tf
from tensorflow.nn import moments
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.nn import batch_normalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.initializers import Ones
from tensorflow.keras.initializers import Zeros
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow_addons.layers import InstanceNormalization


# Ref: https://colab.research.google.com/drive/1WGG8d22KoxXWBThYOeFDcHvt_z9EirHV#scrollTo=-CxyRhZaDSYk
class ConditionBatchNormalization(Layer):
    def __init__(self):
        super(ConditionBatchNormalization, self).__init__()
        self.decay = 0.9
        self.epsilon = 1e-05

    def build(self, input_shape):
        self.num_channels = input_shape[0][-1]
        self.beta_mapping = Dense(self.num_channels)
        self.gamma_mapping = Dense(self.num_channels)
        self.test_mean = tf.Variable(initial_value=Zeros()(self.num_channels), trainable=False, dtype=tf.float32)
        self.test_var = tf.Variable(initial_value=Ones()(self.num_channels), trainable=False, dtype=tf.float32)

    def call(self, x, training=None):
        #Generate beta, gamma
        x, conditions = x
        beta = self.beta_mapping(conditions)
        gamma = self.gamma_mapping(conditions)

        beta = tf.reshape(beta, shape=[-1, 1, 1, self.num_channels])
        gamma = tf.reshape(gamma, shape=[-1, 1, 1, self.num_channels])
        if training:
            #Calculate mean and varience of X.
            batch_mean, batch_var = moments(x, [0, 1, 2])
            #Calculate parameters for test set
            test_mean = self.test_mean * self.decay + batch_mean * (1 - self.decay)
            test_var = self.test_var * self.decay + batch_var * (1 - self.decay)

            def mean_update():
                self.test_mean.assign(test_mean)

            def variance_update():
                self.test_var.assign(test_var)

            self.add_update(mean_update)
            self.add_update(variance_update)

            return batch_normalization(x, batch_mean, batch_var, beta, gamma, self.epsilon)
        else:
            return batch_normalization(x, self.test_mean, self.test_var, beta, gamma, self.epsilon)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.num_channels})
        return config


# Ref: https://github.com/ariG23498/AdaIN-TF/blob/master/AdaIN.ipynb
def ada_in(style, content, epsilon=1e-5):
    axes = [1, 2]

    c_mean, c_var = moments(content, axes=axes, keepdims=True)
    s_mean, s_var = moments(style, axes=axes, keepdims=True)
    c_std, s_std = tf.sqrt(c_var + epsilon), tf.sqrt(s_var + epsilon)

    t = s_std * (content - c_mean) / c_std + s_mean
    return t


def res_block(x, filters, kernelsize=4, sampling=None, sampling_size=2, norm_type=None, condition=None):
    if norm_type == "instance":
        y = InstanceNormalization()(x)
    elif norm_type == "cbn":
        y = ConditionBatchNormalization()([x, condition])
    elif norm_type == "ada_in":
        condition = Conv2D(x.shape[3], 1, padding="same")(condition)
        y = ada_in(condition, x)
    else:
        y = x
    y = LeakyReLU()(y)
    y = Conv2D(filters, kernelsize, padding="same")(y)
    if norm_type == "instance":
        y = InstanceNormalization()(y)
    elif norm_type == "cbn":
        y = ConditionBatchNormalization()([x, condition])
    elif norm_type == "ada_in":
        y = ada_in(condition, x)
    else:
        y = x
    y = LeakyReLU()(y)
    y = Conv2D(filters, kernelsize, padding="same")(y)
    z = Conv2D(filters, 1, padding="same")(x)
    out = y + z
    if sampling == "down":
        return AveragePooling2D(pool_size=sampling_size)(out)
    elif sampling == "up":
        return UpSampling2D(size=sampling_size)(out)
    else:
        return out


# Ref: https://stackoverflow.com/questions/64556120/early-stopping-with-multiple-conditions
class CustomEarlyStopping(Callback):
    def __init__(self, patience=0):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best_identity_loss = np.Inf
        self.best_age_prediction_loss = np.Inf
        self.best_reconstruction_loss = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        identity_loss = logs.get("identity_loss")
        age_prediction_loss = logs.get("reconstruction_loss")
        reconstruction_loss = logs.get("reconstruction_loss")

        # If BOTH the validation loss AND map10 does not improve for 'patience' epochs, stop training early.
        if np.less(identity_loss, self.best_identity_loss) and\
            np.less(age_prediction_loss, self.best_age_prediction_loss) and\
                np.less(reconstruction_loss, self.best_reconstruction_loss):
            self.best_identity_loss = identity_loss
            self.best_age_prediction_loss = age_prediction_loss
            self.best_reconstruction_loss = reconstruction_loss
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
