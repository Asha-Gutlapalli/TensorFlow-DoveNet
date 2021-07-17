import tensorflow as tf
from keras.layers.core import Lambda
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Flatten, Input, Conv2D, LeakyReLU, BatchNormalization, ReLU, Conv2DTranspose, Dropout, MaxPooling2D, UpSampling2D, add, multiply, InputSpec


# Generator Adversarial Network

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Generator

# UNet with Attention

def attention_up_and_concate(down_layer, layer):
    in_channel = down_layer.get_shape().as_list()[3]
    up = UpSampling2D(size=(2, 2))(down_layer)
    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4)
    my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
    concate = my_concat([up, layer])

    return concate


def attention_block_2d(x, g, inter_channel):
    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1])(x)
    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1])(g)
    f = Activation('relu')(add([theta_x, phi_g]))
    psi_f = Conv2D(1, [1, 1], strides=[1, 1])(f)
    rate = Activation('sigmoid')(psi_f)
    att_x = multiply([x, rate])

    return att_x

def att_unet(img_w, img_h, n_label):
    inputs = Input((img_w, img_h, 4))
    x = inputs
    depth = 4
    features = 64
    skips = []
    for i in range(depth):
        x = Conv2D(features, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same')(x)
        skips.append(x)
        x = MaxPooling2D((2, 2))(x)
        features = features * 2

    x = Conv2D(features, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(features, (3, 3), activation='relu', padding='same')(x)

    for i in reversed(range(depth)):
        features = features // 2
        x = attention_up_and_concate(x, skips[i])
        x = Conv2D(features, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same')(x)

    x = Conv2D(n_label, (1, 1), padding='same')(x)

    model = Model(inputs=inputs, outputs=x)

    return model

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Discriminator

#Partial Convolution

class PConv2D(Conv2D):
    def __init__(self, *args, n_channels=3, mono=False, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        """Adapted from original _Conv() layer of Keras
        param input_shape: list of dimensions for [img, mask]
        """

        channel_axis = -1

        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')

        self.input_dim = input_shape[channel_axis]

        kernel_shape = self.kernel_size + (self.input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='img_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.kernel_mask = K.ones(shape=self.kernel_size + (self.input_dim, self.filters))

        self.pconv_padding = (
            (int((self.kernel_size[0]-1)/2), int((self.kernel_size[0]-1)/2)),
            (int((self.kernel_size[0]-1)/2), int((self.kernel_size[0]-1)/2)),
        )

        self.window_size = self.kernel_size[0] * self.kernel_size[1]

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs, mask=None):

        if len(inputs) != 2:
            mask = tf.ones((inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]))

        images = K.spatial_2d_padding(inputs, self.pconv_padding, self.data_format)
        masks = K.spatial_2d_padding(mask, self.pconv_padding, self.data_format)

        mask_output = K.conv2d(
            masks, self.kernel_mask,
            strides=self.strides,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        img_output = K.conv2d(
            (images*masks), self.kernel,
            strides=self.strides,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        mask_ratio = self.window_size / (mask_output + 1e-8)
        mask_output = K.clip(mask_output, 0, 1)
        mask_ratio = mask_ratio * mask_output

        img_output = img_output * mask_ratio

        if self.use_bias:
            img_output = K.bias_add(
                img_output,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            img_output = self.activation(img_output)

        return [img_output, mask_output]


# Domain Verification/Local Discriminator

class OrgDiscriminator(tf.keras.Model):
  def __init__(self):
        super(OrgDiscriminator, self).__init__()

        self.conv1 = PConv2D(filters=24, kernel_size=3, strides=2, padding='valid')
        self.relu1 = LeakyReLU(alpha=0.2)
        self.conv2 = PConv2D(filters=48, kernel_size=3, strides=2, padding='valid')
        self.norm2 = BatchNormalization()
        self.relu2 = LeakyReLU(alpha=0.2)
        self.conv3 = PConv2D(filters=64, kernel_size=3, strides=2, padding='valid')
        self.norm3 = BatchNormalization()
        self.relu3 = LeakyReLU(alpha=0.2)
        self.conv4 = PConv2D(filters=64, kernel_size=3, strides=2, padding='valid')
        self.norm4 = BatchNormalization()
        self.relu4 = LeakyReLU(alpha=0.2)
        self.conv5 = PConv2D(filters=128, kernel_size=3, strides=2, padding='valid')
        self.norm5 = BatchNormalization()
        self.relu5 = LeakyReLU(alpha=0.2)
        self.conv6 = PConv2D(filters=128, kernel_size=3, strides=2, padding='valid')

  def __call__(self, inputs, mask=None):

        x = inputs
        x, _ = self.conv1(x)
        x = self.relu1(x)
        x, _ = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x, _ = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        x, _ = self.conv4(x)
        x = self.norm4(x)
        x = self.relu4(x)
        x, _ = self.conv5(x)
        x = self.norm5(x)
        x = self.relu5(x)
        x, _ = self.conv6(x)

        xf, xb = inputs, inputs
        mf, mb = mask, 1 - mask

        xf, mf = self.conv1(xf, mf)
        xf = self.relu1(xf)
        xf, mf = self.conv2(xf, mf)
        xf = self.norm2(xf)
        xf = self.relu2(xf)
        xf, mf = self.conv3(xf, mf)
        xf = self.norm3(xf)
        xf = self.relu3(xf)
        xf, mf = self.conv4(xf, mf)
        xf = self.norm4(xf)
        xf = self.relu4(xf)
        xf, mf = self.conv5(xf, mf)
        xf = self.norm5(xf)
        xf = self.relu5(xf)
        xf, mf = self.conv6(xf, mf)

        xb, mb = self.conv1(xb, mb)
        xb = self.relu1(xb)
        xb, mb = self.conv2(xb, mb)
        xb = self.norm2(xb)
        xb = self.relu2(xb)
        xb, mb = self.conv3(xb, mb)
        xb = self.norm3(xb)
        xb = self.relu3(xb)
        xb, mb = self.conv4(xb, mb)
        xb = self.norm4(xb)
        xb = self.relu4(xb)
        xb, mb = self.conv5(xb, mb)
        xb = self.norm5(xb)
        xb = self.relu5(xb)
        xb, mb = self.conv6(xb, mb)

        return x, xf, xb


# Global Discriminator

class NLayerDiscriminator(tf.keras.Model):

    def __init__(self):
        super(NLayerDiscriminator, self).__init__()

        self.D = OrgDiscriminator()
        self.convl1 = Conv2D(256, kernel_size=1, strides=1)
        self.relul1 = LeakyReLU(0.2)
        self.convl2 = Conv2D(256, kernel_size=1, strides=1)
        self.relul2 = LeakyReLU(0.2)
        self.convl3 = Conv2D(1, kernel_size=1, strides=1)
        self.convg3 = Conv2D(1, kernel_size=1, strides=1)

    def __call__(self, inputs, mask=None, gp=False, feat_loss=False):

        x, xf, xb = self.D(inputs, mask)
        feat_l, feat_g = tf.concat([xf, xb], axis = 0), x
        x = self.convg3(x)

        sim = xf * xb
        sim = self.convl1(sim)
        sim = self.relul1(sim)
        sim = self.convl2(sim)
        sim = self.relul2(sim)
        sim = self.convl3(sim)
        sim_sum = sim
        if not gp:
            if feat_loss:
                return x, sim_sum, feat_g, feat_l
            return x, sim_sum
        return  (x + sim_sum) * 0.5

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
