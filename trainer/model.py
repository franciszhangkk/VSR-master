import keras
from keras.models import Model, Sequential
from keras.layers.core import Layer
from keras.layers import Conv2D, LeakyReLU, MaxPooling2D, Lambda, Input, Concatenate, Activation, Flatten, Dense, Add, \
    Conv2DTranspose
import tensorflow as tf
from keras import backend as K
import numpy as np


### FLOWNETWORK
def UpSampling2DBilinear(**kwargs):
    def layer(x):
        input_shape = K.int_shape(x)
        output_shape = (2 * input_shape[1], 2 * input_shape[2])
        return tf.image.resize_bilinear(x, output_shape, align_corners=True)

    return Lambda(layer, **kwargs)


class FlowNetwork:
    def __init__(self, input_shape):
        inputs = Input(shape=input_shape)

        x = Conv2D(32, (3, 3), padding="same")(inputs)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = MaxPooling2D()(x)

        x = Conv2D(64, (3, 3), padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = MaxPooling2D()(x)

        x = Conv2D(128, (3, 3), padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(128, (3, 3), padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = MaxPooling2D()(x)

        x = Conv2D(256, (3, 3), padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(256, (3, 3), padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = UpSampling2DBilinear()(x)

        x = Conv2D(128, (3, 3), padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(128, (3, 3), padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = UpSampling2DBilinear()(x)

        x = Conv2D(64, (3, 3), padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = UpSampling2DBilinear()(x)

        x = Conv2D(32, (3, 3), padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        outputs = Conv2D(2, (3, 3), activation='tanh', padding="same")(x)

        self.model = Model(input=inputs, output=outputs)

    # FLOWNETWORK END


# SUPERESOLUTIONNETWORK
class SuperResolutionNetwork:
    def __init__(self, input_shape):
        inputs = Input(shape=input_shape)

        x = Conv2D(64, (3, 3), activation='relu', padding="same")(inputs)

        for _ in range(10):
            y = Conv2D(64, (3, 3), activation='relu', padding="same")(x)
            y = Conv2D(64, (3, 3), padding="same")(y)
            x = Add()([x, y])

        x = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding="same")(x)

        outputs = Conv2D(3, (3, 3), padding="same")(x)

        self.model = Model(inputs=inputs, outputs=outputs)


class LocalizationNetwork:
    def __init__(self, input_shape, init_weights):
        self.model = Sequential()
        input_shape = (512, 512, 5)
        self.model.add(MaxPooling2D(pool_size=(2, 2), input_shape=input_shape))
        self.model.add(Conv2D(20, (5, 5)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(20, (5, 5)))

        self.model.add(Flatten())
        self.model.add(Dense(50))
        self.model.add(Activation('relu'))
        self.model.add(Dense(6, weights=init_weights))
        # model.add(Activation('sigmoid'))


class SpatialTransformer(Layer):
    """Spatial Transformer Layer
    Implements a spatial transformer layer as described in [1]_.
    Borrowed from [2]_:
    downsample_fator : float
        A value of 1 will keep the orignal size of the image.
        Values larger than 1 will down sample the image. Values below 1 will
        upsample the image.
        example image: height= 100, width = 200
        downsample_factor = 2
        output image will then be 50, 100
    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

    .. [3]  https://github.com/EderSantana/seya/blob/keras1/seya/layers/attention.py
    """

    def __init__(self,
                 localization_net,
                 output_size,
                 **kwargs):
        self.locnet = localization_net
        self.output_size = output_size
        super(SpatialTransformer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.locnet.build(input_shape)
        self.trainable_weights = self.locnet.trainable_weights
        # self.constraints = self.locnet.constraints

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        return (None,
                int(output_size[0]),
                int(output_size[1]),
                int(input_shape[-1]))

    def call(self, X, mask=None):
        affine_transformation = self.locnet.call(X)
        output = self._transform(affine_transformation, X, self.output_size)
        return output

    def _repeat(self, x, num_repeats):
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1, 1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    def _interpolate(self, image, x, y, output_size):
        batch_size = tf.shape(image)[0]
        height = tf.shape(image)[1]
        width = tf.shape(image)[2]
        num_channels = tf.shape(image)[3]

        x = tf.cast(x, dtype='float32')
        y = tf.cast(y, dtype='float32')

        height_float = tf.cast(height, dtype='float32')
        width_float = tf.cast(width, dtype='float32')

        output_height = output_size[0]
        output_width = output_size[1]

        x = .5 * (x + 1.0) * (width_float)
        y = .5 * (y + 1.0) * (height_float)

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        max_y = tf.cast(height - 1, dtype='int32')
        max_x = tf.cast(width - 1, dtype='int32')
        zero = tf.zeros([], dtype='int32')

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        flat_image_dimensions = width * height
        pixels_batch = tf.range(batch_size) * flat_image_dimensions
        flat_output_dimensions = output_height * output_width
        base = self._repeat(pixels_batch, flat_output_dimensions)
        base_y0 = base + y0 * width
        base_y1 = base + y1 * width
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = tf.reshape(image, shape=(-1, num_channels))
        flat_image = tf.cast(flat_image, dtype='float32')
        pixel_values_a = tf.gather(flat_image, indices_a)
        pixel_values_b = tf.gather(flat_image, indices_b)
        pixel_values_c = tf.gather(flat_image, indices_c)
        pixel_values_d = tf.gather(flat_image, indices_d)

        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)
        output = tf.add_n([area_a * pixel_values_a,
                           area_b * pixel_values_b,
                           area_c * pixel_values_c,
                           area_d * pixel_values_d])
        return output

    def _meshgrid(self, height, width):
        x_linspace = tf.linspace(-1., 1., width)
        y_linspace = tf.linspace(-1., 1., height)
        x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
        x_coordinates = tf.reshape(x_coordinates, [-1])
        y_coordinates = tf.reshape(y_coordinates, [-1])
        ones = tf.ones_like(x_coordinates)
        indices_grid = tf.concat([x_coordinates, y_coordinates, ones], 0)
        return indices_grid

    def _transform(self, affine_transformation, input_shape, output_size):
        batch_size = tf.shape(input_shape)[0]
        height = tf.shape(input_shape)[1]
        width = tf.shape(input_shape)[2]
        num_channels = tf.shape(input_shape)[3]

        affine_transformation = tf.reshape(affine_transformation, shape=(batch_size, 2, 3))

        affine_transformation = tf.reshape(affine_transformation, (-1, 2, 3))
        affine_transformation = tf.cast(affine_transformation, 'float32')

        width = tf.cast(width, dtype='float32')
        height = tf.cast(height, dtype='float32')
        output_height = output_size[0]
        output_width = output_size[1]
        indices_grid = self._meshgrid(output_height, output_width)
        indices_grid = tf.expand_dims(indices_grid, 0)
        indices_grid = tf.reshape(indices_grid, [-1])  # flatten?

        indices_grid = tf.tile(indices_grid, tf.stack([batch_size]))
        indices_grid = tf.reshape(indices_grid, (batch_size, 3, -1))

        transformed_grid = tf.matmul(affine_transformation, indices_grid)
        x_s = tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])
        x_s_flatten = tf.reshape(x_s, [-1])
        y_s_flatten = tf.reshape(y_s, [-1])

        transformed_image = self._interpolate(input_shape,
                                              x_s_flatten,
                                              y_s_flatten,
                                              output_size)

        transformed_image = tf.reshape(transformed_image, shape=(batch_size,
                                                                 output_height,
                                                                 output_width,
                                                                 num_channels))
        return transformed_image


# SUPERESOLUTIONNETWORK END

def Warp(**kwargs):
    def layer(x):
        b = np.zeros((2, 3), dtype='float32')
        b[0, 0] = 1
        b[1, 1] = 1
        W = np.zeros((50, 6), dtype='float32')
        weights = [W, b.flatten()]
        input_shape = K.int_shape(x)
        locnet = LocalizationNetwork(input_shape, weights)
        return SpatialTransformer(localization_net=locnet.model, output_size=(input_shape[1], input_shape[2]))(x)

    return Lambda(layer, **kwargs)


def SpaceToDepth(**kwargs):
    def layer(x):
        return tf.space_to_depth(x, 2)

    return Lambda(layer, **kwargs)


class FRVSR_Layer:
    def __init__(self, low_res_shape, high_res_shape):
        flow_network_shape = list(low_res_shape)
        flow_network_shape[-1] = low_res_shape[-1] + high_res_shape[-1]
        flow_network_shape = tuple(flow_network_shape)
        # the flow map is concatenate shape

        flow_network = FlowNetwork(flow_network_shape)

        super_resolution_network_shape = list(low_res_shape)
        super_resolution_network_shape[-1] = low_res_shape[-1] + high_res_shape[-1] * 4
        # The reason of *4 is space to depth
        super_resolution_network_shape = tuple(super_resolution_network_shape)

        super_resolution_network = SuperResolutionNetwork(super_resolution_network_shape)

        low_res_inputs = Input(shape=low_res_shape)
        low_res_previous_inputs = Input(shape=low_res_shape)
        previous_frame_input = Input(shape=high_res_shape)

        flow_network_input = Concatenate()([low_res_previous_inputs, low_res_inputs])
        flow = flow_network.model(flow_network_input)
        flow_high_resolution = UpSampling2DBilinear()(flow)
        warp_input = Concatenate()([previous_frame_input, flow_high_resolution])

        warped = Warp()(warp_input)
        warped = Lambda(lambda x: x[:, :, :, :3])(warped)

        warped_to_depth = SpaceToDepth()(warped)
        super_resolution_network_input = Concatenate()([low_res_inputs, warped_to_depth])
        output = super_resolution_network.model(super_resolution_network_input)

        self.model = Model(inputs=[low_res_inputs, low_res_previous_inputs, previous_frame_input], outputs=[output])


class FRVSR_model:
    def __init__(self, low_res_shape, high_res_shape):
        sr_layer = FRVSR_Layer(low_res_shape, high_res_shape)

        low_res_black = Input(shape=low_res_shape)
        high_res_black = Input(shape=high_res_shape)

        lr_input1 = Input(shape=low_res_shape)
        lr_input2 = Input(shape=low_res_shape)
        lr_input3 = Input(shape=low_res_shape)
        lr_input4 = Input(shape=low_res_shape)
        lr_input5 = Input(shape=low_res_shape)

        hr_output1 = sr_layer.model([lr_input1, lr_input1, high_res_black])
        hr_output2 = sr_layer.model([lr_input2, lr_input1, hr_output1])
        hr_output3 = sr_layer.model([lr_input3, lr_input2, hr_output2])
        hr_output4 = sr_layer.model([lr_input4, lr_input3, hr_output3])
        hr_output5 = sr_layer.model([lr_input5, lr_input5, hr_output4])


        self.model = Model(inputs=[lr_input1, lr_input2, lr_input3, lr_input4, lr_input5,low_res_black, high_res_black ],
                           outputs=[hr_output1, hr_output2, hr_output3, hr_output4, hr_output5])
