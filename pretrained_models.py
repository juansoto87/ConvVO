import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow import keras
from keras import layers
from keras.layers import Dense, Dropout, Activation


class Custom_MV2:
    def __init__(self, img_shape):
        self.img_shape = img_shape

    def conv_block(self, input_tensor, c, s, t, expand=True):
        """
        Convolutional Block for mobile net v2
        Args:
            input_tensor (keras tensor): input tensor
            c (int): output channels
            s (int): stride size of first layer in the series
            t (int): expansion factor
            expand (bool): expand filters or not?

        Returns: keras tensor
        """
        first_conv_channels = input_tensor.get_shape()[-1]
        if expand:
            x = layers.Conv2D(
                first_conv_channels * t,
                1,
                1,
                padding='same',
                use_bias=False
            )(input_tensor)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU(6.0)(x)
        else:
            x = input_tensor

        x = layers.DepthwiseConv2D(
            3,
            s,
            'same',
            1,
            use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)

        x = layers.Conv2D(
            c,
            1,
            1,
            padding='same',
            use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)

        if input_tensor.get_shape() == x.get_shape() and s == 1:
            return x + input_tensor

        return x

    def splitted_model(self):

        input = layers.Input(shape=self.img_shape)

        x = layers.Conv2D(
            32,
            3,
            2,
            padding='same',
            use_bias=False
        )(input)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)

        x = self.conv_block(x, 16, 1, 1, expand=False)
        x = self.conv_block(x, 24, 2, 6)
        x = self.conv_block(x, 24, 1, 6)

        x = self.conv_block(x, 32, 2, 6)
        x = self.conv_block(x, 32, 1, 6)
        x = self.conv_block(x, 32, 1, 6)

        x = self.conv_block(x, 64, 2, 6)
        x = self.conv_block(x, 64, 1, 6)
        x = self.conv_block(x, 64, 1, 6)
        x = self.conv_block(x, 64, 1, 6)

        model_f = Model(inputs=input, outputs=x)

        input_2 = layers.Input(shape=(x.shape[1:]))
        x = self.conv_block(input_2, 96, 1, 6)
        x = self.conv_block(x, 96, 1, 6)
        x = self.conv_block(x, 96, 1, 6)

        x = self.conv_block(x, 160, 2, 6)
        x = self.conv_block(x, 160, 1, 6)
        x = self.conv_block(x, 160, 1, 6)

        x = self.conv_block(x, 320, 1, 6)

        x = layers.Conv2D(
            1280,
            1,
            1,
            padding='same',
            use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)

        x = layers.GlobalAveragePooling2D()(x)

        model_h = Model(inputs=input_2, outputs=x)

        return model_f, model_h

    def get_model(self):
        IMG_SHAPE = self.img_shape

        model_f, model_h = self.splitted_model()

        mobile_net = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')
        mobile_net.trainable = False
        layer_f_counter = 0
        layer_h_counter = 0

        for i in range(len(mobile_net.layers)):
            if layer_f_counter < len(model_f.layers):
                if len(mobile_net.layers[i].get_weights()) > 0:
                    if len(model_f.layers[layer_f_counter].get_weights()) > 0:
                        print(mobile_net.layers[i].name, 'here', model_f.layers[layer_f_counter].name, layer_f_counter)
                        model_f.layers[layer_f_counter].set_weights(mobile_net.layers[i].get_weights())
                    layer_f_counter += 1
                    print(layer_f_counter)
                else:
                    if len(model_f.layers[layer_f_counter].get_weights()) > 0:
                        continue
                    else:
                        layer_f_counter += 1

            else:
                if layer_h_counter < len(model_h.layers):
                    if len(mobile_net.layers[i].get_weights()) > 0:
                        if len(model_h.layers[layer_h_counter].get_weights()) > 0:
                            print(mobile_net.layers[i].name, 'here', model_h.layers[layer_h_counter].name,
                                  layer_h_counter)
                            model_h.layers[layer_h_counter].set_weights(mobile_net.layers[i].get_weights())
                        layer_h_counter += 1
                        print(layer_h_counter)
                    else:
                        if len(model_h.layers[layer_h_counter].get_weights()) > 0:
                            continue
                        else:
                            layer_h_counter += 1
        model_f.trainable = False

        return model_f, model_h

class Custom_ResNet:
    def __init__(self, img_shape):
        self.img_shape = img_shape

    def conv_block(self, input_tensor, c, s, t, expand=True):
        """
        Convolutional Block for mobile net v2
        Args:
            input_tensor (keras tensor): input tensor
            c (int): output channels
            s (int): stride size of first layer in the series
            t (int): expansion factor
            expand (bool): expand filters or not?

        Returns: keras tensor
        """
        first_conv_channels = input_tensor.get_shape()[-1]
        if expand:
            x = layers.Conv2D(
                first_conv_channels * t,
                1,
                1,
                padding='same',
                use_bias=False
            )(input_tensor)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU(6.0)(x)
        else:
            x = input_tensor

        x = layers.DepthwiseConv2D(
            3,
            s,
            'same',
            1,
            use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)

        x = layers.Conv2D(
            c,
            1,
            1,
            padding='same',
            use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)

        if input_tensor.get_shape() == x.get_shape() and s == 1:
            return x + input_tensor

        return x


    def get_pre_model(self):

        model_Res = tf.keras.applications.ResNet101(include_top=False, weights="imagenet",
                                input_tensor=None, input_shape=(self.img_shape[0], self.img_shape[1], 3))
        model_Res.trainable = False

        layer_outputs = []
        layer_names = []
        for layer in model_Res.layers:
            if isinstance(layer, (layers.Conv2D, layers.MaxPooling2D, layers.BatchNormalization)):
                layer_outputs.append(layer.output)
                layer_names.append(layer.name)
                activation_model = keras.Model(inputs=model_Res.input, outputs=layer_outputs)

        activation_model.trainable = False



        return activation_model




