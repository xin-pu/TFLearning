import tensorflow as tf
from tensorflow.keras.layers import *

# 下采样
from tensorflow.python.keras import Model
from tensorflow.python.keras.regularizers import l2


def Conv(*args, **kwargs):
    new_kwargs = {'kernel_regularizer': l2(5e-4),
                  'padding': 'valid' if kwargs.get('strides') == (2, 2) else 'same'}
    new_kwargs.update(kwargs)

    return Conv2D(*args, **new_kwargs)


def CBL(x, *args, **kwargs):
    new_kwargs = {'use_bias': False}
    new_kwargs.update(kwargs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv(*args, **new_kwargs)(x)
    return x


def PCBL(x, num_filters):
    x = MaxPooling2D()(x)
    x = CBL(x, num_filters, (3, 3))
    return x


def CBLR(x, num_filters):
    y = CBL(x, num_filters, (1, 1))
    y = CBL(y, num_filters * 2, (3, 3))
    x = Add()([x, y])

    return x


def CBLU(x, num_filters):
    x = CBL(x, num_filters, (1, 1))
    x = UpSampling2D(2)(x)

    return x


def CBLC(x, num_filters, out_filers):
    x = CBL(x, num_filters * 2, (3, 3))
    x = Conv(out_filers, (1, 1))(x)

    return x


def CBL5(x, num_filters):
    x = CBL(x, num_filters, (1, 1))
    x = CBL(x, num_filters * 2, (3, 3))
    x = CBL(x, num_filters, (1, 1))
    x = CBL(x, num_filters * 2, (3, 3))
    x = CBL(x, num_filters, (1, 1))

    return x


def get_body(model_input, out_unit):
    x = model_input

    n = [4, 3, 2, 1]
    for i in range(len(n)):
        x = PCBL(x, 2 ** (n[i] + 1))
        if i < 4:
            x = CBLR(x, 2 ** (n[i]))

    x = Flatten()(x)

    x = Dense(units=36,
              activation=tf.nn.relu,
              kernel_regularizer=l2(5e-4))(x)

    x = Dense(units=out_unit)(x)

    return x


def get_net(input_shape, out_unit=2):
    model_input = Input(shape=(*input_shape, 1), dtype=float)
    model_output = get_body(model_input, out_unit)
    model = Model(inputs=model_input, outputs=model_output)
    return model


if __name__ == "__main__":
    test_x = tf.ones([1, 416, 416, 1])
    net = get_net((416, 416), 4)
    test_y = net(test_x)
    net.summary()
    print(test_y.shape)
    print(test_y)
