import tensorflow as tf
import logging
import sys

# Setup logging
L = logging.getLogger(__name__)

# Reference the module root
reflexive_link = sys.modules[__name__.split(".")[0]]


def TinyMobileNetv2(input_shape, output_classes, soft_fail=False):
    try:
        # with tf.device('/gpu:0'):
        input_layer = tf.keras.Input(input_shape)
        aug_layer = reflexive_link.neural.layers.InputAugmentations.DataAugmentation()(
            input_layer
        )

        mnetv2_layer = tf.keras.applications.MobileNetV2(
            input_shape,
            alpha=0.35,
            classes=output_classes,
            pooling="max",
            weights="imagenet",
            classifier_activation="sigmoid",
        )(aug_layer)
        mnetv2_layer._name = "MobileNetV2_PdfTransferLearn"

        pdf_net = tf.keras.Model(inputs=[input_layer], outputs=[mnetv2_layer,],)

    except Exception as e:
        L.error(f"{e}", exc_info=True)
        if soft_fail:
            return locals()
        else:
            raise
    return pdf_net


def MobileNetV3(input_shape, output_classes, dropout: float = 0.05, soft_fail=False):
    try:
        # with tf.device('/gpu:0'):
        input_layer = tf.keras.Input(input_shape)
        aug_layer = reflexive_link.neural.layers.InputAugmentations.DataAugmentation()(
            input_layer
        )

        tmp_mod = tf.keras.applications.MobileNetV3Large(weights="imagenet")
        mnetv3_layer = tf.keras.Model(
            inputs=[tmp_mod.input], outputs=[tmp_mod.layers[-3].output]
        )(aug_layer)
        mnetv3_layer._name = "MobileNetV3_PdfTransferLearn"
        x = tf.keras.layers.Dense(2 ** 9, activation="relu")(mnetv3_layer)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(2 ** 8, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(2 ** 7, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(output_classes, activation="sigmoid")(x)
        pdf_net = tf.keras.Model(inputs=[input_layer], outputs=[x],)

    except Exception as e:
        L.error(f"{e}", exc_info=True)
        if soft_fail:
            return locals()
        else:
            raise
    return pdf_net
