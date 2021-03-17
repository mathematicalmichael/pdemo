# -*- coding: utf-8 -*-
"""This script contains a few functions to facilitate saving and loading of
the model and any input transformations which are applied to the input
data"""
###############################################################################
# Imports
import logging
import tensorflow as tf
import pandas as pd
import typing as T
import sklearn.preprocessing as pp
import sys
import numpy as np
import pathlib
import PIL.Image as I
import typing as T
import matplotlib.pyplot as plt
import uuid

# Special import for module root
M = sys.modules[__name__.split(".")[0]]
###############################################################################
# Setup Logging
L = logging.getLogger(__name__)

###############################################################################
# Setup Locations
_ROOT = pathlib.Path(__file__).parent.absolute()
_MODEL_H5 = _ROOT / "pdf_net.h5"
_LABEL_ENCODING_PARQUET_FILE = _ROOT / "label_positional_encoding.parquet"
###############################################################################
# Setup Constants
_MLB_INDEX_COL_NAME = "label_index"
_MLB_LABEL_COL_NAME = "label_text"
###############################################################################
# Setup Ops


def load_model(origional=False, enforce_cpu=False):
    """Wrapper around tf.keras.models.load_model.

    Wrapped functionality: custom objects needed by the model, e.g. the data
    augmentation layer which performs random translations to the input data,
    only relevant for training and could be entirely ommited if desired.

    Returns:
        Tensorflow Keras Functional Model
    """
    if origional:
        load_from = str(_MODEL_H5) + ".origional"
    else:
        load_from = str(_MODEL_H5)
    if enforce_cpu:
        with tf.device("/CPU:0"):
            model = tf.keras.models.load_model(
                load_from,
                custom_objects={
                    "DataAugmentation": M.neural.layers.InputAugmentations.DataAugmentation
                },
            )
    else:
        model = tf.keras.models.load_model(
            load_from,
            custom_objects={
                "DataAugmentation": M.neural.layers.InputAugmentations.DataAugmentation
            },
        )
    return model


def load_label_encoder(
    index_col: str = "label_index", label_col: str = "label_text", origional=False
) -> pp.MultiLabelBinarizer:
    """This function loads config file for the label encoder

    Operations:
        1) load parquet
        2) check for the two columns defined as args to this function
        3) sort by the index column, ascending
        4) build MultiLabelBinarizer based on the label column's values
        5) call mlb's fit function to prime it for use
        6) return primed mlb for use to .transform data
    Args:
        index_col (str): default 'label_index' - column to look for in the
            parquet file which has the index information
        label_col (str): default 'label_text' - column to look for in the
            parquet file which has the literal labels
        origional (bool): default false, load the origional model
    Returns:
        sklearn.preprocessing.MultiLabelBinarizer
    """
    if origional:
        load_from = str(_LABEL_ENCODING_PARQUET_FILE) + ".origional"
    else:
        load_from = str(_LABEL_ENCODING_PARQUET_FILE)
    labs = pd.read_parquet(load_from)
    assert (
        index_col in labs
    ), f"{index_col} not in encoding file {_LABEL_ENCODING_PARQUET_FILE}"
    assert (
        label_col in labs
    ), f"{label_col} not in encoding file {_LABEL_ENCODING_PARQUET_FILE}"
    labs.sort_values(index_col, inplace=True)
    mlb = pp.MultiLabelBinarizer(classes=labs[label_col].tolist())
    mlb.fit([])
    return mlb


class SetupTrainingParticulars:
    """This class assembles boilerplate objects for training a neural net"""

    save_model_to = _MODEL_H5
    save_label_encoder_to = _LABEL_ENCODING_PARQUET_FILE
    multilabelbiniarizer_index_col_name = _MLB_INDEX_COL_NAME
    multilabelbinarizer_label_column = _MLB_LABEL_COL_NAME

    def __init__(self, df, size=[256, 256, 3], modality="v3"):
        """This function initalizes all of the particulars for a training op.

        At a high level, this function takes a dataframe and removes
        looks at rows that don't have erros, extracts the labels
        from a column called "pdf_label" and creates a "target" instance
        attribute which is a multilabel binarizer encoded on the "pdf_label"
        column. From that target attribute, another attribute is created
        which is the invese of the sample weight for all examples from
        each pdf. This is done to ensure equal performance for each of
        the input pdfs. The model is a TinyMobileNetv2 (alpha=.35) which
        has been chosen due to its low parameter count, quick to train, and
        existing weights to transfer-learn & help train our pdf neural net.

        The origional MobileNetv2 has been modified with a custom layer at
        the front of it which performs data augmentation and randomization on
        the color chanels, as well as image rotations.

        Optimizer: adam
        Loss: Binary Crossentropy

        Args:
            df (pandas.DataFrame): a pandas dataframe with the following cols:
                'pdf_label'
                'image'
                'errors'
            size (typing.List[int]): a shape specification for how to configure
                the processing pipes for both the image prep, as well as the
                configuration for the dimensionality of the input layers
                to the neural network itself.
        Returns:
            initialized of class instance TrainingParticulars
        """
        assert "pdf_label" in df
        assert "image" in df
        assert "errors" in df
        assert len(size) == 3
        df = df.loc[~df.errors]
        self.classes = list(set(df.pdf_label))
        self.n_classes = len(self.classes)
        self.mlb = pp.MultiLabelBinarizer(classes=self.classes)
        self.mlb.fit([])
        self.target = self.mlb.transform(df.pdf_label.map(lambda x: [x]).tolist())
        self.weights = self.target.sum(axis=0).max() / self.target.sum(axis=0)
        if modality.upper() == "V2":
            self.model = M.neural.builder.TinyMobileNetv2(
                input_shape=size, output_classes=self.n_classes,
            )
        elif modality.upper() == "V3":
            self.model = M.neural.builder.MobileNetV3(
                input_shape=size, output_classes=self.n_classes,
            )
        else:
            raise NotImplementedError("Modalities of only `V2` or `V3` supported")
        self.training_data = self.prep_images(df.image, size=size[:2])
        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            loss_weights=self.weights,
            metrics=["binary_crossentropy", "acc"],
        )

    def kickoff_training_run(
        self, batch_size: int = 64, epochs: int = 128, shuffle=True
    ):
        self.model.fit(
            x=self.training_data,
            y=self.target,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=shuffle,
        )

    def prep_an_image(self, img: I.Image, size=(256, 256), preserve_aspect_ratio=True):
        # These should already be in RGB format as it is handled
        # earlier in the pipe
        img = tf.image.resize(
            images=np.array(img), size=size, preserve_aspect_ratio=preserve_aspect_ratio
        )
        img = tf.image.resize_with_pad(
            image=img, target_height=size[0], target_width=size[1]
        )
        # prep for batch concatenation
        img = tf.expand_dims(img, 0).numpy()
        return img

    def prep_images(
        self, images: T.List[I.Image], size=(256, 256), preserve_aspect_ratio=True
    ):
        with tf.device("/cpu:0"):
            imgs = [
                self.prep_an_image(
                    img, size=size, preserve_aspect_ratio=preserve_aspect_ratio
                )
                for img in images
            ]
            imgs = np.concatenate(imgs, axis=0)
            return imgs

    def save_model(self):
        """This function dumps whatever state the model is in out to disk.
        Be careful, this overwrites the model that is in the hot-seat
        for hosting. Backup whatever is there if you need to.
        """
        #################################################################
        # prep the parquet which is used for the MultiLabelBinarizer
        df = pd.DataFrame(
            self.mlb.classes_, columns=[self.multilabelbinarizer_label_column]
        )
        df.reset_index(inplace=True)
        df.columns = [
            self.multilabelbiniarizer_index_col_name,
            self.multilabelbinarizer_label_column,
        ]
        L.info(f"Saving MultiLabelBinarizer to: {self.save_label_encoder_to}")
        df.to_parquet(self.save_label_encoder_to, index=False)
        #################################################################
        # save the model
        L.info(f"Saving Model to: {self.save_label_encoder_to}")
        self.model.save(self.save_model_to)


class PredictorForStreamlit:
    def __init__(self, origional: bool = False, enforce_cpu=True):
        self.load_model = load_model
        self.load_label_encoder = load_label_encoder
        self.mlb = self.load_label_encoder(origional=origional)
        self.class_array = self.mlb.classes_.reshape(-1, 1)
        self.model = self.load_model(origional=origional, enforce_cpu=enforce_cpu)
        self.input_dim = self.model.layers[0].input_shape[0][1:-1]
        self.label_col = "label"
        self.prediction_col = "prediction"

    def package_result(self, model_output) -> pd.DataFrame:
        assert model_output.shape[0] == 1
        res = np.concatenate([self.class_array, model_output.reshape(-1, 1)], axis=1)
        df = pd.DataFrame(res)
        df.columns = [self.label_col, self.prediction_col]
        df[self.prediction_col] = df[self.prediction_col].astype(np.float)
        df[self.prediction_col] = np.round(df[self.prediction_col], 4)
        return df

    def predict(self, image: I.Image):
        assert isinstance(image, I.Image)
        X = self.image_prep(img=image.convert("RGB"))
        model_output = self.model.predict(x=X)
        result = self.package_result(model_output=model_output)
        return result

    def image_prep(self, img: I.Image, preserve_aspect_ratio=True):
        # These should already be in RGB format as it is handled
        # earlier in the pipe
        img = tf.image.resize(
            images=np.array(img),
            size=self.input_dim,
            preserve_aspect_ratio=preserve_aspect_ratio,
        )
        img = tf.image.resize_with_pad(
            image=img, target_height=self.input_dim[0], target_width=self.input_dim[1]
        )
        # prep for batch concatenation
        img = tf.expand_dims(img, 0)
        return img

    def plot_packaged_result(self, df: pd.DataFrame):
        assert self.label_col in df
        assert self.prediction_col in df
        df.sort_values(self.label_col, inplace=True)

        classes = df[self.label_col]
        pi = np.pi
        N = classes.shape[0]
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        ax = plt.subplot(1, 1, 1, polar=True, label=uuid.uuid4())
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], classes, color="grey", size=8)
        ax.set_rlabel_position(0)
        plt.yticks(
            np.arange(0, 1 + 0.1, 0.1),
            [f"{xxx:.2f}" for xxx in np.arange(0, 1 + 0.1, 0.1)],
            color="grey",
            size=7,
        )
        plt.ylim(0, 1)
        X = df[self.prediction_col]
        X = np.concatenate([X, X[:1]])
        ax.plot(angles, X, color="green", linewidth=2, linestyle="solid")
        ax.fill(angles, X, color="green", alpha=0.4)
        return plt
