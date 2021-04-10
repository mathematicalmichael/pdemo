import tensorflow as tf
import numpy as np
import typing as T
from PIL import Image as I


class DataAugmentation(tf.keras.layers.Layer):
    def __init__(
        self,
        # batch_size:int,
        name: str = "input_data_augmentation",
        flip_chance: float = 0.5,
        dynamic=False,
        luma_fuzz=0.05,
        croma_fuzz=0.05,
        exclude_grand_luma_fuzz_chance: float = 0.5,
        exclude_luma_fuzz_chance: float = 0.8,
        exclude_croma_u_fuzz_chance: float = 0.8,
        exclude_croma_v_fuzz_chance: float = 0.8,
        # dtype=tf.float16,
        **kwargs,
    ):
        """Augmentation Layer - transforms the data to help with generalization.

        Order of Operations:
            1) random flip left right
            2) random flip up down
            3) random transpose
            4) cast to float16
            5) normalize 0-255 rgb vals to 0-1
            6) rgb to yuv
            7) fuzz macro image luma
            8) fuzz pixel value luma
            9) fuzz croma u
            10) fuzz croma v
            11) image gradients on luma channel (opt)

        Args:
            name: (str= 'input_data_augmentation') optional, name of the layer
            flip_chance: (float) defaults to .5, applies to all flip types:
                left/right, up/down, transpose. Flips are global across batch
                and expects a batch to be randomly sampled from the sample
                space.
            luma_fuzz: (float) defaults to .2; implies additive noise drawn
                from Uniform(-luma_fuzz, +luma_fuzz)
            croma_fuzz: (float) defaults to .1; implies additive noise drawn
                from Uniform(-luma_fuzz, +luma_fuzz)
            exclude_grand_luma_fuzz_chance: (float) controls whether the luma
                for the entire patch will be uniformly shifted by a random
                value
            exclude_luma_fuzz_chance: (float) controls whether the per-pixel
                luma will be applied for the batch
            exclude_croma_u_fuzz_chance: (float) controls whether the per-pixel
                croma u will be applied for the batch
            exclude_croma_v_fuzz_chance: (float) controls whether the per-pixel
                croma v will be applied for the batch
        """
        assert 0.0 <= luma_fuzz <= 1.0
        assert 0.0 <= croma_fuzz <= 0.5
        assert 0.0 <= flip_chance <= 1.0
        assert 0.0 <= exclude_grand_luma_fuzz_chance <= 1.0
        assert 0.0 <= exclude_luma_fuzz_chance <= 1.0
        assert 0.0 <= exclude_croma_u_fuzz_chance <= 1.0
        assert 0.0 <= exclude_croma_v_fuzz_chance <= 1.0
        self.flip_chance = flip_chance
        self.luma_fuzz = luma_fuzz
        self.croma_fuzz = croma_fuzz
        self.exclude_grand_luma_fuzz_chance = exclude_grand_luma_fuzz_chance
        self.exclude_luma_fuzz_chance = exclude_luma_fuzz_chance
        self.exclude_croma_u_fuzz_chance = exclude_croma_u_fuzz_chance
        self.exclude_croma_v_fuzz_chance = exclude_croma_v_fuzz_chance
        # self.dtype = dtype
        # self.batch_size = batch_size
        # useful for diagnostic things
        self._yuv_to_rgb_kernel = [
            [1, 1, 1],
            [0, -0.394642334, 2.03206185],
            [1.13988303, -0.58062185, 0],
        ]
        # useful for diagnostic things
        self._rgb_to_yuv_kernel = [
            [0.299, -0.14714119, 0.61497538],
            [0.587, -0.28886916, -0.51496512],
            [0.114, 0.43601035, -0.10001026],
        ]
        allowed_kwargs = {
            "input_dim",
            "input_shape",
            "batch_input_shape",
            "batch_size",
            "weights",
            "activity_regularizer",
            "autocast",
        }
        super_args = {kkk: vvv for kkk, vvv in kwargs.items() if kkk in allowed_kwargs}
        print(super_args)
        super(self.__class__, self).__init__(
            name=name,
            dynamic=dynamic,
            # dtype=dtype
            **super_args,
        )
        # self.build()
        # Other things we could play with are color-space inversion and
        # more augmentations if we get a generalization issue down the line

    def build(self, input_shape):
        # self.batch_size = input_shape[0]
        print(f"Input Shape for Build: {input_shape}")
        super(self.__class__, self).build(input_shape)
        self.shape = input_shape

    def compute_input_shape(self, input_shape):
        print(f"Computed Input Shape: {self.input_shape}")
        return input_shape

    def compute_output_shape(self, input_shape):
        print(f"Computed Output Shape: {input_shape}")
        return input_shape

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "flip_chance": self.flip_chance,
                "luma_fuzz": self.luma_fuzz,
                "croma_fuzz": self.croma_fuzz,
                "_yuv_to_rgb_kernel": self._yuv_to_rgb_kernel,
                "_rgb_to_yuv_kernel": self._rgb_to_yuv_kernel,
            }
        )
        return config

    def call(
        self,
        input_tensor,
        training=False,
        include_grad=False,
        interim_dtype=tf.float32,
        output_dtype=tf.float16,
    ):
        x = input_tensor
        output_dtype = self.dtype
        print(type(input_tensor))
        # assumes [batch, height, width, channel]
        assert len(x.shape) == 4
        if training:
            if tf.random.uniform((1,)) <= self.flip_chance:
                # random flip height
                x = tf.reverse(x, axis=(1,))
            if tf.random.uniform((1,)) <= self.flip_chance:
                # random flip width
                x = tf.reverse(x, axis=(2,))
            if tf.random.uniform((1,)) <= self.flip_chance:
                # random transpose
                x = tf.transpose(x, perm=(0, 2, 1, 3))
        x = tf.cast(x, interim_dtype) / 255.0
        x = tf.image.rgb_to_yuv(x)
        if training:
            # fuzz here, for now just doing uniform addtion
            # multiplicative is another consideration, but not implemented
            # at this time
            luma = x[:, :, :, 0:1]
            if self.luma_fuzz > 0.0:
                if tf.random.uniform((1,)) <= self.exclude_grand_luma_fuzz_chance:
                    # full image luma
                    luma = (
                        tf.random.uniform(
                            (tf.shape(luma)[0], 1, 1, 1),
                            -self.luma_fuzz,
                            self.luma_fuzz,
                            dtype=interim_dtype,
                        )
                        + luma
                    )
                luma = tf.clip_by_value(luma, 0.0, 1.0)
                if tf.random.uniform((1,)) <= self.exclude_luma_fuzz_chance:
                    # per-pixel luma
                    luma = (
                        tf.random.uniform(
                            tf.shape(luma),
                            -(self.luma_fuzz ** 2),
                            self.luma_fuzz ** 2,
                            dtype=interim_dtype,
                        )
                        + luma
                    )
            # squelch signal
            luma = tf.clip_by_value(luma, 0.0, 1.0)
            #
            croma_u = x[:, :, :, 1:2]
            if self.croma_fuzz > 0.0:
                if tf.random.uniform((1,)) <= self.exclude_croma_u_fuzz_chance:
                    croma_u = (
                        tf.random.uniform(
                            tf.shape(croma_u),
                            -self.croma_fuzz,
                            self.croma_fuzz,
                            dtype=interim_dtype,
                        )
                        + croma_u
                    )
            # squelch signal
            croma_u = tf.clip_by_value(croma_u, -0.5, 0.5)
            #
            croma_v = x[:, :, :, 2:3]
            if self.croma_fuzz > 0.0:
                if tf.random.uniform((1,)) <= self.exclude_croma_v_fuzz_chance:
                    croma_v = (
                        tf.random.uniform(
                            tf.shape(croma_v),  # .shape,
                            -self.croma_fuzz,
                            self.croma_fuzz,
                            dtype=interim_dtype,
                        )
                        + croma_v
                    )
            croma_v = tf.clip_by_value(croma_v, -0.5, 0.5)
            x = tf.concat([luma, croma_u, croma_v], axis=-1)
        if include_grad:
            grads = tf.image.image_gradients(x[:, :, :, :1])
            x = tf.concat([x, *grads], axis=-1)
        x = tf.cast(x, output_dtype)
        return x


class DataAugmentationApproximateInverse(tf.keras.Model):
    def __init__(self, name: str = "input_data_augmentation_inverse_transform"):
        """Not to be used for Models, purely a debugging/diagnostic layer"""
        super(self.__class__, self).__init__(name=name)

    def call(self, input_tensor, training=False):
        # Note expects bounds luma, croma_u, croma_v in [0,1],[-.5,.5],[-.5,.5]
        x = input_tensor
        x = x[:, :, :, 0:3]
        x = tf.image.yuv_to_rgb(x)
        x = tf.clip_by_value(x, clip_value_min=0, clip_value_max=1)
        x = x * 255.0
        x = tf.clip_by_value(x, clip_value_min=0, clip_value_max=255.0)
        x = tf.cast(x, tf.uint8)
        return x


def _diagnostic_dump(context):
    din = context.features[0:50]
    lay = DataAugmentation()
    lay2 = DataAugmentationApproximateInverse()
    res = lay2(lay(din, training=True))
    imgs = [I.fromarray(res[iii].numpy()) for iii in range(res.shape[0])]
    for iii, img, in enumerate(imgs):
        fname = f"/scratch_nvme/presentation_assets/augmented_{iii}.jpg"
        print(f"Saving {fname}")
        img.save(fname)


class InputAugmentations:
    DataAugmentationApproximateInverse = DataAugmentationApproximateInverse
    DataAugmentation = DataAugmentation
