"""
"""

from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

import utils_io as uio
from config import CONFIG_FLOWNET, CONFIG_TRAINING


class MalformedNetworkType(Exception):
    """The provided network type doesn't match one of 'simple' or 'correlation'."""


class FlowNet:
    """ FlowNetSimple model from the Computer Vision Group of Freiburg.
        https://lmb.informatik.uni-freiburg.de/
        https://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/flownet.pdf
    """

    def __init__(self, config: Dict):
        self.config = config

        self.model = self._construct_network(config)

    def __getattr__(self, attr):
        """ Rather than potentially override any of the tf.keras.Model methods by subclassing and defining new methods,
            create a composition class with self.model:tf.keras.Model and allow attribute calls directly against self.model
        """
        return getattr(self.model, attr)

    @staticmethod
    def get_simple_model(config: Dict) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(384, 512, 6))

        """ Contracting part """
        conv_1 = tf.keras.layers.Conv2D(name='conv1', filters=64, kernel_size=7, strides=2, padding='same', activation=tf.keras.activations.relu)(inputs)
        conv_2 = tf.keras.layers.Conv2D(name='conv2', filters=128, kernel_size=5, strides=2, padding='same', activation=tf.keras.activations.relu)(conv_1)
        conv_3 = tf.keras.layers.Conv2D(name='conv3', filters=256, kernel_size=5, strides=2, padding='same', activation=tf.keras.activations.relu)(conv_2)
        conv_3_1 = tf.keras.layers.Conv2D(name='conv3_1', filters=256, kernel_size=3, strides=1, padding='same', activation=tf.keras.activations.relu)(conv_3)
        conv_4 = tf.keras.layers.Conv2D(name='conv4', filters=512, kernel_size=3, strides=2, padding='same', activation=tf.keras.activations.relu)(conv_3_1)
        conv_4_1 = tf.keras.layers.Conv2D(name='conv4_1', filters=512, kernel_size=3, strides=1, padding='same', activation=tf.keras.activations.relu)(conv_4)
        conv_5 = tf.keras.layers.Conv2D(name='conv5', filters=512, kernel_size=3, strides=2, padding='same', activation=tf.keras.activations.relu)(conv_4_1)
        conv_5_1 = tf.keras.layers.Conv2D(name='conv5_1', filters=512, kernel_size=3, strides=1, padding='same', activation=tf.keras.activations.relu)(conv_5)
        conv_6 = tf.keras.layers.Conv2D(name='conv6', filters=1024, kernel_size=3, strides=2, padding='same', activation=tf.keras.activations.relu)(conv_5_1)
        conv_6_1 = tf.keras.layers.Conv2D(name='conv6_1', filters=1024, kernel_size=3, strides=1, padding='same', activation=tf.keras.activations.relu)(conv_6)

        """ The paper itself doesn't have this documented but all implementations, including the original authors, use an extra flow path in the code. """
        predict_6 = tf.keras.layers.Conv2D(name='predict_6', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(conv_6_1)

        """ Expanding part """
        upconv_5 = tf.keras.layers.Conv2DTranspose(name='upconv_5', filters=512, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(conv_6)
        flow_6 = tf.keras.layers.Conv2DTranspose(name='flow_6', filters=2, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(predict_6)
        concat_5 = tf.keras.layers.Concatenate(name='concat_5', axis=-1)([upconv_5, conv_5_1, flow_6])
        predict_5 = tf.keras.layers.Conv2D(name='predict_5', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_5)

        upconv_4 = tf.keras.layers.Conv2DTranspose(name='upconv_4', filters=256, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(concat_5)
        flow_5 = tf.keras.layers.Conv2DTranspose(name='flow_5', filters=2, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(predict_5)
        concat_4 = tf.keras.layers.Concatenate(name='concat_4', axis=-1)([upconv_4, conv_4_1, flow_5])
        predict_4 = tf.keras.layers.Conv2D(name='predict_4', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_4)

        upconv_3 = tf.keras.layers.Conv2DTranspose(name='upconv_3', filters=128, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(concat_4)
        flow_4 = tf.keras.layers.Conv2DTranspose(name='flow_4', filters=2, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(predict_4)
        concat_3 = tf.keras.layers.Concatenate(name='concat_3', axis=-1)([upconv_3, conv_3_1, flow_4])
        predict_3 = tf.keras.layers.Conv2D(name='predict_3', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_3)

        upconv_2 = tf.keras.layers.Conv2DTranspose(name='upconv_2', filters=64, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(concat_3)
        flow_3 = tf.keras.layers.Conv2DTranspose(name='flow_3', filters=2, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(predict_3)
        concat_2 = tf.keras.layers.Concatenate(name='concat_2', axis=-1)([upconv_2, conv_2, flow_3])
        predict_2 = tf.keras.layers.Conv2D(name='predict_2', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_2)

        upconv_1 = tf.keras.layers.Conv2DTranspose(name='upconv_1', filters=64, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(concat_2)
        flow_2 = tf.keras.layers.Conv2DTranspose(name='flow_2', filters=2, kernel_size=(4, 4), strides=2, padding='same', activation=tf.keras.activations.relu)(predict_2)
        concat_1 = tf.keras.layers.Concatenate(name='concat_1', axis=-1)([upconv_1, conv_1, flow_2])
        predict_1 = tf.keras.layers.Conv2D(name='predict_1', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_1)

        if config['training']:
            return tf.keras.Model(inputs=inputs, outputs=[predict_6, predict_5, predict_4, predict_3, predict_2, predict_1])

        return tf.keras.Model(inputs=inputs, outputs=predict_1)

    def disable_training(self):
        """ After training is finished, run this method to have self.model predict a single array rather than a list of 6 arrays
        """
        self.model = tf.keras.Model(inputs=self.model.layers[0].input, outputs=self.model.layers[-1].output)

    def enable_training(self):
        """ If you need to re-enable training, run this method to have self.model predict the list of 6 predictions
        """
        output_layers = [layer.output for layer in self.model.layers if 'predict' in layer.name]
        self.model = tf.keras.Model(inputs=self.model.layers[0].input, outputs=output_layers)

    @staticmethod
    def get_corr_model(config: Dict) -> tf.keras.Model:
        raise NotImplementedError("The correlation model hasn't been implemented.")

    @staticmethod
    def _construct_network(config: Dict) -> tf.keras.Model:
        if config['architecture'] == 'simple':
            return FlowNet.get_simple_model(config)
        if config['architecture'] == 'corr':
            return FlowNet.get_corr_model(config)

        raise MalformedNetworkType(f"{config['architecture']}: {MalformedNetworkType.__doc__}")


def get_train_val_test(image_names: List[Path],
                       train_ratio: Union[float, int],
                       test_ratio: Union[float, int],
                       shuffle: bool = True) -> Tuple[List[Path], List[Path], List[Path]]:
    """ Get the train, val, and test sets from a list of all image paths.
        The test set is the last block and shouldn't be handled until after hyperparameter tuning.
        This function is sloppy and can easily be broken.  Reasonable values, such as train_ratio=0.7 and test_ratio=0.1
        will return a train_ratio of 0.7, a validation_ratio of 0.2, and a test_ratio of 0.1 and work fine.
    """
    if (not 0 < train_ratio < 1) or (not 0 < test_ratio < 1) or (train_ratio + test_ratio >= 1):
        raise Exception(f"Why have you done this. Train ratio: {train_ratio}, val ratio: {1-train_ratio-test_ratio}, Test ratio: {test_ratio}.")

    n_images = len(image_names)
    test = image_names[int(-test_ratio*n_images):]  # Don't use the last set of images until done hyperparameter tuning

    image_names = image_names[:int(-test_ratio*n_images)]

    n_train = int(train_ratio * n_images)
    if shuffle:
        np.random.shuffle(image_names)
    train = image_names[:n_train]
    val = image_names[n_train:]

    return train, val, test


class DataGenerator:
    """
    """

    def __init__(self,
                 network_type: str,
                 root_path: Path,
                 batch_size: int,
                 validation_batch_size: int,
                 train_ratio: Union[float, int] = 1,
                 test_ratio: Union[float, int] = 0,
                 shuffle: bool = False,
                 augmentations: Optional[Dict] = None):
        self.network_type = network_type

        images = list(root_path.glob('*1.ppm'))
        self.train, self.val, self.test = get_train_val_test(images, train_ratio, test_ratio, shuffle)
        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.replace = True

    def next_train(self):

        while True:
            images = np.random.choice(self.train, self.batch_size, replace=self.replace)
            img1 = [uio.read(str(img)) for img in images]
            img2 = [uio.read(str(img).replace('1.ppm', '2.ppm')) for img in images]
            label = [uio.read(str(img).replace('img1.ppm', 'flow.flo')) for img in images]

            if self.network_type == 'simple':
                images = np.concatenate([img1, img2], axis=-1)
            elif self.network_type == 'correlation':
                raise NotImplementedError()
            else:
                raise MalformedNetworkType(f'{self.network_type}: {MalformedNetworkType.__doc__}')

            yield (images, np.array(label))

    def next_val(self):

        while True:
            images = np.random.choice(self.val, self.validation_batch_size, replace=False)
            img1 = [uio.read(str(img)) for img in images]
            img2 = [uio.read(str(img).replace('1.ppm', '2.ppm')) for img in images]
            label = [uio.read(str(img).replace('img1.ppm', 'flow.flo')) for img in images]

            if self.network_type == 'simple':
                images = np.concatenate([img1, img2], axis=-1)
            elif self.network_type == 'correlation':
                raise NotImplementedError()
            else:
                raise MalformedNetworkType(f'{self.network_type}: {MalformedNetworkType.__doc__}')

            yield (images, np.array(label))


class EndPointError(tf.keras.losses.Loss):
    """ EndPointError is the Euclidean distance between the predicted flow vector and the ground truth averaged over all pixels.
        The resizing is required because the loss is calculated for each flow prediction which occur at different stride levels,
        resizing effectively averages at that scale.
    """

    def call(self, y_true, y_pred):
        return K.sqrt(K.sum(K.square(tf.image.resize(y_true, y_pred.shape[1:3]) - y_pred), axis=1, keepdims=True))


def load_images():
    """ Debug function to load the first image for visualization
    """
    root_path = Path(r'C:\Users\andre\Documents\Python\FlowNet_TF2\data\FlyingChairs_release\data')
    flo_path = root_path / '00001_flow.flo'
    img1_path = root_path / '00001_img1.ppm'
    img2_path = root_path / '00001_img2.ppm'
    flo = uio.read(str(flo_path))
    img1 = uio.read(str(img1_path))
    img2 = uio.read(str(img2_path))
    img = np.expand_dims(np.concatenate([img1, img2], axis=-1), axis=0)

    # fig, ax = plt.subplots(ncols=2, nrows=2)
    # ax[0,0].imshow(img1)
    # ax[0,1].imshow(img2)
    # ax[1,0].imshow(flo[...,0])
    # ax[1,1].imshow(flo[...,1])
    # plt.show()

    return img, np.expand_dims(flo, axis=0)


def show_images(simple_images, label):
    """
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(ncols=2, nrows=2)
    ax[0, 0].imshow(simple_images[..., :3])
    ax[0, 1].imshow(simple_images[..., 3:])
    ax[1, 0].imshow(label[..., 0])
    ax[1, 1].imshow(label[..., 1])
    plt.show()


def main():
    config_network = deepcopy(CONFIG_FLOWNET)
    config_training = deepcopy(CONFIG_TRAINING)

    flownet = FlowNet(config_network)

    loss = EndPointError()

    flownet.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss)

    data_generator = DataGenerator(network_type=config_network['architecture'],
                                   root_path=config_training['img_path'],
                                   batch_size=config_training['batch_size'],
                                   validation_batch_size=config_training['validation_batch_size'],
                                   train_ratio=config_training['train_ratio'],
                                   test_ratio=config_training['test_ratio'],
                                   shuffle=config_training['shuffle'])

    log_dir = f"logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_filepath = '/tmp/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    flownet.fit(x=data_generator.next_train(),
                epochs=10,
                verbose=1,
                steps_per_epoch=22872 // config_training['batch_size'],
                validation_data=data_generator.next_val(),
                validation_steps=1,
                validation_batch_size=config_training['validation_batch_size'],
                callbacks=[tensorboard_callback, model_checkpoint_callback],
                # use_multiprocessing=True
                )
    flownet.disable_training()

    #
    # Temporary debugging and visualization
    #
    img, flo = load_images()
    predicted_flo = flownet.predict(img)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(ncols=2, nrows=3)
    ax[0, 0].imshow(img[0, ..., :3])
    ax[0, 1].imshow(img[0, ..., 3:])
    ax[1, 0].imshow(flo[..., 0])
    ax[1, 1].imshow(flo[..., 1])
    ax[2, 0].imshow(predicted_flo[0, ..., 0])
    ax[2, 1].imshow(predicted_flo[0, ..., 1])
    plt.show()

    print('stall')


if __name__ == "__main__":
    main()
