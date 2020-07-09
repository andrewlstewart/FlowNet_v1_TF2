"""
"""

from typing import Dict
from pathlib import Path
from copy import deepcopy

import numpy as np
import tensorflow as tf

import utils_io as uio
from config import FLOWNET_CONFIG


class FlowNet:

    def __init__(self, config: Dict):
        self.config = config

        self.model = self._construct_network(config)

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
        upconv_5 = tf.keras.layers.Conv2DTranspose(name='upconv_5', filters=512, kernel_size=(4,4), strides=2, padding='same', activation=tf.keras.activations.relu)(conv_6)
        flow_6 = tf.keras.layers.Conv2DTranspose(name='flow_6', filters=2, kernel_size=(4,4), strides=2, padding='same', activation=tf.keras.activations.relu)(predict_6)
        concat_5 = tf.keras.layers.Concatenate(name='concat_5', axis=-1)([upconv_5, conv_5_1, flow_6])
        predict_5 = tf.keras.layers.Conv2D(name='predict_5', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_5)

        upconv_4 = tf.keras.layers.Conv2DTranspose(name='upconv_4', filters=256, kernel_size=(4,4), strides=2, padding='same', activation=tf.keras.activations.relu)(concat_5)
        flow_5 = tf.keras.layers.Conv2DTranspose(name='flow_5', filters=2, kernel_size=(4,4), strides=2, padding='same', activation=tf.keras.activations.relu)(predict_5)
        concat_4 = tf.keras.layers.Concatenate(name='concat_4', axis=-1)([upconv_4, conv_4_1, flow_5])
        predict_4 = tf.keras.layers.Conv2D(name='predict_4', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_4)

        upconv_3 = tf.keras.layers.Conv2DTranspose(name='upconv_3', filters=128, kernel_size=(4,4), strides=2, padding='same', activation=tf.keras.activations.relu)(concat_4)
        flow_4 = tf.keras.layers.Conv2DTranspose(name='flow_4', filters=2, kernel_size=(4,4), strides=2, padding='same', activation=tf.keras.activations.relu)(predict_4)
        concat_3 = tf.keras.layers.Concatenate(name='concat_3', axis=-1)([upconv_3, conv_3_1, flow_4])
        predict_3 = tf.keras.layers.Conv2D(name='predict_3', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_3)

        upconv_2 = tf.keras.layers.Conv2DTranspose(name='upconv_2', filters=64, kernel_size=(4,4), strides=2, padding='same', activation=tf.keras.activations.relu)(concat_3)
        flow_3 = tf.keras.layers.Conv2DTranspose(name='flow_3', filters=2, kernel_size=(4,4), strides=2, padding='same', activation=tf.keras.activations.relu)(predict_3)
        concat_2 = tf.keras.layers.Concatenate(name='concat_2', axis=-1)([upconv_2, conv_2, flow_3])
        predict_2 = tf.keras.layers.Conv2D(name='predict_2', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_2)

        upconv_1 = tf.keras.layers.Conv2DTranspose(name='upconv_1', filters=64, kernel_size=(4,4), strides=2, padding='same', activation=tf.keras.activations.relu)(concat_2)
        flow_2 = tf.keras.layers.Conv2DTranspose(name='flow_2', filters=2, kernel_size=(4,4), strides=2, padding='same', activation=tf.keras.activations.relu)(predict_2)
        concat_1 = tf.keras.layers.Concatenate(name='concat_1', axis=-1)([upconv_1, conv_1, flow_2])
        predict_1 = tf.keras.layers.Conv2D(name='predict_1', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_1)

        if config['training']:
            return tf.keras.Model(inputs=inputs, outputs=[predict_6, predict_5, predict_4, predict_3, predict_2, predict_1])
        
        return tf.keras.Model(inputs=inputs, outputs=predict_1)

    @staticmethod
    def get_corr_model(config: Dict) -> tf.keras.Model:
        raise NotImplementedError("The correlation model hasn't been implemented.")

    @staticmethod
    def _construct_network(config: Dict) -> tf.keras.Model:
        if config['architecture'] == 'simple':
            return FlowNet.get_simple_model(config)
        if config['architecture'] == 'corr':
            return FlowNet.get_corr_model(config)
        
        raise NotImplementedError(f"Configuration dictionary's architecture hasn't been implemented")


def get_model():
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
    upconv_5 = tf.keras.layers.Conv2DTranspose(name='upconv_5', filters=512, kernel_size=(4,4), strides=2, padding='same', activation=tf.keras.activations.relu)(conv_6)
    flow_6 = tf.keras.layers.Conv2DTranspose(name='flow_6', filters=2, kernel_size=(4,4), strides=2, padding='same', activation=tf.keras.activations.relu)(predict_6)
    concat_5 = tf.keras.layers.Concatenate(name='concat_5', axis=-1)([upconv_5, conv_5_1, flow_6])
    predict_5 = tf.keras.layers.Conv2D(name='predict_5', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_5)

    upconv_4 = tf.keras.layers.Conv2DTranspose(name='upconv_4', filters=256, kernel_size=(4,4), strides=2, padding='same', activation=tf.keras.activations.relu)(concat_5)
    flow_5 = tf.keras.layers.Conv2DTranspose(name='flow_5', filters=2, kernel_size=(4,4), strides=2, padding='same', activation=tf.keras.activations.relu)(predict_5)
    concat_4 = tf.keras.layers.Concatenate(name='concat_4', axis=-1)([upconv_4, conv_4_1, flow_5])
    predict_4 = tf.keras.layers.Conv2D(name='predict_4', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_4)

    upconv_3 = tf.keras.layers.Conv2DTranspose(name='upconv_3', filters=128, kernel_size=(4,4), strides=2, padding='same', activation=tf.keras.activations.relu)(concat_4)
    flow_4 = tf.keras.layers.Conv2DTranspose(name='flow_4', filters=2, kernel_size=(4,4), strides=2, padding='same', activation=tf.keras.activations.relu)(predict_4)
    concat_3 = tf.keras.layers.Concatenate(name='concat_3', axis=-1)([upconv_3, conv_3_1, flow_4])
    predict_3 = tf.keras.layers.Conv2D(name='predict_3', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_3)

    upconv_2 = tf.keras.layers.Conv2DTranspose(name='upconv_2', filters=64, kernel_size=(4,4), strides=2, padding='same', activation=tf.keras.activations.relu)(concat_3)
    flow_3 = tf.keras.layers.Conv2DTranspose(name='flow_3', filters=2, kernel_size=(4,4), strides=2, padding='same', activation=tf.keras.activations.relu)(predict_3)
    concat_2 = tf.keras.layers.Concatenate(name='concat_2', axis=-1)([upconv_2, conv_2, flow_3])
    predict_2 = tf.keras.layers.Conv2D(name='predict_2', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_2)

    upconv_1 = tf.keras.layers.Conv2DTranspose(name='upconv_1', filters=64, kernel_size=(4,4), strides=2, padding='same', activation=tf.keras.activations.relu)(concat_2)
    flow_2 = tf.keras.layers.Conv2DTranspose(name='flow_2', filters=2, kernel_size=(4,4), strides=2, padding='same', activation=tf.keras.activations.relu)(predict_2)
    concat_1 = tf.keras.layers.Concatenate(name='concat_1', axis=-1)([upconv_1, conv_1, flow_2])
    predict_1 = tf.keras.layers.Conv2D(name='predict_1', filters=2, kernel_size=3, strides=1, padding='same', activation=None)(concat_1)

    model = tf.keras.Model(inputs=inputs, outputs=predict_1)

    return model


    # data = np.random.rand(1,384,512,6)

    # out = model.predict(data)

    # upconv_5 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=(5,5), strides=(1,1), padding='valid')


def load_images():
    root_path = Path(r'C:\Users\andre\Documents\Python\FlowNet_TF2\data\FlyingChairs_release\data')
    flo_path = root_path / '00001_flow.flo'
    img1_path = root_path / '00001_img1.ppm'
    img2_path = root_path / '00001_img2.ppm'

    flo = uio.read(str(flo_path))
    img1 = uio.read(str(img1_path))
    img2 = uio.read(str(img2_path))

    # fig, ax = plt.subplots(ncols=2, nrows=2)
    # ax[0,0].imshow(img1)
    # ax[0,1].imshow(img2)
    # ax[1,0].imshow(flo[...,0])
    # ax[1,1].imshow(flo[...,1])
    # plt.show()

    print('stall')


def main():
    config_network = deepcopy(FLOWNET_CONFIG)

    flownet = FlowNet(config_network)

    print('stall')


if __name__=="__main__":
    main()