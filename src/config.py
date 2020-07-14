"""
"""

from pathlib import Path

CONFIG_FLOWNET = {'architecture': 'simple',
                  'training': True,
                  'flo_normalization': (-299.7564, 248.32184), # determined from the training dataset
                  }

CONFIG_TRAINING = {'img_path': Path(r'C:\Users\andre\Documents\Python\FlowNet_TF2\data\FlyingChairs_release\data'),
                   'train_ratio': 0.8,
                   'test_ratio': 0.1,
                   'shuffle': False,
                   'batch_size': 8,
                   'validation_batch_size': 8,
                   'learning_rate': 1e-4,
                   'augmentations': {'translation': [-0.2, 0.2],  # TODO: Not implemented
                                     'rotation': [-17, 17],  # Degrees. Gross.  # TODO: Not implemented
                                     'scaling': [0.9, 2.0],  # TODO: Not implemented
                                     'gaussian_noise': 0.04,  # Implemented
                                     'contrast': [0.2, 1.5],  # Implemented
                                     'multiplicative_colour': [0.5, 1.5],  # Implemented
                                     'gamma': [0.8, 1.5],  # Implemented
                                     'brightness': 0.2  # Implemented
                                     }
                   }
