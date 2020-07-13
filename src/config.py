"""
"""

from pathlib import Path

CONFIG_FLOWNET = {'architecture': 'simple',
                  'training': True,
                  }

CONFIG_TRAINING = {'img_path': Path(r'C:\Users\andre\Documents\Python\FlowNet_TF2\data\FlyingChairs_release\data'),
                   'train_ratio': 0.8,
                   'test_ratio': 0.1,
                   'shuffle': False,
                   'batch_size': 8,
                   'validation_batch_size': 32,  # TODO: Is this 'allowed', what are the ramifications 

                   'learning_rate': 1e-4
                   }
