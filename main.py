# TODO: parse arguments
...


# set seed
seed = ...  # TODO: set seed to allow for reproducibility of results

import os
os.environ['PYTHONHASHSEED'] = str(seed)

import random
random.seed(seed)

import numpy as np
np.random.seed(seed)

import tensorflow as tf
tf.random.set_seed(seed)


# initialize environment
from environment import Environment

data_dir = ...  # TODO: specify relative path to data directory (e.g., './data', not './data/variant_0')
variant = ...  # TODO: specify problem variant (0 for base variant, 1 for first extension, 2 for second extension)
env = Environment(variant, data_dir)
env.feature_mode = "all" # ToDo: train, then change to "env.feature_mode = "no_distance" " and train again
                        # then  change to "env.feature_mode = "no_urgency"" and train and then to "env.feature_mode = "no_capacity"" and train again
                        # Compare results in a table and see which features actually help the agent


# TODO: execute training
...
