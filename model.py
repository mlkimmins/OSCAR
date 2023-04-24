import sys

"""
This file will contain the models we implement
"""

import os
import time
import numpy as np
import tensorflow as tf
from util import load_embeddings
from tensorflow.python.ops import array_ops
from six.moves import xrange
from tensorflow.python import debug as tf_debug
FLAGS = tf.compat.v1.flags.FLAGS

class SummarizationModel(object):

    def __init__(self, hps, vocab, num_gpus):
        self._hps = hps
        self._cur_gpu = 0
        self._vocab = vocab
        self._num_gpus = num_gpus
        if FLAGS.new_attention and FLAGS.hier:
            print('using linear attention mechanism for considering sections')
            from attention_decoder_new import attention_decoder
        else:
            print('using hierarchical attention mechanism for considering sections')
            from attention_decoder import attention_decoder
        self.attn_decoder = attention_decoder

