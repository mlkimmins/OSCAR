import traceback

"""This file contains some utility functions"""

import tensorflow as tf
import time
import os
import numpy as np


def load_embeddings(fpath, word_to_id, emd_init_var=0.25):
	""" loads pretrained embeddings into a dictionary 
	Args:
		fpath: file path to the text file of word embedddings
		word_to_id: mapping of word to ids from Vocab
		emd_init_var: initialization variance of embedding vectors
	returns:
		dictionary
	"""
	f_iter = open(fpath)
	num_words, vector_size = list(map( int,next(f_iter).strip().split(' ')))
	np.random.seed(123)
	embd = np.random.uniform(-emd_init_var,emd_init_var,(len(word_to_id), vector_size))
	i = 0
	print('loading word embeddings')
	for line in f_iter:
		i += 1
		if i % 1000 == 0:
			print('{}K lines processed'.format(i), '\r')
		if i == 1:
			continue
		row = line.strip().split(' ')
		word = row[0]
		vec = list(map(float, row[1:]))
		embd[word_to_id[word]] = np.array(vec)
	print('embeddings loaded, vocab size: {}'.format(num_words))
	f_iter.close()
	return embd

def get_config():
	"""Returns config for tf.session"""
	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth=True
	return config