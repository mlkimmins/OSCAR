import sys
import time
import os
import tensorflow as tf
import numpy as np
from collections import namedtuple
from data import Vocab
from batch_reader import Batcher
from model import SummarizationModel
# from decode import BeamSearchDecoder
from tensorflow.python import debug as tf_debug
import util


# Some common configs. Override them as necessary.
FLAGS = tf.compat.v1.flags.FLAGS

# Where to find data
tf.compat.v1.flags.DEFINE_string(
	'data_path', '', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.compat.v1.flags.DEFINE_string(
	'vocab_path', '', 'Path expression to text vocabulary file.')

# Some keys
tf.compat.v1.flags.DEFINE_string('article_id_key', 'article_id',
						   'tf.Example feature key for article.')
tf.compat.v1.flags.DEFINE_string('article_key', 'article_body',
						   'tf.Example feature key for article.')
tf.compat.v1.flags.DEFINE_string('abstract_key', 'abstract',
						   'tf.Example feature key for abstract.')
tf.compat.v1.flags.DEFINE_string('labels_key', 'labels',
						   'tf.Example feature key for labels.')
tf.compat.v1.flags.DEFINE_string('section_names_key', 'section_names',
						   'tf.Example feature key for section names.')
tf.compat.v1.flags.DEFINE_string('sections_key', 'sections',
						   'tf.Example feature key for sections.')

# Important settings
tf.compat.v1.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')
tf.compat.v1.flags.DEFINE_boolean('single_pass', False, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')

# Where to save output
tf.compat.v1.flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
tf.compat.v1.flags.DEFINE_string(
	'exp_name', '', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# Hyperparameters
tf.compat.v1.flags.DEFINE_integer(
	'hidden_dim', 256, 'dimension of RNN hidden states')
tf.compat.v1.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
tf.compat.v1.flags.DEFINE_integer('batch_size', 16, 'minibatch size')
tf.compat.v1.flags.DEFINE_integer(
	'max_enc_steps', 1200, 'max timesteps of encoder (max source text tokens)')
tf.compat.v1.flags.DEFINE_integer(
	'max_dec_steps', 150, 'max timesteps of decoder (max summary tokens)')
tf.compat.v1.flags.DEFINE_integer(
	'beam_size', 4, 'beam size for beam search decoding.')
tf.compat.v1.flags.DEFINE_integer(
	'min_dec_steps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.compat.v1.flags.DEFINE_integer(
	'vocab_size', 100000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.compat.v1.flags.DEFINE_float('lr', 0.15, 'learning rate')
tf.compat.v1.flags.DEFINE_float('adagrad_init_acc', 0.1,
						  'initial accumulator value for Adagrad')
tf.compat.v1.flags.DEFINE_float('rand_unif_init_mag', 0.05,
						  'magnitude for lstm cells random uniform inititalization')
tf.compat.v1.flags.DEFINE_float('trunc_norm_init_std', 1e-4,
						  'std of trunc norm init, used for initializing everything else')
tf.compat.v1.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')
tf.compat.v1.flags.DEFINE_float('min_lr', 0.005, 'for gradient decent learning rate')

tf.compat.v1.flags.DEFINE_float('max_abstract_len', 500, 'Discards articles with longer abstracts')
tf.compat.v1.flags.DEFINE_float('min_abstract_len', 50, 'Discards articles with short abstracts')
tf.compat.v1.flags.DEFINE_float('max_article_sents', 350, 'Discards articles with short abstracts')

tf.compat.v1.flags.DEFINE_boolean('use_sections', False, 'use hierarchical encoding/decoding over sections')
tf.compat.v1.flags.DEFINE_integer('max_section_len', 400, 'Truncate sections')
tf.compat.v1.flags.DEFINE_integer('min_section_len', 50, 'Discards short sections')
tf.compat.v1.flags.DEFINE_integer('max_article_len', 2400, 'Maximum input article length')
tf.compat.v1.flags.DEFINE_integer('max_intro_len', 400, 'Maximum introduction section length')
tf.compat.v1.flags.DEFINE_integer('max_conclusion_len', 400, 'Maximum conclusion section length')
tf.compat.v1.flags.DEFINE_integer('max_intro_sents', 20, 'Maximum introduction section length')
tf.compat.v1.flags.DEFINE_integer('max_conclusion_sents', 20, 'Maximum conclusion section length')
tf.compat.v1.flags.DEFINE_integer('max_section_sents', 20, 'Maximum section length in sentences')
tf.compat.v1.flags.DEFINE_integer('num_sections', 6, 'Maximum introduction section length')
tf.compat.v1.flags.DEFINE_boolean('hier', False, 'Hierarchical model to utilize section information')
tf.compat.v1.flags.DEFINE_boolean('phased_lstm', False, 'Hierarchical model to utilize section information')
tf.compat.v1.flags.DEFINE_boolean('output_weight_sharing', False, 'If True, the weights of the model are shared between embedding and output projection layer')
tf.compat.v1.flags.DEFINE_boolean('use_do', False, 'If True, use drop out on lstm cells in the encoder')
tf.compat.v1.flags.DEFINE_float('do_prob', 0.2, 'Dropout probability in lstm cells')

tf.compat.v1.flags.DEFINE_boolean('pretrained_embeddings', False, 'use pretrained embeddings')
tf.compat.v1.flags.DEFINE_string('embeddings_path', '', 'path to plain text embedding files')

tf.compat.v1.flags.DEFINE_boolean('pubmed', False, 'pubmed data')

tf.compat.v1.flags.DEFINE_string('optimizer', 'adagrad', 'optimizer can be `adagrad`, `adam` or `sgd`')
tf.compat.v1.flags.DEFINE_boolean('multi_layer_encoder', False, 'whether encoder is a multilayer LSTM')
tf.compat.v1.flags.DEFINE_integer('enc_layers', 1, 'number of encoder layers')

tf.compat.v1.flags.DEFINE_boolean('debug', False, 'debug mode')
tf.compat.v1.flags.DEFINE_string('ui_type', 'curses', "Command-line user interface type (curses | readline)")
tf.compat.v1.flags.DEFINE_string('dump_root', "/home/arman/ext1/tmp/tfdbg/", "Location for dumping tfdbg logs")


def main(unused_argv):
	if len(unused_argv) != 1:  # prints a message if you've entered flags incorrectly
		raise Exception("Problem with flags: %s" % unused_argv)

	# choose what level of logging you want
    # (unimplemented)
	# Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if
	# necessary
    # (unimplemented)

    # read in vocabs
	vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)  # create a vocabulary

	# If in decode mode, set batch_size = beam_size
	# Reason: in decode mode, we decode one example at a time.
	# On each step, we have beam_size-many hypotheses in the beam, so we need
	# to make a batch of these hypotheses.
	if FLAGS.mode == 'decode':
		FLAGS.batch_size = FLAGS.beam_size

	# If single_pass=True, check we're in decode mode
	if FLAGS.single_pass and FLAGS.mode != 'decode':
		raise Exception(
			"The single_pass flag should only be True in decode mode")

	# Make a namedtuple hps, containing the values of the hyperparameters that
	# the model needs
	hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',
				   'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps', 'coverage', 'cov_loss_wt', 'pointer_gen', 'min_lr',
				   'max_abstract_len', 'min_abstract_len', 'max_article_sents',
				   'max_section_len','min_section_len','use_sections','max_article_len',
				   'max_intro_len', 'max_conclusion_len',
				   'max_intro_sents', 'max_conclusion_sents', 'max_section_sents',
				   'enc_layers', 'optimizer', 'multi_layer_encoder',
				   'num_sections', 'hier', 'phased_lstm', 'output_weight_sharing', 'use_do' ,'do_prob', 'embeddings_path', 'pretrained_embeddings', 'pubmed', 'num_gpus', 'split_intro', 'temperature']
	hps_dict = {}
	for key, val in list(FLAGS.__flags.items()):  # for each flag
		if key in hparam_list:  # if it's in the list
			hps_dict[key] = val.value  # add it to the dict
	hps = namedtuple("HParams", list(hps_dict.keys()))(**hps_dict)

	# Create a batcher object that will create minibatches of data
	batcher = Batcher(FLAGS.data_path, vocab, hps,
					  FLAGS.single_pass,
					  FLAGS.article_id_key,
					  FLAGS.article_key,
					  FLAGS.abstract_key,
					  FLAGS.labels_key,
					  FLAGS.section_names_key,
					  FLAGS.sections_key)

	tf.set_random_seed(111)  # a seed value for randomness

	if hps.mode == 'train':
		print("creating model...")
		model = SummarizationModel(hps, vocab, num_gpus=FLAGS.num_gpus)
		setup_training(model, batcher)
	elif hps.mode == 'eval':
		model = SummarizationModel(hps, vocab, num_gpus=FLAGS.num_gpus)
		run_eval(model, batcher, vocab, hps.hier)
	elif hps.mode == 'decode':
		decode_model_hps = hps  # This will be the hyperparameters for the decoder model
		# The model is configured with max_dec_steps=1 because we only ever run
		# one step of the decoder at a time (to do beam search). Note that the
		# batcher is initialized with max_dec_steps equal to e.g. 100 because
		# the batches need to contain the full summaries
		decode_model_hps = hps._replace(max_dec_steps=1)
		model = SummarizationModel(decode_model_hps, vocab, num_gpus=FLAGS.num_gpus)
		decoder = BeamSearchDecoder(model, batcher, vocab)
		decoder.decode()  # decode indefinitely (unless single_pass=True, in which case deocde the dataset exactly once)
	else:
		raise ValueError("The 'mode' flag must be one of train/eval/decode")



