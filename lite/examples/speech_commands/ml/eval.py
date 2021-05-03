#! python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
from model import speech_model, prepare_model_settings
from generator import prepare_words_list
from classes import get_classes
from utils import data_gen
import operator

# disable GPU use
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser(description='set input arguments')

parser.add_argument(
    '-sample_rate',
    action='store',
    dest='sample_rate',
    type=int,
    default=16000,
    help='Sample rate of audio')
parser.add_argument(
    '-batch_size',
    action='store',
    dest='batch_size',
    type=int,
    default=32,
    help='Size of the training batch')
parser.add_argument(
    '-output_representation',
    action='store',
    dest='output_representation',
    type=str,
    default='raw',
    help='raw, spec, mfcc or mfcc_and_raw')
parser.add_argument('-file_names', dest='file_names', nargs='+')

args = parser.parse_args()
# parser.print_help()
print('input args: ', args)

def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)

if __name__ == '__main__':
    output_representation = args.output_representation
    sample_rate = args.sample_rate
    batch_size = args.batch_size
    classes = get_classes(wanted_only=True)
    model_settings = prepare_model_settings(
        label_count=len(prepare_words_list(classes)),
        sample_rate=sample_rate,
        clip_duration_ms=1000,
        window_size_ms=30.0,
        window_stride_ms=10.0,
        dct_coefficient_count=80,
        num_log_mel_features=60,
        output_representation=output_representation)

    print(model_settings)

    model = speech_model(
        'conv_1d_time_stacked',
        model_settings['fingerprint_size']
      if output_representation != 'raw' else model_settings['desired_samples'],
        # noqa
      num_classes=model_settings['label_count'],
        **model_settings)

    checkpoints_path = os.path.join('checkpoints', 'conv_1d_time_stacked_model')
    model.load_weights(checkpoints_path + '/ep-096.hdf5')

    labels = prepare_words_list(classes)
    for file_name in args.file_names :
        audio_binary = tf.io.read_file(file_name)
        audio_samples = decode_audio(audio_binary)
        audio_samples = tf.expand_dims(audio_samples, axis=0)
    
        eval_res = model(audio_samples)
        eval_res = tf.squeeze(eval_res)
        np_res = eval_res.numpy()
        np_res = np_res * 100
        result = dict(zip(labels, np_res.tolist()))
        top_pred, top_val = max(result.items(), key=operator.itemgetter(1))
        print(file_name, top_pred, top_val)
