# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from keras.models import load_model
import numpy as np
import argparse
import os.path


def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)

def wav_feeder():
    for wav_file in wav_files :
        if ( not os.path.isfile(wav_file)) :
            continue

        audio_binary = tf.io.read_file(wav_file)
        audio_samples = decode_audio(audio_binary)
        audio_samples = tf.expand_dims(audio_samples, axis=0)
        if( audio_samples.shape[1] != 16000) :
            continue
        yield [audio_samples]

# fake representative dataset        
def representative_dataset_gen():
    for _ in range(100):
        data = np.random.uniform(-1, 1, (1,16000))
        yield [data.astype(np.float32)]

def gen_filename_list(input_filename):
    file = open(input_filename)
    lines = file.read().split('\n')
    file.close()
    dir_name = os.path.dirname(input_filename) +'/'
    lines = [dir_name + sub for sub in lines if sub.strip() != '']
    return lines

def main(FLAGS, unparsed) :
    input_list = '../data/train/validation_list.txt'
    global wav_files
    wav_files = gen_filename_list(input_list)
    
    keras_model = "../checkpoints/conv_1d_time_stacked_model/ep-096.hdf5"

    model = load_model(keras_model, custom_objects={'tf': tf})

    converter = tf.lite.TFLiteConverter
    converter = converter.from_keras_model(model)
    converter.experimental_new_converter = False
    if (FLAGS.quantize == True ) :
        # This enables quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if (FLAGS.integer == True) :
            # This sets the representative dataset for quantization
            converter.representative_dataset = wav_feeder
            # This ensures that if any ops can't be quantized, the converter throws an error
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            if (FLAGS.no_float) :
                # For full integer quantization, though supported types defaults to int8 only,
                # we explicitly declare it for clarity.
                converter.target_spec.supported_types = [tf.int8]
                # These set the input and output tensors to int8
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    open("converted_speech_model.tflite", "wb").write(tflite_model)

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-quantize', action='store_true', default=False)
    parser.add_argument('-integer', action='store_true', default=False)
    parser.add_argument('-no-float', action='store_true', dest='no_float',default=False)
    FLAGS, unparsed = parser.parse_known_args();
    main(FLAGS, unparsed)
