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

keras_model = "../checkpoints/conv_1d_time_stacked_model/ep-098.hdf5"

model = load_model(keras_model)

converter = tf.lite.TFLiteConverter
converter = converter.from_keras_model(model)
tflite_model = converter.convert()
open("converted_speech_model.tflite", "wb").write(tflite_model)
