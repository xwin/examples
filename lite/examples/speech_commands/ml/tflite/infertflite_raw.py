#! python

import os
import argparse
import numpy as np
import tensorflow as tf
import operator
# disable GPU use
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def read_labels(name) :
    file = open(name)
    lines = file.read().split('\n')
    file.close()
    return lines

def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)

def quantize_audio(audio_samples):
    x = tf.math.multiply(audio_samples, 128)
    x = tf.math.rint(x)
    x = tf.cast(x, tf.int8)
    return x

def dequantize(x, zero_point, scale):
    x = x.astype(np.float32)
    return scale * (x - zero_point)

def do_save_input(wav_file, data):
    (dirname, filename) = os.path.split(wav_file)
    (name, ext) = os.path.splitext(filename)
    data.numpy().tofile(name+".tensor", "\n", format="%f")
    
def main(FLAGS, unparsed) :
    model="../export/converted_speech_model.tflite"
    labels="./conv_actions_labels_keras.txt"
    
    lablist = read_labels(labels)
    
    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    isquantized = not issubclass(input_details[0]['dtype'], np.float32)
    if (isquantized == True) :
        scale, zero_point = output_details[0]['quantization']

    for file_path in unparsed :
        audio_binary = tf.io.read_file(file_path)
        audio_samples = decode_audio(audio_binary)
        if (audio_samples.shape[0] != 16000) :
            continue;
        audio_samples = tf.expand_dims(audio_samples, axis=0)
        if (isquantized == True) :
            audio_samples = quantize_audio(audio_samples)
        if (FLAGS.save_input) :
            do_save_input(file_path, audio_samples)
        interpreter.set_tensor(input_details[0]['index'], audio_samples)
        interpreter.invoke()

        # output is a softmax of the detected keywords
        output = interpreter.get_tensor(output_details[0]['index'])
        if (isquantized == True) :
            output = dequantize(output, zero_point, scale)
        output = output.squeeze() * 100
        result = dict(zip(lablist, output.tolist()))
        top_pred, top_val = max(result.items(), key=operator.itemgetter(1))
        print(file_path, top_pred, top_val)
    
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-save-input', action='store_true', dest='save_input', default=False)
    FLAGS, unparsed = parser.parse_known_args();
    main(FLAGS, unparsed)
