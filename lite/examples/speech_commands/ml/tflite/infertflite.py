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

def main(FLAGS, unparsed) :
    model="./conv_actions_frozen.tflite"
    labels="./conv_actions_labels.txt"
    
    lablist = read_labels(labels)
    
    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    for file_path in unparsed :
        audio_binary = tf.io.read_file(file_path)
        audio_samples, _ = tf.audio.decode_wav(audio_binary)
        sample_rate = np.array((16000,), dtype=np.int32)

        interpreter.set_tensor(input_details[0]['index'], audio_samples)
        interpreter.set_tensor(input_details[1]['index'], sample_rate)
        interpreter.invoke()

        # output is a softmax of the detected keywords
        output = interpreter.get_tensor(output_details[0]['index'])
        output = output.squeeze() * 100
        result = dict(zip(lablist, output.tolist()))
        top_pred, top_val = max(result.items(), key=operator.itemgetter(1))
        print(top_pred, top_val)
    
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    FLAGS, unparsed = parser.parse_known_args();
    main(FLAGS, unparsed)
