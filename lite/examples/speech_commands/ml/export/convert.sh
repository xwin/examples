#! /bin/bash

env PYTHONPATH=../ python convert_keras_lite.py -quantize -integer -no-float
