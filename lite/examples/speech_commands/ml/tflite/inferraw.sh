#! /bin/bash

WFILES="../data/train/left/b1f8326d_nohash_0.wav \
       ../data/train/yes/aff582a1_nohash_0.wav \
       ../data/train/five/ff21fb59_nohash_1.wav \
       ../data/train/up/ffd2ba2f_nohash_3.wav \
       ../data/train/up/b16f2d0d_nohash_1.wav \
       ../data/train/wow/ffb86d3c_nohash_0.wav"

#WFILES="alex/yes/*.wav"
#WFILES="alex/no/*.wav"
#WFILES="alex/on/*.wav"
#WFILES="alex/off/*.wav"
#WFILES="alex/stop/*.wav"
#WFILES="alex/go/*.wav"
#WFILES="alex/up/*.wav"
#WFILES="alex/down/*.wav"
#WFILES="alex/right/*.wav"
#WFILES="alex/left/*.wav"
#WFILES="../data/train/five/*.wav"

python infertflite_raw.py \
       ${WFILES}
