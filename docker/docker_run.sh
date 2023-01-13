#!/bin/sh
GGS_PATH=$1
DATA_PATH=$2

docker run --gpus all --name ggs-container -t -v $GGS_PATH/greedy-grid-search/:/greedy-grid-search -v $DATA_PATH:/data --shm-size=8182m ggs-image
