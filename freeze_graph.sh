#!/bin/bash

tfpath=`pip show tensorflow | grep "Location: \(.\+\)$" | sed 's/Location: //'`

python $tfpath/tensorflow/python/tools/freeze_graph.py \
    --input_graph=$1/graph.pbtxt \
    --input_checkpoint=$1/model.ckpt \
    --output_graph=$1/frozen_graph.pb \
    --output_node_names=fakes \
