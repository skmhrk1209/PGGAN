#!/bin/bash

tfpath=`pip show tensorflow | grep "Location: \(.\+\)$" | sed 's/Location: //'`

python $tfpath/tensorflow/python/tools/freeze_graph.py \
    --input_graph=celeba_dcgan_model/graph.pbtxt \
    --input_checkpoint=celeba_dcgan_model/model.ckpt-300001 \
    --output_node_names=celeba_dcgan_model/fakes \
    --output_graph=celeba_dcgan_model/frozen_graph.pb \
