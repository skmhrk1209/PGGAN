#!/bin/bash

# tensorflowのインストールパスを取得します。
# pipでインストールしていることを想定しています。
# ネイティブでもpyenvでも取得できるはず。多分。
# 使用しているTensorFlowのバージョンと一致したfreeze_graph.pyが必要なためです。
tfpath=`pip show tensorflow | grep "Location: \(.\+\)$" | sed 's/Location: //'`

# tensorflowパッケージに付属しているfreeze_graph.pyを使用してチェックポイントファイルから
# graph_def ProtocolBuffersファイルに変換します。
python $tfpath/tensorflow/python/tools/freeze_graph.py \
    --input_graph=celeba_dcgan_model/graph.pbtxt \
    --input_checkpoint=celeba_dcgan_model/model.ckpt-300001 \
    --output_node_names="celeba_dcgan_model/fakes" \
    --output_graph=celeba_dcgan_model/frozen_graph.pb \
