#!/usr/bin/env bash

set -eu

configs_path=configs/train_config.toml
data_type=wsj0_2mix
cpt_dir=exp/pTFGridNet_3.5

python nnet/decode.py --config $configs_path --data_type $data_type --cpt_dir $cpt_dir
