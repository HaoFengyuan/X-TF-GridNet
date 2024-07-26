#!/usr/bin/env bash

set -eu

configs_path=configs/train_config.toml

python nnet/train.py --config $configs_path
