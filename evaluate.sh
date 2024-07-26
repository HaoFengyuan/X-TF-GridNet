#!/usr/bin/env bash

set -eu

sep_scp=data/wsj0_2mix/tt/sys.scp
ref_scp=data/wsj0_2mix/tt/ref.scp

python nnet/evaluate.py --sep_scp $sep_scp --ref_scp $ref_scp
