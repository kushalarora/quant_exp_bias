#!/bin/sh
export PYTHONPATH=.:${PYTHONPATH}
source ~/scratch/envs/qeb/bin/activate
mkdir ${1}/ood
for domain in acquis it subtitles emea koran; do
  python quant_exp_bias/run.py evaluate ${1}/training data/domain-adaptation-data/${domain}-filtered.tar.gz --output-file ${1}/ood/${domain}.txt --cuda-device 0 2>&1 | tee ${1}/ood/out-${domain}.log
done