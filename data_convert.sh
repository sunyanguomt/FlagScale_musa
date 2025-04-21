#!/bin/bash

python megatron/tools/preprocess_data.py \
       --input /home/dist/devtech_files/datasets/RedPajama-Data-1T-Sample/merged_file.jsonl \
       --output-prefix /home/dist/RedPajama/RedPajama \
       --tokenizer-model /home/dist/Llama-2-7b/tokenizer.model \
       --tokenizer-type Llama2Tokenizer \
       --workers 1
