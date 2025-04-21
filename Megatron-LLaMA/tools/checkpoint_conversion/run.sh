#!/bin/bash

python ./llama_checkpoint_conversion.py \
--load_path "/home/dist/Llama-2-7b/" \
--save_path "/home/dist/llama-2-7b-base-weights/" \
--target_tensor_model_parallel_size 2 \
--target_pipeline_model_parallel_size 2 \
--target_data_parallel_size 2 \
--target_params_dtype "bf16" \
--make_vocab_size_divisible_by 128 \
--print-checkpoint-structure