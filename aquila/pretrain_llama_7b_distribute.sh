#!/bin/bash

# Please change the following envrioment variables
# base on the cluster configuration
export OMP_NUM_THREADS=4
export MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export MUSA_KERNEL_TIMEOUT=5400000
export MCCL_PROTOS=2
export MUSA_PRINT_ENV=1
export MCCL_PEER_ACCESS_IPC_FLAG=5
export MCCL_PEER_ACCESS_FLAG=4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export LD_LIBRARY_PATH=/usr/mpi/gcc/openmpi-4.1.5rc2/lib:/home/apex/amp_C/lib:$LD_LIBRARY_PATH
export MCCL_DEBUG=WARNING
export MASS_MODEL_PARAMETER="7b"

TP_SIZE=1
PP_SIZE=1
WORLD_SIZE=8
MICRO_BATCH_SIZE=1
NUM_MICROBATCHES=128
(( DP_SIZE = $WORLD_SIZE / ($TP_SIZE * $PP_SIZE) ))
(( GLOBAL_BATCH_SIZE = $MICRO_BATCH_SIZE * $NUM_MICROBATCHES * $DP_SIZE ))

PROJ_HOME=$(pwd)
EXPNAME="tp${TP_SIZE}_pp${PP_SIZE}_dp${DP_SIZE}_mbs${MICRO_BATCH_SIZE}_numbs${NUM_MICROBATCHES}_gbs${GLOBAL_BATCH_SIZE}_gpus${WORLD_SIZE}"
HOSTFILE=./hostfile
DATA_DIR="../data/RedPajama-Data-1T-Sample/"
DATA_CACHE_PATH=../data/datasets-test/
PRETRAIN_FILE_PATH="$(dirname "$(pwd)")/megatron/pretrain_gpt.py"

export PYTHONPATH=$PYTHONPATH:$(dirname $PROJ_HOME)

CHECKPOINT_PATH=$PROJ_HOME/checkpoints/$EXPNAME
mkdir -p $CHECKPOINT_PATH

DATA_FILE_PREFIX=$(ls $DATA_DIR | grep ".idx" | head -n 1 | sed "s/.idx//")
DATA_PATH=$DATA_DIR/$DATA_FILE_PREFIX

VOCAB_FILE=../aquila/tokenizer/vocab.json
MERGE_FILE=../aquila/tokenizer/merges.txt
SPECIAL_TOKENS_FILE=../aquila/tokenizer/special_tokens.txt
TOKENIZER_MODEL=$PROJ_HOME/llama_config/tokenizer.model

LOG_PATH=$PROJ_HOME/logs/$EXPNAME
mkdir -p $LOG_PATH
cp $0 $LOG_PATH/

TB_PATH=$PROJ_HOME/tboard/$EXPNAME
mkdir -p $TB_PATH
WB_PATH=$PROJ_HOME/wandb/$EXPNAME
mkdir -p $WB_PATH

export NODE_ADDR=$(ifconfig -a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2;}'|tr -d "addr:"|head -n 1)
export GPUS_PER_NODE=$(awk '{$1=$1;print}' $HOSTFILE|awk -F" |=" '{ranks[$1]=$NF;}END{print ranks["'$NODE_ADDR'"];}')
export NNODES=$(cat $HOSTFILE | wc -l)
export MASTER_ADDR=$(head -n1 $HOSTFILE | awk '{print $1;}')
export NODE_RANK=$(awk '{ranks[$1]=(FNR-1);}END{print ranks["'$NODE_ADDR'"];}' $HOSTFILE)
export MASTER_PORT=12361
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

CURRENT_TIME=$(date "+%Y-%m-%d_%H:%M:%S")
mkdir -p ./output/$CURRENT_TIME
LOG_FILE=./output/$CURRENT_TIME/$EXPNAME.log

DISTRIBUTED_ARGS="
    --nproc_per_node 8 \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

TRAINING_ARGS="
    --train-samples 24414062 \
    --eval-iters 0 \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --no-gradient-accumulation-fusion \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --disable-bias-linear \
    --use-distributed-optimizer \
    --distributed-backend mccl \
    --use-flash-attn \
    --sequence-parallel \
    --device-type mthreads
 "

    #  --recompute-granularity full \
    # --recompute-method block \
    # --recompute-num-layers 1

MIXED_PRECISION_ARGS="
    --bf16 \
    --embedding-weights-in-fp32 \
    --attention-softmax-in-fp32 \
    --no-masked-softmax-fusion \
    --rotary-position-embeddings-in-fp32
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --data-cache-path $DATA_CACHE_PATH \
    --vocab-file $VOCAB_FILE \
    --vocab-size 100008 \
    --merge-file $MERGE_FILE \
    --special-tokens-file $SPECIAL_TOKENS_FILE \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model $TOKENIZER_MODEL \
    --data-impl mmap \
    --split 1
"

NETWORK_ARGS="
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --layernorm-epsilon 1e-5 \
    --use-rotary-position-embeddings \
    --no-position-embedding \
    --swiglu \
    --multiple-of 256 \
    --normalization RMSNorm \
    --apply-layernorm-rms \
    --untie-embeddings-and-output-weights
"

INITIALIZATION_ARGS="
    --init-method-std 0.02 \
    --seed 42
"

REGULARIZATION_ARGS="
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --clip-grad 1.0
"

LEARNING_RATE_ARGS="
    --lr 1.5e-5 \
    --lr-decay-style cosine \
    --lr-warmup-samples 14745600 \
    --min-lr 1.5e-6 \
    --initial-loss-scale 65536 \
    --min-loss-scale 1.0
"

CHECKPOINTING_ARGS="
    --save-interval 100 \
    --save $CHECKPOINT_PATH
"
#  --save-interval 60 \
#     --save $CHECKPOINT_PATH \
#     --load $CHECKPOINT_PATH
LOGGING_ARGS="
    --log-interval 1
"

cmd="torchrun $DISTRIBUTED_ARGS $PRETRAIN_FILE_PATH \
              $TRAINING_ARGS \
              $MIXED_PRECISION_ARGS \
              $DATA_ARGS \
              $NETWORK_ARGS \
              $INITIALIZATION_ARGS \
              $REGULARIZATION_ARGS \
              $LEARNING_RATE_ARGS \
              $CHECKPOINTING_ARGS \
              $LOGGING_ARGS > ./output/$CURRENT_TIME/$EXPNAME.log 2>&1
    "
echo $cmd
eval $cmd
