#!/bin/bash

# Please change the following envrioment variables
# base on the cluster configuration
export OMP_NUM_THREADS=4
export MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export MUSA_KERNEL_TIMEOUT=1200000
export NCCL_PROTOS=2
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=$PYTHONPATH:/home/dist/zhiyuan-test/FlagScale/megatron

set -u
  PROJ_HOME=$1
  EXPNAME=$2
  HOSTFILE=$3
  DATA_DIR=$4
  TP_SIZE=$5
  PP_SIZE=$6
  WORLD_SIZE=$7
  MICRO_BATCH_SIZE=$8
  GLOBAL_BATCH_SIZE=$9
set +u

CHECKPOINT_PATH=$PROJ_HOME/checkpoints/$EXPNAME
mkdir -p $CHECKPOINT_PATH
DATA_PATH=$DATA_DIR/wudao_pretrain_text_document
VOCAB_FILE=../aquila/tokenizer/vocab.json
MERGE_FILE=../aquila/tokenizer/merges.txt
SPECIAL_TOKENS_FILE=../aquila/tokenizer/special_tokens.txt
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
export MASTER_PORT=12346
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT 
"

TRAINING_ARGS="
    --train-samples 11632640 \
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

MIXED_PRECISION_ARGS="
    --fp16 \
    --initial-loss-scale 65536 \
    --min-loss-scale 1.0 \
    --embedding-weights-in-fp32 \
    --attention-softmax-in-fp32 \
    --no-masked-softmax-fusion \
    --rotary-position-embeddings-in-fp32
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --tokenizer-type AquilaTokenizer \
    --vocab-file $VOCAB_FILE \
    --vocab-size 100008\
    --merge-file $MERGE_FILE \
    --special-tokens-file $SPECIAL_TOKENS_FILE \
    --data-impl mmap \
    --split 1
"

NETWORK_ARGS="
    --num-layers 64 \
    --hidden-size 5120 \
    --num-attention-heads 40 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --layernorm-epsilon 1e-6 \
    --layernorm-init-weight 0.3 \
    --use-rotary-position-embeddings \
    --no-position-embedding \
    --swiglu \
    --apply-layernorm-rms \
    --untie-embeddings-and-output-weights
"

INITIALIZATION_ARGS="
    --init-method-std 0.0165 \
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
    --lr-warmup-samples 11468800 \
    --min-lr 1.5e-6
"

CHECKPOINTING_ARGS="
    --save-interval 30 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
"

LOGGING_ARGS="
    --log-interval 1
"

cmd="torchrun $DISTRIBUTED_ARGS /home/dist/zhiyuan-test/FlagScale/megatron/pretrain_gpt.py \
              $TRAINING_ARGS \
              $MIXED_PRECISION_ARGS \
              $DATA_ARGS \
              $NETWORK_ARGS \
              $INITIALIZATION_ARGS \
              $REGULARIZATION_ARGS \
              $LEARNING_RATE_ARGS \
              $CHECKPOINTING_ARGS \
              $LOGGING_ARGS
    "
echo $cmd
eval $cmd
