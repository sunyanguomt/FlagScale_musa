#!/bin/bash

CURRENT_TIME=$(date "+%Y-%m-%d_%H:%M:%S")
echo $CURRENT_TIME
mkdir -p ./output/$CURRENT_TIME

TP_SIZE=2
PP_SIZE=2
WORLD_SIZE=8
MICRO_BATCH_SIZE=2
NUM_MICROBATCHES=256
(( DP_SIZE = $WORLD_SIZE / ($TP_SIZE * $PP_SIZE) ))
(( GLOBAL_BATCH_SIZE = $MICRO_BATCH_SIZE * $NUM_MICROBATCHES * $DP_SIZE ))

set -u
  FLAGSCALE_HOME=$(pwd)
  PROJ_HOME=$(pwd)
  PRETRAIN_FILE_PATH="$(dirname "$(pwd)")/megatron/pretrain_gpt.py"
  EXPNAME="tp${TP_SIZE}_pp${PP_SIZE}_dp${DP_SIZE}_mbs${MICRO_BATCH_SIZE}_numbs${NUM_MICROBATCHES}_gbs${GLOBAL_BATCH_SIZE}_gpus${WORLD_SIZE}"
  DATA_PATH=/home/dist/dataset/oscar
  DATA_CACHE_PATH=/home/dist/datasets/mcc-dataset-test
  HOSTFILE=./hostfile
  LOG_FILE=./output/$CURRENT_TIME/$EXPNAME.log
  SCRIPT_FILE=./7B/pretrain_llama_7b_distribute.sh
set +u

cmd="bash -c 'cd $FLAGSCALE_HOME; \
     bash $SCRIPT_FILE $PROJ_HOME $EXPNAME $HOSTFILE \"$DATA_PATH\" \
     $TP_SIZE $PP_SIZE $WORLD_SIZE \
     $MICRO_BATCH_SIZE $GLOBAL_BATCH_SIZE \
     $DATA_CACHE_PATH $PRETRAIN_FILE_PATH"

COUNT=0
hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)
hostlen=$(cat $HOSTFILE | wc -l )

for host in ${hostlist[@]}; do
    ssh $host "pkill -9 torchrun"
    ssh $host "pkill -9 python"
    echo "$host is killed."
done


set -e
for host in ${hostlist[@]}; do
  COUNT=$((COUNT+1))
  echo $COUNT","$hostlen
  if [ $COUNT -eq $hostlen ]
  then
    cmd_ssh=$cmd" > $LOG_FILE.$COUNT.$host.log 2>&1 '"
  else
    cmd_ssh=$cmd" > $LOG_FILE.$COUNT.$host.log 2>&1 &'"
  fi
  echo $cmd_ssh
  ssh $host $MAAS_ENVS $cmd_ssh
  echo $?
done
