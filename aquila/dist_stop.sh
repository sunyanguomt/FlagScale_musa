#!/bin/bash

if [ $# -eq 0 ]; then
  echo "Usage: $0 <hostfile>"
  exit 1
fi

set -u
    HOSTFILE=$1
set +u
NUM_NODES=$(grep -v '^#\|^$' $HOSTFILE | wc -l)
echo "NUM_NODES: $NUM_NODES"

hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)
for host in ${hostlist[@]}; do
    ssh $host "pkill -f '/opt/conda/envs/py38/bin/python /opt/conda/envs/py38/bin/torchrun'" 
    echo "$host is killed."
done
