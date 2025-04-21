# Path: FlagScale/megatron/megatron/model_studio_monitor.py

import torch

nan_count = 0


def check_nan(input_tensor, desc):
    global nan_count
    if nan_count > 2000:
        pass
    if input_tensor is not None and torch.isnan(input_tensor).any():
        import time

        ts = time.time()
        rank = torch.distributed.get_rank()
        device = input_tensor.device
        print(
            "check_nan_ts#{}#{}#{}#{}#{}".format(
                ts, desc, rank, device, (time.time() - ts)
            ),
            flush=True,
        )
        if nan_count < 10:
            write_loss_nan(rank, ts, desc, device)
        nan_count = nan_count + 1


def write_loss_nan(rank, ts, desc, device):
    with open("/etc/mccflow/nan_monitor", "a", encoding="utf-8") as f:
        f.write("nan#{}#{}#{}#{}\n".format(ts, desc, rank, device))


def make_success():
    file = open("/etc/mccflow/success", "w", encoding="utf-8")
    file.close()


import subprocess
from subprocess import getoutput
import time
import os


def run_cmd(cmd, interval, outputstream):
    while True:
        subprocess.Popen(cmd, shell=True, stdout=outputstream, stderr=subprocess.STDOUT)
        time.sleep(interval)


def gpu_monitor_collect():
    gpu_cmd = "grep -v 'CST' /etc/mccflow/gpu_log* | sed 's/[^0-9| |.]//g' | \
                awk 'BEGIN{gpu_ratio_total=0;gpu_power_total=0;count=0} \
                {count=count+1;gpu_power_total=gpu_power_total+$2;gpu_ratio_total=gpu_ratio_total+$5} \
                END {print gpu_power_total\" \"gpu_power_total/count\" \"gpu_ratio_total/count}'"
    ret = getoutput(gpu_cmd)
    rets = ret.split(" ")
    if len(rets) != 3:
        return 0.0, 0.0, 0.0
    os.system("rm -Rf /etc/mccflow/gpu_log*")
    return float(rets[0]), float(rets[1]), float(rets[2])
