# 基于FlagScale的Pretraining

## 1.llama系列模型的Pretraining

### 1.1 确定使用的IP地址
在hostfile里面填入只用服务器容器的ip地址例如下面只用到了1台服务器
```bash
10.68.204.91 slots=8
```

### 1.2 调整参数与路径

在dist_start_7B_8gpus里面进行修改:

修改参数
```bash
TP_SIZE=2
PP_SIZE=2
WORLD_SIZE=8
MICRO_BATCH_SIZE=2
NUM_MICROBATCHES=2
```

修改DATA_PATH和hostfile的路径
```bash
DATA_PATH=/home/dist/dataset/oscar

HOSTFILE=./hostfile
```

### 1.3 启动训练脚本

```bash
bash dist_start_7B_8gpus.sh
```

在output里面查看对应的log结合显卡的利用来确定是不是启动了起来。

```bash
 iteration        1/   23841 | consumed samples:         1024 | elapsed time per iteration (ms): 438288.1 | mfu: 59.8109 | chip_throughput: 1196.22 /s  learning rate: 1.875E-09 | global batch size:  1024 | lm loss: 1.120865E+01 | loss scale: 65536.0 | max_total_norm_before_clip: 0.012 | grad norm: 7.549 | max_total_norm_after_clip: 0.002 | grad_norm_after_clip: 1.000 | number of skipped iterations:   0 | number of nan iterations:   0 |
[Rank 5] (after 1 iterations) memory (MB) | allocated: 0.0 | max allocated: 0.0 | reserved: 0.0 | max reserved: 0.0
[Rank 4] (after 1 iterations) memory (MB) | allocated: 0.0 | max allocated: 0.0 | reserved: 0.0 | max reserved: 0.0
 iteration        2/   23841 | consumed samples:         2048 | elapsed time per iteration (ms): 428149.6 | mfu: 61.2272 | chip_throughput: 1224.54 /s  learning rate: 3.750E-09 | global batch size:  1024 | lm loss: 1.118107E+01 | loss scale: 65536.0 | max_total_norm_before_clip: 0.019 | grad norm: 10.353 | max_total_norm_after_clip: 0.002 | grad_norm_after_clip: 1.000 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration        3/   23841 | consumed samples:         3072 | elapsed time per iteration (ms): 418467.3 | mfu: 62.6439 | chip_throughput: 1252.88 /s  learning rate: 5.625E-09 | global batch size:  1024 | lm loss: 1.118218E+01 | loss scale: 65536.0 | max_total_norm_before_clip: 0.019 | grad norm: 10.330 | max_total_norm_after_clip: 0.002 | grad_norm_after_clip: 1.000 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration        4/   23841 | consumed samples:         4096 | elapsed time per iteration (ms): 429151.3 | mfu: 61.0843 | chip_throughput: 1221.69 /s  learning rate: 7.500E-09 | global batch size:  1024 | lm loss: 1.118403E+01 | loss scale: 65536.0 | max_total_norm_before_clip: 0.019 | grad norm: 10.314 | max_total_norm_after_clip: 0.002 | grad_norm_after_clip: 1.000 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration        5/   23841 | consumed samples:         5120 | elapsed time per iteration (ms): 425091.6 | mfu: 61.6677 | chip_throughput: 1233.35 /s  learning rate: 9.375E-09 | global batch size:  1024 | lm loss: 1.118148E+01 | loss scale: 65536.0 | max_total_norm_before_clip: 0.019 | grad norm: 10.663 | max_total_norm_after_clip: 0.002 | grad_norm_after_clip: 1.000 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration        6/   23841 | consumed samples:         6144 | elapsed time per iteration (ms): 428634.9 | mfu: 61.1579 | chip_throughput: 1223.16 /s  learning rate: 1.125E-08 | global batch size:  1024 | lm loss: 1.118147E+01 | loss scale: 65536.0 | max_total_norm_before_clip: 0.019 | grad norm: 10.366 | max_total_norm_after_clip: 0.002 | grad_norm_after_clip: 1.000 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration        7/   23841 | consumed samples:         7168 | elapsed time per iteration (ms): 417291.2 | mfu: 62.8204 | chip_throughput: 1256.41 /s  learning rate: 1.313E-08 | global batch size:  1024 | lm loss: 1.118180E+01 | loss scale: 65536.0 | max_total_norm_before_clip: 0.020 | grad norm: 10.428 | max_total_norm_after_clip: 0.002 | grad_norm_after_clip: 1.000 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration        8/   23841 | consumed samples:         8192 | elapsed time per iteration (ms): 430383.7 | mfu: 60.9094 | chip_throughput: 1218.19 /s  learning rate: 1.500E-08 | global batch size:  1024 | lm loss: 1.118307E+01 | loss scale: 65536.0 | max_total_norm_before_clip: 0.019 | grad norm: 10.308 | max_total_norm_after_clip: 0.002 | grad_norm_after_clip: 1.000 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration        9/   23841 | consumed samples:         9216 | elapsed time per iteration (ms): 425414.7 | mfu: 61.6208 | chip_throughput: 1232.42 /s  learning rate: 1.688E-08 | global batch size:  1024 | lm loss: 1.118129E+01 | loss scale: 65536.0 | max_total_norm_before_clip: 0.020 | grad norm: 10.342 | max_total_norm_after_clip: 0.002 | grad_norm_after_clip: 1.000 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration       10/   23841 | consumed samples:        10240 | elapsed time per iteration (ms): 420717.4 | mfu: 62.3088 | chip_throughput: 1246.18 /s  learning rate: 1.875E-08 | global batch size:  1024 | lm loss: 1.117971E+01 | loss scale: 65536.0 | max_total_norm_before_clip: 0.020 | grad norm: 10.375 | max_total_norm_after_clip: 0.002 | grad_norm_after_clip: 1.000 | number of skipped iterations:   0 | number of nan iterations:   0 |
```
### 1.4 计算均值MFU

使用aquila下面的cal_mfu_tps.py来进行计算均值的MFU
```bash
python cal_mfu_tps.py \
  --log_path ./output/2024-09-05_18\:40\:24/tp2_pp2_dp2_mbs2_numbs256_gbs1024_gpus8.log.1.10.72.180.138.log \
  --global_batchsize 1024 \
  --parameters_count 7000000000 \
  --seq_length 4096 \
  --gpu_numbers 8 \
  --hidden_size 4096
```

得到的均值MFU和吞吐量如下：
```bash
Calculated MFU is: 0.615
Calculated throughout is : 1230.674 tokens/s
```

### 1.5 停止掉当前的预训练

```bash
# 这里需要注意跟上你使用的hostfile
bash dist_stop.sh hostfile
```


### 1.6 Pretraining的性能数据

见当前的performance.csv