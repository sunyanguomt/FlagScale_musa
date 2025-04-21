import re
import argparse


def calculate_mfu(
    parameters_count: int,
    batch_size: int,
    seq_length: int,
    second_per_step: float,
    gpu_numbers: int,
    hidden_size: int,
    ffn_hidden_size: int,
    num_hidden_layers: int,
    vocab_size: int
) -> float:
    '''
    flops_per_step = (
        6 * parameters_count * batch_size * seq_length / (second_per_step * gpu_numbers)
    )
    '''
    flops_per_step = 6 * seq_length * batch_size * hidden_size * (3 * num_hidden_layers * ffn_hidden_size + 4 * num_hidden_layers * hidden_size + vocab_size)
    MFU = flops_per_step / (gpu_numbers * second_per_step * 458 * 10**12)
    return MFU


def calculate_mfu_from_files(
    file_path: str,
    parameters_count: int,
    batch_size: int,
    seq_length: int,
    gpu_numbers: int,
    hidden_size: int,
    ffn_hidden_size: int,
    num_hidden_layers: int,
    vocab_size: int
) -> float:
    mfu_collections = []
    pattern = r"elapsed time per iteration \(ms\): (\d+\.\d+)"

    with open(file_path, errors='replace') as f:
        for line in f.readlines():
            match = re.search(pattern, line)
            if match and line.strip().startswith("iteration"):
                time_per_iteration_s = float(match.group(1)) / 1000
                mfu = calculate_mfu(
                    parameters_count,
                    batch_size,
                    seq_length,
                    time_per_iteration_s,
                    gpu_numbers,
                    hidden_size,
                    ffn_hidden_size,
                    num_hidden_layers,
                    vocab_size
                )
                mfu_collections.append(mfu)

    return sum(mfu_collections) / len(mfu_collections)


def calculate_throughput_from_log(
    file_path: str, global_batchsize: int, seq_length: int, gpu_numbers: int
) -> float:
    time_collections = []
    pattern = r"elapsed time per iteration \(ms\): (\d+\.\d+)"

    with open(file_path, errors='replace') as f:
        for line in f.readlines():
            match = re.search(pattern, line)
            if match and line.strip().startswith("iteration"):
                time_per_iteration_s = float(match.group(1)) / 1000
                time_collections.append(time_per_iteration_s)

    average_time_per_step = sum(time_collections) / len(time_collections)

    throughput = global_batchsize * seq_length / (gpu_numbers * average_time_per_step)
    return throughput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_path",
        default="xinghuan_tp4_pp8_dp2_mbs1_numbs256_gbs512_gpus64.log.7.10.67.106.62",
        type=str,
        help="the file path",
    )
    parser.add_argument(
        "--parameters_count",
        default=70 * 10**9,
        type=int,
        help="the number of parameters of the model",
    )
    parser.add_argument(
        "--global_batchsize", default=1024, type=int, help="global batch size"
    )
    parser.add_argument("--seq_length", default=4096, type=int, help="sequence length")
    parser.add_argument(
        "--gpu_numbers",
        default=8,
        type=int,
        help="the number of gpu used for training",
    )
    parser.add_argument(
        "--hidden_size", default=4096, type=int, help="hidden size of the model"
    )
    parser.add_argument(
        "--ffn_hidden_size", default=11088, type=int, help="ffn hidden size of the model"
    )
    parser.add_argument(
        "--num_hidden_layers", default=32, type=int, help="layers of the model"
    )
    parser.add_argument(
        "--vocab_size", default=32000, type=int, help="vocal size of the model"
    )




    return parser.parse_args()


def _main() -> None:
    args = parse_args()
    parameters_count = args.parameters_count
    batch_size = args.global_batchsize
    seq_length = args.seq_length
    gpu_numbers = args.gpu_numbers
    hidden_size = args.hidden_size
    ffn_hidden_size = args.ffn_hidden_size
    num_hidden_layers = args.num_hidden_layers
    vocab_size = args.vocab_size

    mfu = calculate_mfu_from_files(
        args.log_path,
        parameters_count,
        batch_size,
        seq_length,
        gpu_numbers,
        hidden_size,
        ffn_hidden_size,
        num_hidden_layers,
        vocab_size
    )

    throughout = calculate_throughput_from_log(
        args.log_path, batch_size, seq_length, gpu_numbers
    )

    print(f"Calculated MFU is: {round(mfu, 3)}")
    print(f"Calculated throughout is : {round(throughout, 3)} tokens/s")
    print(f"Calculated throughout per device is : {round(throughout/8, 3)} tokens/s")


if __name__ == "__main__":
    _main()
