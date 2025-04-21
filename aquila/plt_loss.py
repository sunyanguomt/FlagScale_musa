import matplotlib.pyplot as plt
import re
import argparse

def plot_loss_curve(file_name, output_file):
    # 用于存储迭代次数和对应的loss值
    iterations = []
    losses = []

    # 根据提供的日志条目格式调整的正则表达式模式
    pattern = re.compile(r'iteration\s+(\d+)/\s*\d+\s+\|\s+consumed samples:\s+\d+\s+\|\s+.*?\|\s+lm loss:\s+([0-9.]+)E[+-]\d+')

    # 打开并读取日志文件
    with open(file_name, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                iteration = int(match.group(1))
                loss = float(match.group(2)) * 10 ** int(line.split('lm loss: ')[1].split('E')[1])
                iterations.append(iteration)
                losses.append(loss)

    if not losses or not iterations:
        print("未能从日志文件中提取到任何数据，请检查日志格式和正则表达式是否匹配。")
        return
    
    # 绘制loss曲线
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, losses, label='Training Loss')
    plt.title('Training Loss Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 保存图像到文件
    plt.savefig(output_file)
    print(f"Loss curve has been saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot loss curve from training log and save it as an image.")
    parser.add_argument('file', type=str, help='Path to the training log file.')
    parser.add_argument('output', type=str, help='Path to save the output image file.')
    args = parser.parse_args()

    plot_loss_curve(args.file, args.output)
