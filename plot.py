# 标注数字版
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 文件路径
files = {
    "Dual-TCN": "excels/results_Dual_TCN.json",
    "RPSR-DQN-Dual-TCN": "excels/results_rpsr_dqn_Dual_TCN.json",
    "RPSR-DQN-TCN": "excels/results_rpsr_dqn_TCN.json",
    "TCN": "excels/results_TCN.json",
    "TDNN": "excels/results_TDNN.json",
    "DTSAT-DRQN": "excels/results_TDRQN_Dual_TCN.json",
    "DRQN-TCN": "excels/results_TDRQN_TCN.json"
}

# 信噪比列表
snr_list = [0, 5, 10, 15, 20]
interference_types = ["rayleigh", "rician", "nakagami"]

# 提取数据
def extract_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# 绘制误码率（BER）图表
def plot_ber(interference_type):
    snr_data = {snr: {} for snr in snr_list}
    for model, file_name in files.items():
        data = extract_data(file_name)
        for snr in snr_list:
            snr_key = str(snr)
            if interference_type in data and snr_key in data[interference_type]:
                ber = data[interference_type][snr_key]["BER"]
                snr_data[snr][model] = ber

    snr_ticks = list(snr_data.keys())
    models = list(snr_data[snr_list[0]].keys())

    # 绘制折线图
    plt.figure(figsize=(12, 8))
    for model in models:
        ber_values = [snr_data[snr][model] for snr in snr_ticks]
        plt.plot(snr_ticks, ber_values, label=model, marker='o', linewidth=2)
        # 标注数值
        # for i, ber in enumerate(ber_values):
        #     plt.text(snr_ticks[i], ber, f'{ber:.4f}', ha='center', va='bottom', fontsize=9)
    plt.xlabel('SNR (dB)', fontsize=14)
    plt.ylabel('BER', fontsize=14)
    # plt.title(f'{interference_type.capitalize()} Channel - BER vs SNR', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'BER_{interference_type}_annotated.eps', dpi=300)
    plt.show()

# 绘制训练时间图表
def plot_train_time(interference_type):
    snr_data = {snr: {} for snr in snr_list}
    for model, file_name in files.items():
        data = extract_data(file_name)
        for snr in snr_list:
            snr_key = str(snr)
            if interference_type in data and snr_key in data[interference_type]:
                if model in ["rpsr_dqn_Dual_TCN", "rpsr_dqn_TCN"]:
                    train_time = data[interference_type][snr_key]["total_time"]
                else:
                    train_time = data[interference_type][snr_key]["train_time"]
                snr_data[snr][model] = train_time

    snr_ticks = list(snr_data.keys())
    models = list(snr_data[snr_list[0]].keys())

    # 绘制柱状图
    plt.figure(figsize=(12, 8))
    bar_width = 0.15
    x = np.arange(len(snr_ticks))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    for i, model in enumerate(models):
        train_times = [snr_data[snr][model] for snr in snr_ticks]
        plt.bar(x + i * bar_width, train_times, bar_width, label=model, color=colors[i % len(colors)])
        # 标注数值
        # for j, time in enumerate(train_times):
        #     plt.text(x[j] + i * bar_width, time, f'{time:.2f}', ha='center', va='bottom', fontsize=9)
    plt.xlabel('SNR (dB)', fontsize=14)
    plt.ylabel('Training Time (seconds)', fontsize=14)
    # plt.title(f'{interference_type.capitalize()} Channel - Training Time', fontsize=16)
    plt.xticks(x + bar_width * (len(models) - 1) / 2, snr_ticks)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'Training_Time_{interference_type}_annotated.eps', dpi=300)
    plt.show()

# 主函数
def main():
    for interference in interference_types:
        # 绘制误码率图表
        plot_ber(interference)
        # 绘制训练时间图表
        plot_train_time(interference)

if __name__ == "__main__":
    main()