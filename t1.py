# 编辑时间：202412291714
# 说明: 基本复现

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch.nn.functional as F
import torch.fft
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


# 强制使用 CPU
device = torch.device("cpu")


# 定义 TDNN 层
class TDNN(nn.Module):
    def __init__(self):
        super(TDNN, self).__init__()
        self.tdnn = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=2, dilation=1),
            nn.ReLU(),
            nn.Conv1d(4, 8, kernel_size=4, dilation=2),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=8, dilation=4),
            nn.ReLU()
        )

    def forward(self, x):
        return self.tdnn(x)


# 双向 LSTM 网络
class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                         batch_first=True, bidirectional=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class FullyConnected(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FullyConnected, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class BatchNormalization(nn.Module):
    def __init__(self, num_features):
        super(BatchNormalization, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        return self.bn(x)


# LSTMDemodulator 解调
class LSTMDemodulator(nn.Module):
    def __init__(self):
        super(LSTMDemodulator, self).__init__()
        self.tdnn = TDNN()
        self.bilstm = BiLSTM(16, 128, 2)
        self.fc1 = FullyConnected(256, 128)
        self.bn = BatchNormalization(128)
        self.fc2 = FullyConnected(128, 2)
        self.dropout = nn.Dropout(0.2)  # 添加 Dropout 层，防止过拟合

    def forward(self, x):
        x = self.tdnn(x)
        x = x.transpose(1, 2)
        x = self.bilstm(x)
        x = x[:, -1, :]  # 只取最后一个时间步的状态
        x = F.relu(self.fc1(x))
        x = self.bn(x)
        x = self.dropout(x)  # 应用 Dropout
        x = self.fc2(x)
        return x


def train_model(model, train_loader, num_epochs, learning_rate, device):
    model.to(device)  # 确保模型位于正确的设备上
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)  # 学习率调度器
    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1} Training')):
            inputs, labels = inputs.unsqueeze(1).to(device), labels.to(device).long()  # 添加通道维度并转换标签类型

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_acc)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        scheduler.step()  # 更新学习率

    return model, epoch_losses, epoch_accuracies


def test_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs, labels = inputs.unsqueeze(1).to(device), labels.to(device).long()

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_predictions), np.array(all_labels)

def calculate_ber(original_data, received_data):
    if len(original_data)!= len(received_data):
        raise ValueError("Original data and received data must have the same length.")
    # 将 BPSK 信号从 [-1, 1] 映射回 [0, 1]
    original_bits = ((np.array(original_data) + 1) / 2).astype(int)
    received_bits = ((np.array(received_data) + 1) / 2).astype(int)
    # 计算误比特数
    error_bits = np.sum(original_bits!= received_bits)
    # 计算总比特数
    total_bits = len(original_bits)
    # 计算 BER
    ber = error_bits / total_bits
    return ber

def calculate_precision_recall(y_true, y_pred):
    # 计算精确率和召回率
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    return precision, recall


def calculate_f1_score(precision, recall):
    # 计算 F1 分数
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def add_awgn_noise(signal, snr_db):
    # 添加 AWGN 噪声
    snr_linear = 10 ** (snr_db / 10)
    signal_power = torch.mean(torch.abs(signal) ** 2)
    noise_power = signal_power / snr_linear
    noise = torch.sqrt(noise_power) * (torch.randn(signal.size()) + 1j * torch.randn(signal.size()))
    return signal + noise


def add_rayleigh_noise(signal, snr_db):
    # 添加瑞利噪声
    snr_linear = 10 ** (snr_db / 10)
    signal_power = torch.mean(torch.abs(signal) ** 2)
    noise_power = signal_power / snr_linear
    h = torch.sqrt(torch.randn(signal.size()) ** 2 + torch.randn(signal.size()) ** 2) / np.sqrt(2)
    noise = torch.sqrt(noise_power) * (torch.randn(signal.size()) + 1j * torch.randn(signal.size())) / h
    return signal * h + noise


def mmse_equalizer(received_signal, channel_estimate):
    # MMSE 均衡器
    return received_signal / (channel_estimate + 1e-10)

def zf_equalizer(received_signal, channel_estimate):
    # ZF 均衡器
    # 计算信道估计的逆
    channel_inverse = 1 / channel_estimate  # 这里假设 channel_estimate 是一个标量或者每个元素都不为零的向量
    return received_signal * channel_inverse  # 将接收信号乘以信道估计的逆

def simulate_attack(signal, attack_type='signal_interference'):
    if attack_type == 'signal_interference':
        # 模拟信号干扰，添加随机噪声
        interference = torch.randn(signal.size())
        return signal + interference
    elif attack_type == 'data_tampering':
        # 模拟数据篡改，随机修改部分数据
        indices = torch.randint(0, signal.numel(), (signal.numel() // 10,))
        signal.view(-1)[indices] = torch.randn(indices.size(), dtype=signal.dtype)  # 生成相同类型的数据
        return signal
    else:
        raise ValueError("Invalid attack type")


def evaluate_model_performance(model, dataset, device, snr_db):
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    predictions, true_labels = test_model(model, test_loader, device)
    ber = calculate_ber(dataset.tensors[1].numpy(), predictions)
    precision, recall = calculate_precision_recall(true_labels, predictions)
    f1_score = calculate_f1_score(precision, recall)
    print(f"SNR: {snr_db} dB")
    print(f'Bit Error Rate (BER): {ber:.6f}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}')
    print(f'F1 Score: {f1_score:.4f}')
    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, predictions))
    print("Classification Report:")
    print(classification_report(true_labels, predictions, zero_division=1))  # 添加 zero_division 参数


if __name__ == "__main__":
    # 参数设置
    N = 64  # OFDM 子载波数量 64
    K = 128  # 混沌序列长度 128
    num_symbols = 10000  # 数据符号数量
    # SNR_dbs = [0, 5, 10, 15, 20]  # 不同信噪比 (dB)
    SNR_dbs = [10]  # 不同信噪比 (dB)

    # 生成二阶切比雪夫多项式函数用于生成混沌序列
    def chebyshev_map(x):
        return 1 - 2 * x ** 2

    # 生成 BPSK 调制的用户数据
    user_data = np.random.randint(0, 2, num_symbols) * 2 - 1  # [-1, 1]

    # 生成混沌序列 x
    x = np.zeros(K)
    x[0] = np.random.uniform(-1, 1)  # 初始值
    for k in np.arange(1, K):
        x[k] = chebyshev_map(x[k - 1])

    # 调制过程：将用户数据与混沌序列相乘
    modulated_symbols = np.zeros((num_symbols, K))
    for n in np.arange(num_symbols):
        modulated_symbols[n, :] = user_data[n] * x

    # OFDM 调制
    ofdm_symbols = np.zeros((num_symbols, N), dtype=complex)
    for k in range(K):
        ofdm_symbols[:, k % N] += modulated_symbols[:, k] / np.sqrt(N)

    # 添加循环前缀 (CP)
    cp_length = N // 4
    ofdm_symbols_with_cp = np.concatenate([ofdm_symbols[:, -cp_length:], ofdm_symbols], axis=1)

    # 存储不同 SNR 下的性能指标
    all_bers = []
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    all_losses = []
    all_accuracies = []

    for SNR_db in SNR_dbs:
        # 信道传输：添加 AWGN 噪声
        received_ofdm_symbols = add_awgn_noise(torch.tensor(ofdm_symbols_with_cp), SNR_db)

        # 接收端处理：去除 CP 并进行 FFT
        received_ofdm_symbols_no_cp = received_ofdm_symbols[:, cp_length:]
        fft_received_symbols = torch.fft.fft(received_ofdm_symbols_no_cp, dim=1)

        # 均衡处理
        channel_estimate = torch.ones_like(fft_received_symbols)  # 假设简单的信道估计
        equalized_symbols = mmse_equalizer(fft_received_symbols, channel_estimate)
        equalized_symbols = equalized_symbols.real

        # 构建训练数据 yn 和对应的标签
        yn = equalized_symbols.reshape(num_symbols, -1)  # 只取实部作为输入特征

        # 将用户比特从 [-1, 1] 映射到 [0, 1]
        labels = ((user_data + 1) / 2).astype(int)  # 将 -1 转换为 0, 1 保持不变

        # 创建数据集和加载器
        dataset = TensorDataset(equalized_symbols.float(), torch.tensor(labels, dtype=torch.long))
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # 创建模型实例
        model = LSTMDemodulator()

        # 训练轮数和学习率设置
        num_epochs = 3
        learning_rate = 0.001

        # 训练模型
        trained_model, epoch_losses, epoch_accuracies = train_model(
            model,
            train_loader,
            num_epochs,
            learning_rate,
            device
        )
        all_losses.append(epoch_losses)
        all_accuracies.append(epoch_accuracies)

        # 测试模型性能
        evaluate_model_performance(trained_model, dataset, device, SNR_db)

        predictions, true_labels = test_model(trained_model, train_loader, device)
        precision, recall = calculate_precision_recall(true_labels, predictions)
        f1_score = calculate_f1_score(precision, recall)
        ber = calculate_ber(user_data, predictions)

        all_bers.append(ber)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1_scores.append(f1_score)

    print(all_bers)

    # # 性能评估：绘制损失曲线和性能指标曲线
    # plt.figure(figsize=(12, 8))
    # for i, SNR_db in enumerate(SNR_dbs):
    #     plt.subplot(2, 2, 1)
    #     plt.plot(all_losses[i], label=f'SNR={SNR_db} dB')
    #     plt.title('Training Loss')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #
    #     plt.subplot(2, 2, 2)
    #     plt.plot(all_accuracies[i], label=f'SNR={SNR_db} dB')
    #     plt.title('Training Accuracy')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Accuracy (%)')
    #     plt.legend()
    #
    #     plt.subplot(2, 2, 3)
    #     plt.plot(all_bers, label='BER')
    #     plt.title('Bit Error Rate (BER)')
    #     plt.xlabel('SNR (dB)')
    #     plt.ylabel('BER')
    #     plt.legend()
    #
    #     plt.subplot(2, 2, 4)
    #     plt.plot(all_f1_scores, label='F1 Score')
    #     plt.title('F1 Score')
    #     plt.xlabel('SNR (dB)')
    #     plt.ylabel('F1 Score')
    #     plt.legend()
    #
    # plt.tight_layout()
    # plt.show()
    #
    #
    # # 安全性评估：模拟攻击
    # attack_types = ['signal_interference', 'data_tampering']
    # for attack_type in attack_types:
    #     print(f"Simulating {attack_type} attack...")
    #     attacked_signal = simulate_attack(received_ofdm_symbols, attack_type)
    #     # 接收端处理：去除 CP 并进行 FFT
    #     attacked_signal_no_cp = attacked_signal[:, cp_length:]
    #     fft_attacked_symbols = torch.fft.fft(attacked_signal_no_cp, dim=1)
    #
    #     # 均衡处理
    #     attacked_equalized_symbols = mmse_equalizer(fft_attacked_symbols, channel_estimate)
    #     attacked_equalized_symbols = attacked_equalized_symbols.real
    #
    #     # 构建训练数据 yn 和对应的标签
    #     attacked_yn = attacked_equalized_symbols.reshape(num_symbols, -1)  # 只取实部作为输入特征
    #
    #     # 创建数据集和加载器
    #     attacked_dataset = TensorDataset(attacked_equalized_symbols.float(), torch.tensor(labels, dtype=torch.long))
    #     attacked_loader = DataLoader(attacked_dataset, batch_size=32, shuffle=True)
    #
    #     # 测试模型性能
    #     evaluate_model_performance(trained_model, attacked_dataset, device, SNR_db)
    #
    #
    # # 鲁棒性评估：不同噪声环境
    # noise_types = ['awgn', 'rayleigh']
    # for noise_type in noise_types:
    #     print(f"Testing in {noise_type} noise environment...")
    #     if noise_type == 'awgn':
    #         received_signal = add_awgn_noise(torch.tensor(ofdm_symbols_with_cp), SNR_db)
    #     elif noise_type == 'rayleigh':
    #         received_signal = add_rayleigh_noise(torch.tensor(ofdm_symbols_with_cp), SNR_db)
    #
    #     # 接收端处理：去除 CP 并进行 FFT
    #     received_signal_no_cp = received_signal[:, cp_length:]
    #     fft_received_signal = torch.fft.fft(received_signal_no_cp, dim=1)
    #
    #     # 均衡处理
    #     equalized_signal = mmse_equalizer(fft_received_signal, channel_estimate)
    #     equalized_signal = equalized_signal.real
    #
    #     # 构建训练数据 yn 和对应的标签
    #     yn = equalized_signal.reshape(num_symbols, -1)  # 只取实部作为输入特征
    #
    #     # 创建数据集和加载器
    #     dataset = TensorDataset(equalized_signal.float(), torch.tensor(labels, dtype=torch.long))
    #     test_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    #
    #     # 测试模型性能
    #     evaluate_model_performance(trained_model, dataset, device, SNR_db)