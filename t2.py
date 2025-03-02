import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import torch.nn.functional as F
import torch.fft
import numpy as np
import os
from torch.nn.utils import weight_norm
import time
from torch.distributions import Gamma

# 强制使用 CPU
device = torch.device("cuda")
# # 固定随机种子
# seed = 20  # 可以选择任意整数作为种子
# np.random.seed(seed)
# torch.manual_seed(seed)


# 定义基础的TCN模块
class BaseTCN(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size=2, dilation_base=2):
        super(BaseTCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = dilation_base ** i
            in_channels = input_channels if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                      dilation=dilation_size, padding=(kernel_size - 1) * dilation_size // 2)),
                nn.ReLU()
            ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# 定义局部TCN（Local TCN），用于捕捉局部时间模式
class LocalTCN(BaseTCN):
    def __init__(self, input_channels, num_channels):
        super(LocalTCN, self).__init__(input_channels, num_channels, kernel_size=3, dilation_base=2)


# 定义全局TCN（Global TCN），用于捕捉全局时间模式
class GlobalTCN(BaseTCN):
    def __init__(self, input_channels, num_channels):
        super(GlobalTCN, self).__init__(input_channels, num_channels, kernel_size=5, dilation_base=3)


# 定义注意力机制模块，用于融合局部和全局TCN的特征
class TCNAttention(nn.Module):
    def __init__(self, local_feature_dim, global_feature_dim):
        super(TCNAttention, self).__init__()
        combined_dim = local_feature_dim + global_feature_dim
        self.query = nn.Linear(combined_dim, combined_dim // 2)
        self.key = nn.Linear(combined_dim, combined_dim // 2)
        self.value = nn.Linear(combined_dim, combined_dim // 2)  # 保持输出维度和query、key一致，方便后续计算
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, local_feature, global_feature):
        # 确认local_feature和global_feature维度符合预期，确保都是三维张量且批次大小、序列长度维度一致
        assert local_feature.dim() == 3 and global_feature.dim() == 3, "Local and Global features should be 3D tensors"
        assert local_feature.size(0) == global_feature.size(
            0), "Batch sizes of Local and Global features should be the same"
        assert local_feature.size(2) == global_feature.size(
            2), "Sequence lengths of Local and Global features should be the same"

        combined_feature = torch.cat([local_feature, global_feature], dim=1)
        # 调整维度顺序，将通道维度转置到最后一维，以符合nn.Linear对输入维度的默认处理方式（按最后一维做线性变换）
        combined_feature = combined_feature.transpose(1, 2).contiguous()

        query = self.query(combined_feature).transpose(1, 2).contiguous()
        key = self.key(combined_feature).transpose(1, 2).contiguous()
        value = self.value(combined_feature).transpose(1, 2).contiguous()

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        attention_weights = self.softmax(attention_scores)

        attended_feature = torch.matmul(attention_weights, value)
        # 先展平为一维张量
        flattened_feature = attended_feature.view(-1)
        total_elements = flattened_feature.numel()
        # 根据期望维度和元素总数计算各维度大小，这里假设batch_size为attended_feature.size(0)
        batch_size = attended_feature.size(0)
        new_sequence_length = total_elements // (batch_size * 16)
        # 重塑为期望的维度
        attended_feature = flattened_feature.view(batch_size, new_sequence_length, 16)
        attended_feature = attended_feature.transpose(1, 2).contiguous()
        return attended_feature


# 双向LSTM网络
class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


# 全连接层
class FullyConnected(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FullyConnected, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


# 批量归一化层
class BatchNormalization(nn.Module):
    def __init__(self, num_features):
        super(BatchNormalization, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        return self.bn(x)


# LSTMDemodulator解调，融入Dual TCN和注意力机制
class LSTMDemodulator(nn.Module):
    def __init__(self):
        super(LSTMDemodulator, self).__init__()
        self.local_tcn = LocalTCN(input_channels=1, num_channels=[4, 8])
        self.global_tcn = GlobalTCN(input_channels=1, num_channels=[4, 8])
        self.attention = TCNAttention(local_feature_dim=8, global_feature_dim=8)
        self.bilstm = BiLSTM(16, 128, 2)
        self.fc1 = FullyConnected(256, 128)
        self.bn = BatchNormalization(128)
        self.fc2 = FullyConnected(128, 2)

    def forward(self, x):
        local_feature = self.local_tcn(x)
        global_feature = self.global_tcn(x)
        attended_feature = self.attention(local_feature, global_feature)
        attended_feature = attended_feature.transpose(1, 2)
        x = self.bilstm(attended_feature)
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.bn(x)
        x = self.fc2(x)
        return x


def generate_dataset(N, K, num_symbols, snr_db, channel_model='awgn'):
    # 生成二阶切比雪夫多项式函数用于生成混沌序列
    def chebyshev_map(x):
        return 1 - 2 * x ** 2

    # 生成 BPSK 调制的用户数据
    user_data = np.random.randint(0, 2, num_symbols) * 2 - 1  # [-1, 1]

    # 生成混沌序列 x
    x = np.zeros(K)
    x[0] = np.random.uniform(-1, 1)  # 初始值
    for k in range(1, K):
        x[k] = chebyshev_map(x[k - 1])

    # 调制过程：将用户数据与混沌序列相乘
    modulated_symbols = user_data[:, np.newaxis] * x[np.newaxis, :]

    # OFDM 调制
    ofdm_symbols = np.zeros((num_symbols, N), dtype=complex)
    for k in range(K):
        ofdm_symbols[:, k % N] += modulated_symbols[:, k] / np.sqrt(N)

    # 添加循环前缀 (CP)
    cp_length = N // 4
    ofdm_symbols_with_cp = np.concatenate([ofdm_symbols[:, -cp_length:], ofdm_symbols], axis=1)

    # 信道传输：添加不同类型的噪声
    transmitted_signal = torch.tensor(ofdm_symbols_with_cp, dtype=torch.complex64)
    if channel_model == 'awgn':
        received_signal = add_awgn_noise(transmitted_signal.clone().detach(), snr_db)
    elif channel_model == 'rayleigh':
        received_signal = add_rayleigh_noise(transmitted_signal.clone().detach(), snr_db)
    elif channel_model == 'rician':
        received_signal = add_high_speed_railway_fading(transmitted_signal.clone().detach(), snr_db)
    elif channel_model == 'nakagami':
        received_signal = add_nakagami_fading(transmitted_signal.clone().detach(), snr_db)
    else:
        raise ValueError("Invalid channel_model")

    # 接收端处理：去除 CP 并进行 FFT
    received_ofdm_symbols_no_cp = received_signal[:, cp_length:]
    fft_received_symbols = torch.fft.fft(received_ofdm_symbols_no_cp, dim=1)

    # 更复杂的信道估计
    channel_estimate = generate_channel_estimate(fft_received_symbols)
    equalized_symbols = mmse_equalizer(fft_received_symbols, channel_estimate)
    equalized_symbols = equalized_symbols.real

    # 构建训练数据 yn 和对应的标签
    yn = equalized_symbols.reshape(num_symbols, -1)  # 只取实部作为输入特征

    # 将用户比特从 [-1, 1] 映射到 [0, 1]
    labels = ((user_data + 1) // 2).astype(int)

    # 存储原始比特信息
    original_bits = user_data.astype(int)

    # 打印生成数据集的形状信息
    print(f"Equalized symbols shape: {yn.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Original bits shape: {original_bits.shape}")

    dataset = TensorDataset(
        torch.from_numpy(yn.detach().numpy().astype(np.float32)),  # 将 PyTorch 张量转换为 NumPy 数组
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(original_bits, dtype=torch.long)
    )

    return dataset


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


def add_high_speed_railway_fading(signal, snr_db, K_rician=2.83, max_doppler=100):
    """改进的高速铁路信道模型（时变Rician衰落）"""
    # 转换莱斯因子为线性值
    K_linear = 10 ** (K_rician / 10)

    # 生成时变参数
    num_samples = signal.shape[0]
    t = torch.linspace(0, 1, num_samples)

    # 生成LOS分量（考虑多普勒频移）
    f_d = max_doppler * torch.rand(1)  # 随机多普勒频移
    los_component = torch.sqrt(torch.tensor(K_linear / (K_linear + 1)) * torch.exp(1j * 2 * np.pi * f_d * t))

    # 生成NLOS分量（瑞利衰落）
    nlos_real = torch.randn(num_samples)
    nlos_imag = torch.randn(num_samples)
    nlos_component = torch.sqrt(torch.tensor(1 / (K_linear + 1)) * (nlos_real + 1j * nlos_imag) / np.sqrt(2))

    # 合成信道系数
    h = los_component + nlos_component

    # 添加噪声
    snr_linear = 10 ** (snr_db / 10)
    signal_power = torch.mean(torch.abs(signal) ** 2)
    noise_power = signal_power / snr_linear
    noise = torch.sqrt(noise_power / 2) * (torch.randn_like(signal) + 1j * torch.randn_like(signal))

    return signal * h.unsqueeze(1) + noise


def generate_nakagami_fading(num_samples, m=2, Omega=1.0):
    """生成 Nakagami-m 衰落系数"""
    shape = torch.tensor([m], dtype=torch.float32)
    scale = torch.tensor([Omega / (2 * m)], dtype=torch.float32)
    gamma_dist = Gamma(shape, scale)
    gamma_samples = gamma_dist.sample((num_samples,)).squeeze()

    # Nakagami-m 的随机变量是 Gamma 分布的平方根
    real_part = torch.sqrt(gamma_samples)
    imag_part = torch.sqrt(torch.rand_like(gamma_samples))  # 修正生成复数部分
    h = real_part + 1j * imag_part

    return h.unsqueeze(1) / torch.sqrt(torch.mean(torch.abs(h.unsqueeze(1)) ** 2))  # 归一化功率并扩展维度


def add_nakagami_fading(signal, snr_db, m=1.5, Omega=1.0):
    """将 Nakagami-m 衰落应用到信号"""
    h = generate_nakagami_fading(signal.shape[0], m, Omega)
    signal_faded = signal * h.expand_as(signal)  # 扩展 h 的维度以匹配 signal

    # 添加 AWGN 噪声
    snr_linear = 10 ** (snr_db / 10)
    signal_power = torch.mean(torch.abs(signal_faded) ** 2)
    noise_power = signal_power / snr_linear
    noise = torch.sqrt(noise_power / 2) * (torch.randn_like(signal_faded) + 1j * torch.randn_like(signal_faded))

    return signal_faded + noise


def mmse_equalizer(received_signal, channel_estimate):
    # MMSE 均衡器
    return received_signal / (channel_estimate + 1e-10)


def zf_equalizer(received_signal, channel_estimate):
    # ZF 均衡器
    # 计算信道估计的逆
    channel_inverse = 1 / channel_estimate  # 这里假设 channel_estimate 是一个标量或者每个元素都不为零的向量
    return received_signal * channel_inverse  # 将接收信号乘以信道估计的逆


def generate_channel_estimate(received_signal):
    # 简单的最小二乘信道估计，实际应用中可使用更复杂的信道估计方法
    return torch.mean(received_signal, dim=0)


def train_model(model, train_loader, num_epochs, learning_rate, device, scheduler_type='plateau'):
    model.to(device)  # 确保模型位于正确的设备上
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    if scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)  # 调整学习率调度器
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    epoch_losses = []
    epoch_accuracies = []

    # 开始时间
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1} Training')):
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
        if scheduler_type == 'plateau':
            scheduler.step(epoch_loss)  # 更新学习率
        elif scheduler_type == 'cosine':
            scheduler.step()

        # 打印每轮训练的损失和准确率
        print(f"Epoch {epoch + 1}: Loss = {epoch_loss}, Accuracy = {epoch_acc}%")

    # 结束时间
    end_time = time.time()
    print(f'Total Training Time: {end_time - start_time:.2f}s')

    return model, epoch_losses, epoch_accuracies


def evaluate_model(model, test_loader, device):
    model.eval()
    total_error_bits = 0
    total_bits = 0
    correct = 0
    precision = 0
    recall = 0
    f1_score = 0

    # 开始时间
    start_time = time.time()

    with torch.no_grad():
        for i, (inputs, labels, original_bits) in enumerate(tqdm(test_loader, desc='Testing')):
            inputs, labels = inputs.unsqueeze(1).to(device), labels.to(device).long()
            original_bits = original_bits.to(device).long()

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # # 打印测试集的输入和输出形状信息
            # print(f"Test Batch {i}: Input shape = {inputs.shape}, Output shape = {outputs.shape}, Predicted shape = {predicted.shape}")

            # 计算误码率相关
            received_bits = predicted.cpu().numpy()
            original_bits = original_bits.cpu().numpy()
            # 将 original_bits 的取值范围从 [-1, 1] 转换为 [0, 1]
            original_bits = ((original_bits + 1) / 2).astype(int)
            # print(f"Original bits: {original_bits[:10]}")
            # print(f"Received bits: {received_bits[:10]}")
            # print(f"Original bits shape: {original_bits.shape}, Received bits shape: {received_bits.shape}")
            # 确保形状和数据类型一致
            if original_bits.shape != received_bits.shape:
                min_length = min(original_bits.shape[0], received_bits.shape[0])
                original_bits = original_bits[:min_length]
                received_bits = received_bits[:min_length]
            error_bits = np.sum(original_bits != received_bits)
            total_error_bits += error_bits
            total_bits += len(original_bits)

            # 计算精度和召回率相关
            correct += (predicted == labels).sum().item()

        if len(test_loader.dataset) > 0:
            precision = correct / len(test_loader.dataset)
            recall = correct / len(test_loader.dataset)
            if (precision + recall) > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            print("Warning: Test dataset is empty.")

    # 结束时间
    end_time = time.time()
    print(f'Total Training Time: {end_time - start_time:.2f}s')

    ber = total_error_bits / total_bits
    print(f"Bit Error Rate (BER): {ber:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
    return ber, precision, recall, f1_score


if __name__ == "__main__":
    # 参数设置
    N = 64  # OFDM 子载波数量
    K = 128  # 混沌序列长度
    num_symbols = 10000  # 固定数据符号数量
    # SNR_dbs = [0, 5, 10, 15, 20]  # 不同信噪比 (dB)
    SNR_dbs = [10]  # 不同信噪比 (dB)
    # channel_models = ['rayleigh', 'rician', 'nakagami']  # 信道模型列表
    channel_models = ['rician']  # 信道模型列表

    # 存储结果
    results = {}

    for channel_model in channel_models:
        results[channel_model] = {}
        for snr_db in SNR_dbs:
            start_time = time.time()  # 记录开始时间

            # 生成数据集
            dataset = generate_dataset(N, K, num_symbols, snr_db, channel_model=channel_model)

            # 划分训练集和测试集
            train_size = int(0.8 * len(dataset))  # 80% 作为训练集
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

            # 创建训练集和测试集的加载器
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # 创建模型实例
            model = LSTMDemodulator()

            # 训练轮数和学习率设置
            num_epochs = 3
            learning_rate = 0.001

            # 训练模型
            start_train_time = time.time()
            trained_model, _, _ = train_model(
                model,
                train_loader,
                num_epochs,
                learning_rate,
                device,
                scheduler_type='cosine'
            )
            end_train_time = time.time()
            total_train_time = end_train_time - start_train_time

            # 测试模型性能
            start_test_time = time.time()
            ber, precision, recall, f1_score = evaluate_model(trained_model, test_loader, device)
            end_test_time = time.time()
            total_test_time = end_test_time - start_test_time

            # 存储结果
            results[channel_model][snr_db] = {
                'train_time': total_train_time,
                'test_time': total_test_time,
                'BER': float(ber),
                'Precision': float(precision),
                'Recall': float(recall),
                'F1 Score': float(f1_score)
            }

            end_time = time.time()
            print(
                f"{channel_model} - SNR: {snr_db}dB - Num Symbols: {num_symbols} - Total Time: {end_time - start_time:.2f}s")

    # 将结果保存为 JSON 文件
    with open('results_Dual_TCN.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(results)
