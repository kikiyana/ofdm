# 效果最好
# 创新点：使用双向TCN网络；引入注意力机制，以增强模型的泛化能力，并提高模型的性能。引入强化学习，构建监督学习和强化学习混合模型
#

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from torch.nn.utils import weight_norm
import copy
import torch.nn.functional as F
import torch.optim as optim
import time

# 强制使用 CPU
device = torch.device("cuda")


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


# Transformer层
class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.transformer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.positional_encoding = PositionalEncoding(d_model, max_len=1000)

    def forward(self, x):
        x = self.positional_encoding(x)
        x = self.transformer(x)
        return x


# 位置编码层
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# 双向LSTM网络
class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        return x, hidden


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


# LSTMDemodulator解调，融入Dual TCN、注意力机制和Transformer
class LSTMDemodulator(nn.Module):
    def __init__(self):
        super(LSTMDemodulator, self).__init__()
        self.local_tcn = LocalTCN(input_channels=1, num_channels=[4, 8])
        self.global_tcn = GlobalTCN(input_channels=1, num_channels=[4, 8])
        self.attention = TCNAttention(local_feature_dim=8, global_feature_dim=8)
        self.transformer = TransformerLayer(d_model=16, nhead=2)  # 添加Transformer层
        self.bilstm = BiLSTM(16, 128, 2)
        self.fc1 = FullyConnected(256, 128)
        self.bn = BatchNormalization(128)
        self.fc2 = FullyConnected(128, 2)

    def forward(self, x, hidden=None):
        local_feature = self.local_tcn(x)
        global_feature = self.global_tcn(x)
        attended_feature = self.attention(local_feature, global_feature)
        attended_feature = attended_feature.transpose(1, 2)
        attended_feature = self.transformer(attended_feature)  # 通过Transformer处理
        x, hidden = self.bilstm(attended_feature)
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.bn(x)
        x = self.fc2(x)
        return x, hidden


def generate_dataset(N, K, num_symbols, snr_db, channel_model='awgn'):
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

    # 信道传输：添加不同类型的噪声
    if channel_model == 'awgn':
        received_ofdm_symbols = add_awgn_noise(torch.tensor(ofdm_symbols_with_cp), snr_db)
    elif channel_model == 'rayleigh':
        received_ofdm_symbols = add_rayleigh_noise(torch.tensor(ofdm_symbols_with_cp), snr_db)
    elif channel_model == 'multi_path_fast_rayleigh':
        received_ofdm_symbols = add_multi_path_fast_rayleigh_fading(torch.tensor(ofdm_symbols_with_cp), snr_db, 3, 5)
    elif channel_model == 'rician':
        received_ofdm_symbols = add_rician_noise(torch.tensor(ofdm_symbols_with_cp), snr_db)
    else:
        raise ValueError("Invalid channel model")

    # 接收端处理：去除 CP 并进行 FFT
    received_ofdm_symbols_no_cp = received_ofdm_symbols[:, cp_length:]
    fft_received_symbols = torch.fft.fft(received_ofdm_symbols_no_cp, dim=1)

    # 更复杂的信道估计
    channel_estimate = generate_channel_estimate(fft_received_symbols)
    equalized_symbols = mmse_equalizer(fft_received_symbols, channel_estimate)
    equalized_symbols = equalized_symbols.real

    # 构建训练数据 yn 和对应的标签
    yn = equalized_symbols.reshape(num_symbols, -1)  # 只取实部作为输入特征

    # 将用户比特从 [-1, 1] 映射到 [0, 1]
    labels = ((user_data + 1) / 2).astype(int)  # 将 -1 转换为 0, 1 保持不变

    # 存储原始比特信息
    original_bits = user_data.astype(int)

    # 打印生成数据集的形状信息
    print(f"Equalized symbols shape: {equalized_symbols.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Original bits shape: {original_bits.shape}")

    dataset = TensorDataset(equalized_symbols.float(), torch.tensor(labels, dtype=torch.long), torch.tensor(original_bits, dtype=torch.long))
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

def add_multi_path_fast_rayleigh_fading(signal, snr_db, num_paths=3, max_delay=5):
    """
    添加多径快瑞利衰落和噪声到信号中
    :param signal: 输入信号
    :param snr_db: 信噪比（dB）
    :param num_paths: 多径数量
    :param max_delay: 最大时延
    :return: 经过多径快瑞利衰落和噪声处理后的信号
    """
    # 计算信噪比的线性值
    snr_linear = 10 ** (snr_db / 10)
    signal_power = torch.mean(torch.abs(signal) ** 2)
    noise_power = signal_power / snr_linear

    # 生成每个路径的时延和增益
    delays = np.random.randint(0, max_delay, num_paths)
    gains = torch.sqrt(torch.randn(num_paths) ** 2 + torch.randn(num_paths) ** 2) / np.sqrt(2)

    # 初始化多径信号
    multi_path_signal = torch.zeros_like(signal, dtype=torch.complex64)

    # 处理每个路径的信号
    for i in range(num_paths):
        delay = delays[i]
        gain = gains[i]

        # 生成该路径的瑞利衰落系数
        h = torch.sqrt(torch.randn(signal.size()) ** 2 + torch.randn(signal.size()) ** 2) / np.sqrt(2)

        # 对信号进行时延处理
        delayed_signal = torch.roll(signal, delay, dims=0)

        # 应用衰落系数和增益
        path_signal = gain * h * delayed_signal

        # 叠加到多径信号上
        multi_path_signal += path_signal

    # 添加高斯噪声
    noise = torch.sqrt(noise_power) * (torch.randn(signal.size()) + 1j * torch.randn(signal.size()))
    received_signal = multi_path_signal + noise

    return received_signal

def add_rician_noise(signal, snr_db):
    # 添加莱斯噪声
    K_factor = 1  # 莱斯因子，可调整
    snr_linear = 10 ** (snr_db / 10)
    signal_power = torch.mean(torch.abs(signal) ** 2)
    noise_power = signal_power / snr_linear
    h_rayleigh = torch.sqrt(torch.randn(signal.size()) ** 2 + torch.randn(signal.size()) ** 2) / np.sqrt(2)
    k_tensor = torch.tensor(K_factor / (K_factor + 1))
    one_tensor = torch.tensor(1 / (K_factor + 1))
    h_rician = torch.sqrt(k_tensor) + torch.sqrt(one_tensor) * h_rayleigh
    noise = torch.sqrt(noise_power) * (torch.randn(signal.size()) + 1j * torch.randn(signal.size())) / h_rician
    return signal * h_rician + noise

# MMSE 均衡器
def mmse_equalizer(received_signal, channel_estimate):
    return received_signal / (channel_estimate + 1e-10)

# 最小二乘信道估计
def generate_channel_estimate(received_signal):
    return torch.mean(received_signal, dim=0)

# 训练模型
def train_model(model, train_loader, num_epochs, learning_rate, device, scheduler_type='plateau'):
    model.to(device)  # 确保模型位于正确的设备上
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)  # 调整学习率调度器
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    epoch_losses = []
    epoch_accuracies = []

    # 开始时间
    start_time = time.time()

    # DRQN 相关参数
    gamma = 0.99  # 折扣因子
    update_target_freq = 100  # 目标网络更新频率
    target_model = LSTMDemodulator().to(device)  # 初始化目标网络
    target_model.load_state_dict(model.state_dict())  # 同步初始权重

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        hidden = None

        for i, (inputs, labels, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1} Training')):
            inputs, labels = inputs.unsqueeze(1).to(device), labels.to(device).long()  # 添加通道维度并转换标签类型

            optimizer.zero_grad()
            outputs, hidden = model(inputs, hidden)
            if isinstance(hidden, tuple):
                hidden = (hidden[0].detach(), hidden[1].detach())
            else:
                hidden = hidden.detach()

            class_loss = criterion(outputs, labels)  # 分类损失

            # Q-learning 损失，奖励基于预测是否正确
            _, predicted = torch.max(outputs.data, 1)
            rewards = (predicted == labels).float() * 1.0  # 奖励：1表示正确，0表示错误

            # #奖励基于误码率
            # _, predicted = torch.max(outputs.data, 1)
            # error_rate = (predicted != labels).float().mean()  # 计算误码率
            # rewards = 1 - error_rate  # 奖励基于误码率

            selected_q = outputs.gather(1, predicted.unsqueeze(1)).squeeze()

            with torch.no_grad():
                next_outputs, _ = target_model(inputs, hidden)
            max_next_q = next_outputs.detach().max(1)[0]
            target_q = rewards + gamma * max_next_q
            q_loss = F.mse_loss(selected_q, target_q)

            # 总损失
            total_loss = class_loss + 0.1 * q_loss  # 调整权重

            total_loss.backward()
            optimizer.step()

            # 更新目标网络
            if (i + 1) % update_target_freq == 0:
                target_model.load_state_dict(model.state_dict())

            running_loss += class_loss.item()
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


# 测试模型
def evaluate_model(model, test_loader, device):
    model.eval()
    total_error_bits = 0
    total_bits = 0
    correct = 0
    precision = 0
    recall = 0
    f1_score = 0

    predictions = []
    labels_list = []

    # 开始时间
    start_time = time.time()

    with torch.no_grad():
        for i, (inputs, labels, original_bits) in enumerate(tqdm(test_loader, desc='Testing')):
            inputs, labels = inputs.unsqueeze(1).to(device), labels.to(device).long()
            original_bits = original_bits.to(device).long()

            # 重新初始化隐藏状态
            batch_size = inputs.size(0)
            num_layers = 2
            hidden_dim = 128
            hidden = (torch.zeros(2 * num_layers, batch_size, hidden_dim).to(device),
                      torch.zeros(2 * num_layers, batch_size, hidden_dim).to(device))

            outputs, hidden = model(inputs, hidden)

            _, predicted = torch.max(outputs.data, 1)

            # 误差统计
            received_bits = predicted.cpu().numpy()
            original_bits = ((original_bits.cpu().numpy() + 1) / 2).astype(int)
            error_bits = np.sum(original_bits != received_bits)
            total_error_bits += error_bits
            total_bits += len(original_bits)

            # 正确统计
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    # 计算 BER
    ber = total_error_bits / total_bits

    # 计算分类报告
    classification_report_data = classification_report(labels_list, predictions, output_dict=True)
    precision = classification_report_data['macro avg']['precision']
    recall = classification_report_data['macro avg']['recall']
    f1_score = classification_report_data['macro avg']['f1-score']

    # 结束时间
    end_time = time.time()
    print(f'Total Testing Time: {end_time - start_time:.2f}s')


    print(f"Bit Error Rate (BER): {ber:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")

    return ber, precision, recall, f1_score


if __name__ == "__main__":
    # 参数设置
    N = 64  # OFDM 子载波数量 64
    K = 128  # 混沌序列长度 128
    num_symbols = 1000  # 数据符号数量
    # SNR_dbs = [0, 5, 10, 15, 20]  # 不同信噪比 (dB)
    SNR_dbs = [10]  # 不同信噪比 (dB)

    # 存储不同 SNR 下的性能指标
    all_bers = []
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    all_losses = []
    all_accuracies = []

    for SNR_db in SNR_dbs:
        # 生成数据集
        dataset = generate_dataset(N, K, num_symbols, SNR_db, channel_model='rayleigh')

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
        num_epochs = 3  # 增加训练轮数
        learning_rate = 0.0005

        # 训练模型
        trained_model, epoch_losses, epoch_accuracies = train_model(
            model,
            train_loader,
            num_epochs,
            learning_rate,
            device,
            scheduler_type='cosine'
        )
        all_losses.append(epoch_losses)
        all_accuracies.append(epoch_accuracies)

        # 测试模型性能
        ber, precision, recall, f1_score = evaluate_model(trained_model, test_loader, device)

        all_bers.append(ber)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1_scores.append(f1_score)

    print("\nSummary across evaluated SNRs:")
    print(f"BER: {all_bers}")
    print(f"Precision: {all_precisions}")
    print(f"Recall: {all_recalls}")
    print(f"F1 Score: {all_f1_scores}")