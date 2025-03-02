# 只用baseline1.py TDNN结构, Q-learing修改版

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import classification_report
import torch.optim as optim
import time

# 强制使用 CPU
device = torch.device("cuda")

# 定义 TDNN 层
class TDNN(nn.Module):
    def __init__(self):
        super(TDNN, self).__init__()
        self.tdnn = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=2, dilation=1, padding=1),  # 减小 kernel_size 或增加 padding
            nn.ReLU(),
            nn.Conv1d(4, 8, kernel_size=3, dilation=1, padding=1),  # 调整 kernel_size 和 padding
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, dilation=1, padding=1),
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

    def forward(self, x, hidden=None):
        x = self.tdnn(x)
        x = x.transpose(1, 2)
        x, hidden = self.bilstm(x, hidden)
        x = x[:, -1, :]  # 只取最后一个时间步的状态
        x = F.relu(self.fc1(x))
        x = self.bn(x)
        x = self.dropout(x)  # 应用 Dropout
        x = self.fc2(x)
        return x, hidden


# 生成数据集
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
    def add_awgn_noise(signal, snr_db):
        snr_linear = 10 ** (snr_db / 10)
        signal_power = torch.mean(torch.abs(signal) ** 2)
        noise_power = signal_power / snr_linear
        noise = torch.sqrt(noise_power) * (torch.randn(signal.size()) + 1j * torch.randn(signal.size()))
        return signal + noise

    def add_rayleigh_noise(signal, snr_db):
        snr_linear = 10 ** (snr_db / 10)
        signal_power = torch.mean(torch.abs(signal) ** 2)
        noise_power = signal_power / snr_linear
        h = torch.sqrt(torch.randn(signal.size()) ** 2 + torch.randn(signal.size()) ** 2) / np.sqrt(2)
        noise = torch.sqrt(noise_power) * (torch.randn(signal.size()) + 1j * torch.randn(signal.size())) / h
        return signal * h + noise

    def add_rician_noise(signal, snr_db):
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

    if channel_model == 'awgn':
        received_ofdm_symbols = add_awgn_noise(torch.tensor(ofdm_symbols_with_cp), snr_db)
    elif channel_model == 'rayleigh':
        received_ofdm_symbols = add_rayleigh_noise(torch.tensor(ofdm_symbols_with_cp), snr_db)
    elif channel_model == 'rician':
        received_ofdm_symbols = add_rician_noise(torch.tensor(ofdm_symbols_with_cp), snr_db)
    else:
        raise ValueError("Invalid channel model")

    # 接收端处理：去除 CP 并进行 FFT
    received_ofdm_symbols_no_cp = received_ofdm_symbols[:, cp_length:]
    fft_received_symbols = torch.fft.fft(received_ofdm_symbols_no_cp, dim=1)

    # 更复杂的信道估计
    def generate_channel_estimate(received_signal):
        return torch.mean(received_signal, dim=0)

    def mmse_equalizer(received_signal, channel_estimate):
        return received_signal / (channel_estimate + 1e-10)

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

    dataset = TensorDataset(yn.float(), torch.tensor(labels, dtype=torch.long), torch.tensor(original_bits, dtype=torch.long))
    return dataset


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
    # 冻结目标网络的权重
    for param in target_model.parameters():
        param.requires_grad = False

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        hidden = None  # 初始化隐藏状态为 None

        for i, (inputs, labels, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1} Training')):
            inputs, labels = inputs.unsqueeze(1).to(device), labels.to(device).long()  # 添加通道维度并转换标签类型

            optimizer.zero_grad()
            outputs, hidden = model(inputs, hidden)
            # 分离隐藏状态的梯度
            if isinstance(hidden, tuple):
                hidden = (hidden[0].detach(), hidden[1].detach())
            else:
                hidden = hidden.detach()

            # 计算分类损失
            class_loss = criterion(outputs, labels)

            # Q-learning 损失，奖励基于预测是否正确
            _, predicted = torch.max(outputs.data, 1)
            rewards = (predicted == labels).float() * 1.0  # 奖励：1表示正确，0表示错误

            # 计算当前动作的 Q 值
            selected_q = outputs.gather(1, predicted.unsqueeze(1)).squeeze()

            # 计算目标 Q 值
            with torch.no_grad():
                # 假设 next_inputs 是当前 inputs 的下一个时间步
                seq_length = inputs.size(2)
                if seq_length <= 1:
                    next_inputs = inputs  # 如果序列长度为 1，则 next_inputs 与 inputs 相同
                else:
                    split_point = seq_length - 1  # 确保 next_inputs 至少有一个时间步
                    next_inputs = inputs[:, :, split_point:]

                # 前向传播 next_inputs 到目标网络
                next_outputs, _ = target_model(next_inputs, hidden)
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

    # 结束时间
    end_time = time.time()
    print(f'Total Training Time: {end_time - start_time:.2f}s')

    # 计算 BER
    ber = total_error_bits / total_bits

    # 计算分类报告
    classification_report_data = classification_report(labels_list, predictions, output_dict=True)
    precision = classification_report_data['macro avg']['precision']
    recall = classification_report_data['macro avg']['recall']
    f1_score = classification_report_data['macro avg']['f1-score']

    print(f"Bit Error Rate (BER): {ber:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
    return ber, precision, recall, f1_score


if __name__ == "__main__":
    # 参数设置
    N = 128  # OFDM 子载波数量
    K = 256  # 混沌序列长度
    num_symbols = 1000  # 数据符号数量
    SNR_dbs = [10]  # 不同信噪比 (dB)

    # 存储不同 SNR 下的性能指标
    all_bers = []
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    all_losses = []
    all_accuracies = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for SNR_db in SNR_dbs:
        # 生成数据集
        dataset = generate_dataset(N, K, num_symbols, SNR_db, channel_model='awgn')

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