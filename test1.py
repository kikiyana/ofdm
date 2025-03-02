import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.nn.utils import weight_norm
import torch.nn.functional as F


from torch.nn.utils import weight_norm


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

    dataset = TensorDataset(equalized_symbols.float(), torch.tensor(labels, dtype=torch.long),
                         torch.tensor(original_bits, dtype=torch.long))
    return dataset


# 训练模型
def train_model(model, train_loader, num_epochs, learning_rate, device, scheduler_type='plateau'):
    model.to(device)  # 确保模型位于正确的设备上
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)  # 调整学习率调度器
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    epoch_losses = []
    epoch_accuracies = []

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
    return model, epoch_losses, epoch_accuracies


# 评估模型
def evaluate_model(model, test_loader, device):
    model.eval()
    total_error_bits = 0
    total_bits = 0
    correct = 0
    precision = 0
    recall = 0
    f1_score = 0

    with torch.no_grad():
        for i, (inputs, labels, original_bits) in enumerate(tqdm(test_loader, desc='Testing')):
            inputs, labels = inputs.unsqueeze(1).to(device), labels.to(device).long()
            original_bits = original_bits.to(device).long()

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # 计算误码率相关
            received_bits = predicted.cpu().numpy()
            original_bits = original_bits.cpu().numpy()
            # 将 original_bits 的取值范围从 [-1, 1] 转换为 [0, 1]
            original_bits = ((original_bits + 1) / 2).astype(int)
            # 确保形状和数据类型一致
            if original_bits.shape!= received_bits.shape:
                min_length = min(original_bits.shape[0], received_bits.shape[0])
                original_bits = original_bits[:min_length]
                received_bits = received_bits[:min_length]
            error_bits = np.sum(original_bits!= received_bits)
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

    ber = total_error_bits / total_bits
    print(f"Bit Error Rate (BER): {ber:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
    return ber, precision, recall, f1_score


# 添加 AWGN 噪声
def add_awgn_noise(signal, snr_db):
    # 添加 AWGN 噪声
    snr_linear = 10 ** (snr_db / 10)
    signal_power = torch.mean(torch.abs(signal) ** 2)
    noise_power = signal_power / snr_linear
    noise = torch.sqrt(noise_power) * (torch.randn(signal.size()) + 1j * torch.randn(signal.size()))
    return signal + noise


# 添加瑞利噪声
def add_rayleigh_noise(signal, snr_db):
    # 添加瑞利噪声
    snr_linear = 10 ** (snr_db / 10)
    signal_power = torch.mean(torch.abs(signal) ** 2)
    noise_power = signal_power / snr_linear
    h = torch.sqrt(torch.randn(signal.size()) ** 2 + torch.randn(signal.size()) ** 2) / np.sqrt(2)
    noise = torch.sqrt(noise_power) * (torch.randn(signal.size()) + 1j * torch.randn(signal.size())) / h
    return signal * h + noise


# 添加莱斯噪声
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
    # MMSE 均衡器
    return received_signal / (channel_estimate + 1e-10)


# ZF 均衡器
def zf_equalizer(received_signal, channel_estimate):
    # ZF 均衡器
    # 计算信道估计的逆
    channel_inverse = 1 / channel_estimate  # 这里假设 channel_estimate 是一个标量或者每个元素都不为零的向量
    return received_signal * channel_inverse  # 将接收信号乘以信道估计的逆


# 模拟攻击
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


# 生成信道估计
def generate_channel_estimate(received_signal):
    # 简单的最小二乘信道估计
    # 这里仅作为示例，实际应用中可使用更复杂的信道估计方法
    return torch.mean(received_signal, dim=0)


# DQN 网络
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# 强化学习环境
class RLNetworkEnv:
    def __init__(self, N, K, num_symbols, snr_db, channel_model='awgn'):
        self.N = N
        self.K = K
        self.num_symbols = num_symbols
        self.snr_db = snr_db
        self.channel_model = channel_model
        self.best_ber = float('inf')
        self.current_state = self.get_state()

    def get_state(self):
        # 这里假设初始网络结构
        model = LSTMDemodulator()
        dataset = generate_dataset(self.N, self.K, self.num_symbols, self.snr_db, self.channel_model)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, _, epoch_accuracies = train_model(model, train_loader, 3, 0.001, device)
        ber, _, _, _ = evaluate_model(model, test_loader, device)
        state = np.array([self.N, self.K, self.num_symbols, self.snr_db] + epoch_accuracies + [ber])
        return state

    def reset(self):
        return self.get_state()

    def step(self, action):
        # 应用动作修改网络结构
        # 这里简单示例，根据动作修改网络结构，可根据实际需求完善
        if action == 0:
            # 增加一层
            pass
        elif action == 1:
            # 减少一层
            pass
        # 重新训练和评估模型
        model = LSTMDemodulator()
        dataset = generate_dataset(self.N, self.K, self.num_symbols, self.snr_db, self.channel_model)
        train_size = int(0.8 * len(dataset) )
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, _, epoch_accuracies = train_model(model, train_loader, 3, 0.001, device)
        ber, _, _, _ = evaluate_model(model, test_loader, device)
        next_state = np.array([self.N, self.K, self.num_symbols, self.snr_db] + epoch_accuracies + [ber])
        # 计算奖励
        reward = -ber if ber < self.best_ber else -1.0
        self.best_ber = min(ber, self.best_ber)
        done = False  # 可根据终止条件修改
        return next_state, reward, done


# DQN 智能体
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state)
                target = (reward + self.gamma * torch.max(self.target_model(next_state)[0]).item())
            state = torch.FloatTensor(state)
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = F.mse_loss(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    # 参数设置
    N = 64  # OFDM 子载波数量 64
    K = 128  # 混沌序列长度 128
    num_symbols = 10000  # 数据符号数量
    SNR_db = 10  # 不同信噪比 (dB)
    channel_model = 'rayleigh'

    # 尝试使用 CPU 设备，避免 GPU 可能出现的 cuDNN 错误
    device = torch.device("cuda")

    # 生成数据集
    dataset = generate_dataset(N, K, num_symbols, SNR_db, channel_model)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # 减小批量大小，避免 GPU 内存问题
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 创建模型实例
    model = LSTMDemodulator()
    model.to(device)

    # 训练轮数和学习率设置
    num_epochs = 1
    learning_rate = 0.001

    # 训练模型
    trained_model, epoch_losses, epoch_accuracies = train_model(
        model,
        train_loader,
        num_epochs,
        learning_rate,
        device,
        scheduler_type='cosine'
    )

    # 测试模型性能
    ber, precision, recall, f1_score = evaluate_model(trained_model, test_loader, device)

    # 强化学习部分
    env = RLNetworkEnv(N, K, num_symbols, SNR_db, channel_model)
    state_size = len(env.current_state)
    action_size = 2  # 假设只有两种动作，可根据实际需求增加
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    episodes = 1

    for e in range(episodes):
        state = env.reset()
        for time in range(2):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode: {e + 1}/{episodes}, Score: {time}")
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if (e + 1) % 10 == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())
    print("Training completed.")