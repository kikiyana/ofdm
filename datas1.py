import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.fftpack import fft
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import os

# 强制使用CPU
device = torch.device("cpu")


# 定义TDNN层
class TDNN(nn.Module):
    def __init__(self):
        super(TDNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 4, kernel_size=2, dilation=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(4, 8, kernel_size=4, dilation=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(8, 16, kernel_size=8, dilation=4)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        return x


# 双向LSTM网络
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


# LSTMDemodulator解调
class LSTMDemodulator(nn.Module):
    def __init__(self):
        super(LSTMDemodulator, self).__init__()
        self.tdnn = TDNN()
        self.bilstm = BiLSTM(16, 128, 2)
        self.fc1 = FullyConnected(256, 128)
        self.bn = BatchNormalization(128)
        self.fc2 = FullyConnected(128, 2)

    def forward(self, x):
        x = self.tdnn(x)
        x = x.transpose(1, 2)
        x = self.bilstm(x)
        x = x[:, -1, :]  # 只取最后一个时间步的状态
        x = F.relu(self.fc1(x))
        x = self.bn(x)
        x = self.fc2(x)
        return x


def train_model(model, train_loader, num_epochs, learning_rate, device):
    model.to(device)  # 确保模型位于正确的设备上
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

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

    return model, epoch_losses, epoch_accuracies


# def test_model(model, test_loader, device):
#     model.eval()
#     model.to(device)
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for inputs, labels in tqdm(test_loader, desc='Testing'):
#             inputs, labels = inputs.unsqueeze(1).to(device), labels.to(device).long()
#
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#     accuracy = 100 * correct / total
#     return accuracy

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


# def calculate_ber(original_data, received_data):
#     # 将BPSK信号从 [-1, 1] 映射回 [0, 1]
#     original_bits = ((original_data + 1) / 2).astype(int)
#
#     # 计算误比特数
#     error_bits = np.sum(original_bits != received_data)
#
#     # 计算总比特数
#     total_bits = len(original_bits)
#
#     # 计算BER
#     ber = error_bits / total_bits
#
#     return ber

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

if __name__ == "__main__":
    # 参数设置
    N = 64  # OFDM子载波数量
    K = 128  # 混沌序列长度
    num_symbols = 1000  # 数据符号数量
    SNR_db = 20  # 信噪比 (dB)


    # 生成二阶切比雪夫多项式函数用于生成混沌序列
    def chebyshev_map(x):
        return 1 - 2 * x ** 2


    # 生成BPSK调制的用户数据
    user_data = np.random.randint(0, 2, num_symbols) * 2 - 1  # [-1, 1]

    # 生成混沌序列 x
    x = np.zeros(K)
    x[0] = np.random.uniform(-1, 1)  # 初始值
    for k in range(1, K):
        x[k] = chebyshev_map(x[k - 1])

    # 调制过程：将用户数据与混沌序列相乘
    modulated_symbols = np.zeros((num_symbols, K))
    for n in range(num_symbols):
        modulated_symbols[n, :] = user_data[n] * x

    # OFDM调制
    ofdm_symbols = np.zeros((num_symbols, N), dtype=complex)
    for k in range(K):
        ofdm_symbols[:, k % N] += modulated_symbols[:, k] / np.sqrt(N)

    # 添加循环前缀 (CP)
    cp_length = N // 4
    ofdm_symbols_with_cp = np.concatenate([ofdm_symbols[:, -cp_length:], ofdm_symbols], axis=1)

    # 信道传输：简单地添加AWGN噪声
    snr_linear = 10 ** (SNR_db / 10)
    noise_variance = 1 / snr_linear
    noise = np.sqrt(noise_variance / 2) * (
            np.random.randn(*ofdm_symbols_with_cp.shape) + 1j * np.random.randn(*ofdm_symbols_with_cp.shape))
    received_ofdm_symbols = ofdm_symbols_with_cp + noise

    # 接收端处理：去除CP并进行FFT
    received_ofdm_symbols_no_cp = received_ofdm_symbols[:, cp_length:]
    fft_received_symbols = fft(received_ofdm_symbols_no_cp, axis=1)

    # 频域均衡处理（FFE）
    # 假设信道频率响应是已知的，这里我们使用一个简单的示例信道响应
    channel_frequency_response = np.ones(N)  # 实际情况中，这需要从信道估计中获得
    # 应用信道频率响应
    fft_received_symbols_equalized = fft_received_symbols * channel_frequency_response
    # 应用均衡后的信号
    fft_received_symbols_equalized = fft_received_symbols_equalized / np.abs(channel_frequency_response) ** 2

    # 构建训练数据 yn 和对应的标签
    yn = fft_received_symbols.reshape(num_symbols, -1).real  # 只取实部作为输入特征

    # 将用户比特从 [-1, 1] 映射到 [0, 1]
    labels = ((user_data + 1) / 2).astype(int)  # 将 -1 转换为 0, 1 保持不变

    # 创建数据集和加载器
    dataset = TensorDataset(torch.tensor(yn, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
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

    # # 保存模型
    # model_out_name = 'models/model_11.pt'
    # os.makedirs(os.path.dirname(model_out_name), exist_ok=True)
    # torch.save(trained_model.state_dict(), model_out_name)
    # print("模型保存成功")

    print("训练结束,开始推理：")

    # # 使用相同的数据集创建测试加载器
    # test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    #
    # # 测试模型
    # accuracy = test_model(trained_model, test_loader, device)
    # print(f'Test Accuracy of the model: {accuracy:.4f}%')

    # 使用相同的数据集创建测试加载器
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 测试模型
    predictions, true_labels = test_model(trained_model, test_loader, device)

    # 计算BER
    ber = calculate_ber(user_data, predictions)
    print(f'Bit Error Rate (BER): {ber:.6f}')