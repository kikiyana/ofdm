# Dual TCN+Self-Sttention

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
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


def train_model(model, train_data, train_labels, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # 使用学习率调度器，每5个epoch学习率衰减为原来的0.1倍
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_loss = float('inf')
    patience = 5
    no_improvement = 0

    # 用于存储每轮的损失和准确率
    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_loop = tqdm(enumerate(train_data), total=len(train_data), desc=f'Epoch {epoch + 1} Training')
        for i, inputs in train_loop:
            inputs = inputs.transpose(1, 2).to(device)
            labels = train_labels[i].to(device).long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_loop.set_postfix(loss=running_loss / (i + 1), accuracy=100 * correct / total)

        epoch_loss = running_loss / len(train_data)
        epoch_losses.append(epoch_loss)
        epoch_accuracy = 100 * correct / total
        epoch_accuracies.append(epoch_accuracy)

        print(f'Epoch {epoch + 1} Loss: {epoch_loss}, Accuracy: {epoch_accuracy}%')

        # 早停机制
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print("Early stopping!")
                break

        scheduler.step()

    return model, epoch_losses, epoch_accuracies


def test_model(model, test_data, test_labels):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    correct = 0
    total = 0
    test_loop = tqdm(enumerate(test_data), total=len(test_data), desc='Testing')
    with torch.no_grad():
        for i, inputs in test_loop:
            inputs = inputs.transpose(1, 2).to(device)
            labels = test_labels[i].to(device).long()

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loop.set_postfix(accuracy=100 * correct / total)  # 更新进度条显示的准确率

    accuracy = 100 * correct / total
    return accuracy


if __name__ == "__main__":
    # 创建模型实例
    model = LSTMDemodulator()
    # print(model)

    train_data = torch.randn(1000, 32, 100, 1)
    train_labels = torch.randint(0, 2, (1000, 32)).float()

    # 模拟测试数据和标签
    test_data = torch.randn(200, 32, 100, 1)
    test_labels = torch.randint(0, 2, (200, 32)).float()

    # 训练轮数和学习率设置
    num_epochs = 4
    learning_rate = 0.01

    # 训练模型，获取训练后的模型、每轮损失列表和每轮准确率列表
    trained_model, epoch_losses, epoch_accuracies = train_model(model, train_data, train_labels, num_epochs,
                                                                learning_rate)

    print("训练结束,开始推理：")
    # 提取出模型对象传入test_model函数进行测试
    accuracy = test_model(trained_model, test_data, test_labels)
    print(f'Accuracy of the model on the test data: {accuracy:.4f}%')