# 原始版，来源 Reliable and Secure Deep Learning-Based OFDM-DCSK
# Transceiver Design Without Delivery of Reference Chaotic Sequences
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import os

class TDNN(nn.Module):
    def __init__(self):
        super(TDNN, self).__init__()
        # 第一个卷积层，输入通道为1，输出通道为4，卷积核大小为2，膨胀系数为1
        self.conv1 = nn.Conv1d(1, 4, kernel_size=2, dilation=1)
        self.relu1 = nn.ReLU()
        # 第二个卷积层，输入通道为4，输出通道为8，卷积核大小为4，膨胀系数为2
        self.conv2 = nn.Conv1d(4, 8, kernel_size=4, dilation=2)
        self.relu2 = nn.ReLU()
        # 第三个卷积层，输入通道为8，输出通道为16，卷积核大小为8，膨胀系数为4
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
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.bn(x)
        x = self.fc2(x)
        return x

def train_model(model, train_data, train_labels, num_epochs, learning_rate):

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # 每5个epoch学习率衰减为原来的0.1倍
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
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
    # 参数设置
    # input_dim = 64  # input size
    # hidden_dim = 128  # Hidden dimension for TDNN and LSTM
    # output_dim = 2  # Output dimension (e.g., binary classification)
    # kernel_sizes = [2, 4, 8]  # Kernel sizes for TDNN
    # num_layers = 2  # Number of layers for LSTM

    # 创建模型实例
    model = LSTMDemodulator()
    # print(model)

    train_data = torch.randn(840, 32, 100, 1)
    train_labels = torch.randint(0, 2, (840, 32)).float()
    print(train_data.shape)
    print(train_labels.shape)

    # 模拟测试数据和标签
    test_data = torch.randn(210, 32, 100, 1)
    test_labels = torch.randint(0, 2, (210, 32)).float()

    # 训练轮数和学习率设置
    num_epochs = 1
    learning_rate = 0.01

    # 训练模型，获取训练后的模型、每轮损失列表和每轮准确率列表
    trained_model, epoch_losses, epoch_accuracies = train_model(model, train_data, train_labels, num_epochs, learning_rate)

    model_out_name = 'models/model_11'+'.pt'
    directory = os.path.dirname(model_out_name)
    # 如果不存在就创建
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model.state_dict(), model_out_name)
    print("模型保存成功")
    # model.load_state_dict(torch.load('./models/model_1_test.pt'))

    print("训练结束,开始推理：")
    # 提取出模型对象传入test_model函数进行测试
    accuracy = test_model(trained_model, test_data, test_labels)
    print(f'Accuracy of the model on the test data: {accuracy:.4f}%')