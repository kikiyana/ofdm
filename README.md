### 强化学习：

RL+Dual TCN+Self-Sttention



pip install einops


### 创新点：

监督学习部分：通过交叉熵损失优化模型的分类能力。
强化学习部分：通过Q-learning机制优化模型的决策能力，使得模型能够在做出分类的同时，学习到更好的策略。
混合模型适用于需要同时进行分类和决策优化的任务，例如复杂的序列预测问题、增强现实应用中的决策制定等


1.基于信道模型更新状态

```
# 训练模型
def train_model(model, train_loader, num_epochs, learning_rate, device, scheduler_type='plateau', snr_db=10):
    model.to(device)  # 确保模型位于正确的设备上
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)  # 调整学习率调度器
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    epoch_losses = []
    epoch_accuracies = []

    # DRQN 相关参数
    gamma = 0.99  # 折扣因子
    update_target_freq = 100  # 目标网络更新频率
    target_model = LSTMDemodulator().to(device)  # 初始化目标网络
    target_model.load_state_dict(model.state_dict())  # 同步初始权重

    # 定义信道噪声添加函数
    def add_awgn_noise(signal, snr_db):
        snr_linear = 10 ** (snr_db / 10)
        signal_power = torch.mean(torch.abs(signal) ** 2)
        noise_power = signal_power / snr_linear
        noise = torch.sqrt(noise_power) * (torch.randn(signal.size()) + 1j * torch.randn(signal.size()))
        return signal + noise

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

            # Q-learning 损失，奖励基于误码率
            _, predicted = torch.max(outputs.data, 1)
            error_rate = (predicted != labels).float().mean()  # 计算误码率
            rewards = 1 - error_rate  # 奖励基于误码率

            selected_q = outputs.gather(1, predicted.unsqueeze(1)).squeeze()

            # 模拟下一个状态：重新添加噪声
            next_inputs = add_awgn_noise(inputs.squeeze(1), snr_db).unsqueeze(1).to(device)
            with torch.no_grad():
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
    return model, epoch_losses, epoch_accuracies
```

2.基于历史信息预测下一个状态

此方案使用一个简单的线性预测模型，根据当前状态来预测下一个状态。

```
# 简单的线性预测模型示例
class LinearPredictor(nn.Module):
    def __init__(self, input_dim):
        super(LinearPredictor, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.linear(x)


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

    # DRQN 相关参数
    gamma = 0.99  # 折扣因子
    update_target_freq = 100  # 目标网络更新频率
    target_model = LSTMDemodulator().to(device)  # 初始化目标网络
    target_model.load_state_dict(model.state_dict())  # 同步初始权重

    input_dim = train_loader.dataset.tensors[0].shape[1]
    predictor = LinearPredictor(input_dim).to(device)
    predictor_optimizer = optim.Adam(predictor.parameters(), lr=learning_rate)

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

            # Q-learning 损失，奖励基于误码率
            _, predicted = torch.max(outputs.data, 1)
            error_rate = (predicted != labels).float().mean()  # 计算误码率
            rewards = 1 - error_rate  # 奖励基于误码率

            selected_q = outputs.gather(1, predicted.unsqueeze(1)).squeeze()

            # 预测下一个状态
            next_inputs = predictor(inputs.squeeze(1)).unsqueeze(1)
            with torch.no_grad():
                next_outputs, _ = target_model(next_inputs, hidden)
            max_next_q = next_outputs.detach().max(1)[0]
            target_q = rewards + gamma * max_next_q
            q_loss = F.mse_loss(selected_q, target_q)

            # 总损失
            total_loss = class_loss + 0.1 * q_loss  # 调整权重

            total_loss.backward()
            optimizer.step()

            # 模拟获取真实的下一个状态（这里简化为下一个批次的输入，实际应用需根据具体场景获取）
            if i < len(train_loader) - 1:
                actual_next_inputs = next(iter(train_loader))[0].unsqueeze(1).to(device)
            else:
                actual_next_inputs = inputs

            # 更新预测器
            predictor_loss = F.mse_loss(next_inputs, actual_next_inputs)
            predictor_optimizer.zero_grad()
            predictor_loss.backward()
            predictor_optimizer.step()

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
    return model, epoch_losses, epoch_accuracies
```


```
应用场景
SNR：主要用于模拟通信系统或评估整体信号质量，特别是在无线通信、音频处理等领域。
Eb/N0：主要用于数字通信系统，尤其是在设计和评估调制方案、编码技术以及误码率性能时。
总结
SNR 衡量的是信号功率与噪声功率的比值，适用于模拟和数字通信系统。
Eb/N0 衡量的是每比特能量与噪声功率谱密度的比值，特别适用于数字通信系统，并且与系统的比特率和带宽有关。
```

7 到 10 页的
不同信噪比： [0, 5, 10, 15, 20, 25, 30] 
不同干扰：['awgn','rayleigh','multi_path_fast_rayleigh', 'rician']

不同 num_symbols: [1000,10000,50000,100000]

不同的通信场景下，常见的 SNR 范围有所不同，以下是一些典型场景的 SNR 范围示例：
室内 Wi - Fi 通信：通常在 10 - 30dB 之间。在距离无线路由器较近且周围干扰较少的情况下，SNR 可能会达到 25 - 30dB；而在距离较远或者有较多干扰源的地方，SNR 可能会降至 10 - 15dB。
蜂窝移动通信（如 4G、5G）：在城市环境中，SNR 一般在 5 - 20dB 之间。在信号覆盖较好的区域，如基站附近，SNR 可能会接近 20dB；而在信号较弱的边缘区域，SNR 可能会低至 5dB 甚至更低。


不同信噪比： [0, 5, 10, 15, 20] 
不同干扰：['rayleigh','multi_path_fast_rayleigh', 'rician']
不同 num_symbols: [1000,10000,100000]



单独验证模型num_symbols
固定num_symbols:500000
epoch:20

1.baseline.py:
基于TDNN的baseline模型，使用PyTorch实现。
t0.py


2.baseline_TCN.py:
基于TCN的baseline模型，使用PyTorch实现。
t3.py


3.baseline_DualTCN.py:
基于双TCN的baseline模型，使用PyTorch实现。
pythonn t2.py


4.强化学习版 RPSR - DQN RL_TCN.py:
test3.py


5.强化学习版：RPSR - DQN RL_DualTCN.py:
test4.py


6.强化学习版Transformer-based DRQN RL_TCN.py:
d1.py


7.强化学习版Transformer-based DRQN RL_DualTCN.py:
code1.py是第2版强化学习，d2.py是第3版强化学习
test5.py第三版测试数据集大小





8.强化学习版Transformer-based DRQN RL_TDNN.py:
a2.py是第2版强化学习，a3.py是第3版强化学习



t01.py, t02.py:
包含最新 信道噪声 生成方法。





实验结果：

以下是根据不同信道和信噪比（SNR）列出的模型误码率（BER）和训练时间（秒）的 Markdown 表格：

### Rayleigh信道

| SNR (dB)           | Dual_TCN | rpsr_dqn_Dual_TCN | rpsr_dqn_TCN | TCN     | TDNN    | TDRQN_Dual_TCN | TDRQN_TCN |
| ------------------ | -------- | ----------------- | ------------ | ------- | ------- | -------------- | --------- |
| **BER**            |          |                   |              |         |         |                |           |
| 0                  | 0.0228   | 0.0429            | 0.0601       | 0.03125 | 0.03655 | 0.0189         | 0.0412    |
| 5                  | 0.0078   | 0.00735           | 0.01555      | 0.0111  | 0.00875 | 0.00315        | 0.0124    |
| 10                 | 0.00115  | 0.0013            | 0.0027       | 0.00395 | 0.00275 | 0.0024         | 0.00485   |
| 15                 | 0.0006   | 0.00105           | 0.00135      | 0.001   | 0.00065 | 0.00125        | 0.0057    |
| 20                 | 0.00015  | 0.00045           | 0.00075      | 0.0007  | 0.00025 | 0.0005         | 0.0024    |
| **训练时间（秒）** |          |                   |              |         |         |                |           |
| 0                  | 590.74   | 2205.86           | 2063.58      | 741.98  | 532.34  | 800.50         | 804.94    |
| 5                  | 584.83   | 2215.68           | 2070.57      | 707.96  | 629.16  | 773.61         | 805.88    |
| 10                 | 670.23   | 2224.12           | 2083.38      | 714.55  | 546.19  | 785.49         | 807.89    |
| 15                 | 639.62   | 2248.63           | 2087.11      | 730.24  | 543.12  | 783.26         | 795.94    |
| 20                 | 579.64   | 1932.33           | 2014.26      | 711.95  | 675.84  | 780.53         | 795.31    |

### Rician信道

| SNR (dB)           | Dual_TCN | rpsr_dqn_Dual_TCN | rpsr_dqn_TCN | TCN     | TDNN    | TDRQN_Dual_TCN | TDRQN_TCN |
| ------------------ | -------- | ----------------- | ------------ | ------- | ------- | -------------- | --------- |
| **BER**            |          |                   |              |         |         |                |           |
| 0                  | 0.0058   | 0.00685           | 0.00845      | 0.01955 | 0.00915 | 0.00445        | 0.0121    |
| 5                  | 0.0068   | 0.0025            | 0.003        | 0.00665 | 0.0064  | 0.00425        | 0.0059    |
| 10                 | 0.0017   | 0.00095           | 0.00115      | 0.00095 | 0.00775 | 0.00075        | 0.0091    |
| 15                 | 0.0013   | 0.0007            | 0.00115      | 0.00435 | 0.0092  | 0.0006         | 0.00885   |
| 20                 | 0.0004   | 0.00045           | 0.0006       | 0.00485 | 0.0033  | 0.0013         | 0.0016    |
| **训练时间（秒）** |          |                   |              |         |         |                |           |
| 0                  | 605.75   | 1759.25           | 1479.59      | 707.64  | 594.20  | 785.78         | 804.17    |
| 5                  | 682.54   | 2817.49           | 2566.50      | 724.94  | 616.93  | 790.15         | 810.84    |
| 10                 | 586.57   | 2794.36           | 2672.73      | 708.58  | 541.37  | 787.99         | 802.61    |
| 15                 | 589.68   | 2809.17           | 2662.96      | 707.17  | 615.11  | 772.85         | 801.97    |
| 20                 | 673.26   | 2170.14           | 2291.53      | 722.76  | 608.48  | 805.86         | 809.90    |

### Nakagami信道

| SNR (dB)           | Dual_TCN | rpsr_dqn_Dual_TCN | rpsr_dqn_TCN | TCN     | TDNN    | TDRQN_Dual_TCN | TDRQN_TCN |
| ------------------ | -------- | ----------------- | ------------ | ------- | ------- | -------------- | --------- |
| **BER**            |          |                   |              |         |         |                |           |
| 0                  | 0.001    | 0.0007            | 0.0013       | 0.00175 | 0.00065 | 0.00065        | 0.00455   |
| 5                  | 0.0      | 5e-05             | 5e-05        | 0.0003  | 0.0001  | 5e-05          | 0.00025   |
| 10                 | 5e-05    | 0.0               | 0.0          | 0.0     | 0.0     | 0.0            | 5e-05     |
| 15                 | 5e-05    | 0.0               | 0.0          | 0.0     | 0.0     | 0.0            | 0.0       |
| 20                 | 0.0      | 0.0               | 0.0          | 0.0     | 0.0     | 0.0            | 0.0       |
| **训练时间（秒）** |          |                   |              |         |         |                |           |
| 0                  | 601.93   | 1769.61           | 2024.86      | 705.01  | 612.34  | 771.15         | 785.97    |
| 5                  | 594.03   | 1578.53           | 1495.71      | 705.87  | 577.70  | 694.60         | 763.45    |
| 10                 | 627.17   | 1533.06           | 1466.86      | 715.85  | 567.38  | 703.41         | 779.45    |
| 15                 | 633.31   | 1527.92           | 1462.55      | 707.65  | 654.84  | 694.33         | 761.91    |
| 20                 | 595.07   | 1420.85           | 1478.85      | 677.14  | 555.31  | 692.43         | 740.43    |