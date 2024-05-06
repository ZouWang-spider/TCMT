import torch
import torch.nn as nn
import torch.nn.functional as F

class FCLayer(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_labels):
        super(FCLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_labels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 不应用激活函数，以便使用 CrossEntropyLoss 计算损失

        # 在每个时间步应用 softmax 函数
        x = F.softmax(x, dim=2)
        return x

# 示例用法
input_size = 768  # 假设输入的特征维度为100
hidden_size1 = 256
hidden_size2 = 256
num_labels = 7  # 标签类别数
sequence_length = 10  # 序列长度

# 创建 FCLayer 实例
fc_layer = FCLayer(input_size, hidden_size1, hidden_size2, num_labels)

# 生成示例输入数据（假设 batch_size 为1）
x = torch.randn(1, sequence_length, input_size)

# 前向传播
output = fc_layer(x)

# 输出大小为 (batch_size, sequence_length, num_labels)，表示每个单词在7种标签上的概率分布
print("Output shape:", output.shape)