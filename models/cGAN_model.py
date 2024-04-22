import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import numpy as np

class ContentAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ContentAttention, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # 计算注意力权重
        x = torch.tanh(self.linear1(x))
        x = self.linear2(x)
        attention_weights = self.softmax(x)
        # 加权求和
        output = torch.sum(x * attention_weights, dim=1, keepdim=True)
        return output

class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, output_size)
        self.relu = nn.ReLU()
        self.attention = ContentAttention(output_size, 128)

    def forward(self, z, condition):
        # 将条件数据与输入噪声拼接
        x = torch.cat((z, condition), dim=1)
        x = self.relu(self.fc1(x))
        # 使用内容注意力机制调整生成的数据
        attention_weights = self.attention(x)
        x = x * attention_weights
        x = self.fc2(x)
        return x

# 定义鉴别器模型
class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = self.sigmoid(self.fc2(x))
        return x

# 定义CGAN模型
class CGAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(CGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, z, c):
        fake_data = self.generator(torch.cat((z, c), dim=1))
        return fake_data

# 定义系统发育树信息

Edge = torch.randn(13307, 2)  # 用随机数据代替
Number_of_Nodes = 6634
Tip_Labels = torch.tensor([str(i) for i in range(6674)])
Edge_Lengths = torch.randn(13307)  # 用随机数据代替
Node_Labels = torch.tensor([str(i) for i in range(6633)])

# 将信息转换为张量
Edge = torch.tensor(Edge)
Number_of_Nodes = torch.tensor(Number_of_Nodes)
Tip_Labels = torch.tensor(Tip_Labels)
Edge_Lengths = torch.tensor(Edge_Lengths)
Node_Labels = torch.tensor(Node_Labels)

# 定义生成器和鉴别器输入维度
generator_input_size = 100
discriminator_input_size = 6674  # 根据系统发育树信息中的 Tip_Labels 维度确定

# 实例化生成器、鉴别器和CGAN模型
generator = Generator(generator_input_size + discriminator_input_size, 1)
discriminator = Discriminator(discriminator_input_size)
cgan = CGAN(generator, discriminator)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 训练CGAN模型
num_epochs = 100
batch_size = 64

for epoch in range(num_epochs):
    # 省略数据加载部分，你需要根据你的数据格式加载数据
    # 这里假设使用 DataLoader 加载数据
    
    for batch_idx, (real_data, _) in enumerate(train_loader):
        # 生成随机噪声和条件
        z = torch.randn(batch_size, generator_input_size)
        c = Tip_Labels[torch.randint(0, len(Tip_Labels), (batch_size,))]
        
        # 训练鉴别器
        discriminator.zero_grad()
        real_data = real_data.view(-1, discriminator_input_size)
        real_output = discriminator(real_data)
        fake_data = cgan.generator(torch.cat((z, c), dim=1)).detach()
        fake_output = discriminator(fake_data)
        real_loss = criterion(real_output, torch.ones_like(real_output))
        fake_loss = criterion(fake_output, torch.zeros_like(fake_output))
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_d.step()
        
        # 训练生成器
        generator.zero_grad()
        z = torch.randn(batch_size, generator_input_size)
        c = Tip_Labels[torch.randint(0, len(Tip_Labels), (batch_size,))]
        fake_data = cgan.generator(torch.cat((z, c), dim=1))
        output = discriminator(fake_data)
        g_loss = criterion(output, torch.ones_like(output))
        g_loss.backward()
        optimizer_g.step()
        
        # 打印训练信息
        if batch_idx % 100 == 0:
            print('Epoch [{}/{}], Batch {}, d_loss: {:.4f}, g_loss: {:.4f}'.format(
                epoch+1, num_epochs, batch_idx, d_loss.item(), g_loss.item()))

# 生成器的输出就是生成的假数据
z = torch.randn(10, generator_input_size)
c = Tip_Labels[torch.randint(0, len(Tip_Labels), (10,))]
fake_data = cgan.generator(torch.cat((z, c), dim=1))
print(fake_data)
