import torch
import torch.nn as nn
import torch.optim as optim

# 1. 准备数据：3个特征(面积㎡, 房间数, 建造年份) -> 预测房价(万元)
# 这里是模拟数据，实际中会使用真实的房屋数据集
X_raw = torch.tensor([
    [80.0, 2, 2010],   # 80㎡, 2室, 2010年建
    [120.0, 3, 2015],  # 120㎡, 3室, 2015年建
    [95.0, 2, 2005],   # 95㎡, 2室, 2005年建
    [150.0, 4, 2020]   # 150㎡, 4室, 2020年建
], dtype=torch.float32)

y_raw = torch.tensor([180.0, 320.0, 220.0, 450.0], dtype=torch.float32)  # 对应房价

# 数据标准化：防止数值不稳定
# 对特征进行标准化 (X - mean) / std
X_mean = X_raw.mean(dim=0)
X_std = X_raw.std(dim=0)
X = (X_raw - X_mean) / X_std

# 对目标值进行标准化
y_mean = y_raw.mean()
y_std = y_raw.std()
y = (y_raw - y_mean) / y_std

# 2. 定义模型：使用线性层实现从3个特征到1个输出(房价)的映射
class HousePriceModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入3个特征，输出1个预测值(房价)
        self.linear = nn.Linear(in_features=3, out_features=1)
    
    def forward(self, x):
        return self.linear(x).flatten()  # 展平输出为一维

# 3. 初始化模型、损失函数和优化器
model = HousePriceModel()
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降，调整学习率

# 4. 训练模型
for epoch in range(1000):
    # 前向传播：计算预测值
    y_pred = model(X)
    
    # 计算损失
    loss = criterion(y_pred, y)
    
    # 反向传播和参数更新
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新权重和偏置
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

# 5. 预测新数据
new_house_raw = torch.tensor([[110.0, 3, 2018]], dtype=torch.float32)  # 新房屋特征
# 对新数据进行同样的标准化处理
new_house = (new_house_raw - X_mean) / X_std
predicted_price_normalized = model(new_house)
# 将预测结果反标准化回原始尺度
predicted_price = predicted_price_normalized * y_std + y_mean
print(f'预测房价: {predicted_price.item():.2f} 万元')