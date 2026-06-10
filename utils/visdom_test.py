import numpy as np
from visdom import Visdom

# 初始化 Visdom 客户端
viz = Visdom()

# 检查 Visdom 服务器是否连接成功
if not viz.check_connection():
    raise Exception("Visdom server not running. Please start the server with `python -m visdom.server`.")

# 测试标量数据可视化（绘制曲线）
print("Testing scalar data visualization...")
viz.line(
    Y=[0],  # 初始 Y 值
    X=[0],  # 初始 X 值
    win='loss',  # 窗口 ID
    opts=dict(title='Training Loss', xlabel='Epoch', ylabel='Loss')  # 配置选项
)
for epoch in range(10):
    loss = 1 / (epoch + 1)  # 模拟损失值
    viz.line(
        Y=[loss],  # 新的 Y 值
        X=[epoch],  # 新的 X 值
        win='loss',  # 窗口 ID
        update='append'  # 追加模式
    )

# 测试图像可视化
print("Testing image visualization...")
image = np.random.rand(3, 256, 256)  # 生成随机图像（3 通道，256x256）
viz.image(
    image,
    win='image',
    opts=dict(title='Random Image', caption='This is a random image.')
)

# 测试文本可视化
print("Testing text visualization...")
viz.text(
    'Hello, Visdom!',
    win='text',
    opts=dict(title='Text Window')
)

# 测试直方图可视化
print("Testing histogram visualization...")
data = np.random.rand(1000)  # 生成随机数据
viz.histogram(
    data,
    win='histogram',
    opts=dict(title='Data Distribution', numbins=20)
)

# 测试散点图可视化
print("Testing scatter plot visualization...")
points = np.random.rand(100, 2)  # 生成 100 个 2D 点
viz.scatter(
    points,
    win='scatter',
    opts=dict(title='2D Scatter Plot', markersize=5)
)

# 测试条形图可视化
print("Testing bar chart visualization...")
values = np.random.rand(10)  # 生成 10 个随机值
viz.bar(
    values,
    win='bar',
    opts=dict(title='Bar Chart', xlabel='Index', ylabel='Value')
)

print("Visdom test completed. Check the web interface for results.")