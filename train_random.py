import torch
from torch import nn
from torch.optim import SGD
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import get_model
from utils import setup_device, save_model, load_data
from config import device, num_epochs, lr, momentum, step_size, gamma
from torch.utils.tensorboard import SummaryWriter

def setup_optimizer(model):
    # 为所有参数设置统一的学习率
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    return optimizer


def train():
    # 设置设备
    device = setup_device()

    # 数据加载
    data_loaders = load_data(split_train=True)
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']

    # 模型加载
    model = get_model(pretrained=False)
    model.to(device)

    # 定义优化器和损失函数
    # optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    # 在训练脚本中设置优化器
    optimizer = setup_optimizer(model)
    criterion = nn.CrossEntropyLoss()
    # 学习率调整器（每step_size次，乘gamma）
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # 初始化TensorBoard
    # writer = SummaryWriter()

    # 训练和验证循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()   # 将之前的梯度清零，否则会累计
            loss.backward()         # 计算梯度，并储存在param的.grad中
            optimizer.step()        # 根据参数的grad进行参数更新

            train_loss += loss.item() * images.size(0)    # loss.item()将损失的tensor变为python的float，再乘上batch size
        
        # 平均训练损失
        train_loss /= len(train_loader.dataset)
        # writer.add_scalar('Loss/train', train_loss, epoch)    # 将标量数据写入日志，tensor board的x轴为epoch

        # 验证过程
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():   # 不追踪梯度，提高模型效率
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)   # 在预测结果的第一个维度上找到最大索引值
                total += labels.size(0)    # 累加本批次中的样本数
                correct += (predicted == labels).sum().item()   # 累加预测正确的样本数量

        val_loss /= len(val_loader.dataset)
        val_accuracy = 100 * correct / total
        # writer.add_scalar('Loss/val', val_loss, epoch)
        # writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        # scheduler.step()   # 跟新学习率

    save_model(model, 'model_random2.pth')
    # writer.close()

if __name__ == "__main__":
    train()
