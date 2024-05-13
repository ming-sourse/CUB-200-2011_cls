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
    new_layer_params = model.fc.parameters()
    base_params = [p for p in model.parameters()][:-2]

    # 为新层和预训练层设置不同的学习率
    optimizer = SGD([
        {'params': base_params, 'lr': 0.001},  
        {'params': new_layer_params, 'lr': 0.01}  
    ], momentum=0.9, weight_decay=0.0001)      

    return optimizer

def train():
    # 设置设备
    device = setup_device()

    # 数据加载
    data_loaders = load_data(split_train=True)
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']

    # 模型加载
    model = get_model()
    model.to(device)

    # 定义优化器和损失函数
    # optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    optimizer = setup_optimizer(model)
    criterion = nn.CrossEntropyLoss()
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # 初始化TensorBoard
    # writer = SummaryWriter()

    # 训练和验证循环
    writer_train = SummaryWriter('runs/Training_1')
    writer_val = SummaryWriter('runs/Validation_1')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # print("Input shape:", images.shape)
            outputs = model(images)
            # print("Outputs shape:", outputs.shape)
            # print("Labels shape:", labels.shape)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()   
            loss.backward()         
            optimizer.step()        

            train_loss += loss.item() * images.size(0)    
        
        # 平均训练损失
        train_loss /= len(train_loader.dataset)
        writer_train.add_scalar('Loss1', train_loss, epoch)    

        # 验证过程
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():   
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)   
                total += labels.size(0)    
                correct += (predicted == labels).sum().item()   

        val_loss /= len(val_loader.dataset)
        val_accuracy = 100 * correct / total
        writer_val.add_scalar('Loss1', val_loss, epoch)
        writer_val.add_scalar('Accuracy1/val', val_accuracy, epoch)
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        # scheduler.step()   

    save_model(model, 'model_4.pth')
    writer_train.close()
    writer_val.close()
    # writer.close()

if __name__ == "__main__":
    train()
