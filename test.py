import torch
from model import get_model
from utils import setup_device, load_data

def test():
    device = setup_device()

    # 加载测试数据
    data_loaders = load_data(split_train=False)
    test_loader = data_loaders['test']


    # 加载模型
    model = get_model()
    model.load_state_dict(torch.load('model_1.pth'))
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on test images: {100 * correct / total}%')

if __name__ == "__main__":
    test()
