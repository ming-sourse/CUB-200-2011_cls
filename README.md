# CUB-200-2011_cls
## 数据
数据的路径可以在config中修改（CUB-200-2011中images文件夹所在的目录），由于elearning上提供的链接下载太慢，所以我是从kaggle上下的数据，链接为：https://www.kaggle.com/datasets/veeralakrishna/200-bird-species-with-11788-images。应该是和原数据集一样的，如有问题可以在链接中下载。
## 训练
使用预训练参数初始化的模型，请用train.py进行训练。  
使用随机初始化参数的模型，请用train_random.py进行训练。  
训练后的模型权重保存在代码的同一目录下。 
## 测试
请使用test.py文件进行测试，对于不同的模型权重需要先修改模型权重名称，再进行测试。
