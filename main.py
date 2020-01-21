from dataset import MSRAction3D
from torch.utils.data.dataloader import DataLoader
from torch.nn import CrossEntropyLoss
from model import ActionConvNet
from torch.optim import SGD


def training_single_convnet():
    batch_size = 32

    train_data = MSRAction3D(root='data/MSRAction3D',
                             method='train', resize_isize=(52, 52, 3))
    test_data = MSRAction3D(root='data/MSRAction3D',
                            method='test', resize_isize=(52, 52, 3))

    train_loader = DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size)

    convnet = ActionConvNet()
    print(convnet)

    convnet.set_dataloder('train', train_loader)
    convnet.set_dataloder('test', test_loader)
    convnet.set_criterion(CrossEntropyLoss())
    convnet.set_optimizer(SGD(convnet.parameters(),
                              lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True))
    convnet.train_module(epochs=100)
    convnet.save_module('model/rbg_mapping.pth')


if __name__ == "__main__":
    pass
