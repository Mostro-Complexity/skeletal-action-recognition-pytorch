import torch
from dataset import MSRAction3D
from torch.utils.data.dataloader import DataLoader
from torch.nn import Conv2d, MaxPool2d, Linear, BatchNorm2d, Dropout2d
from torch.nn import CrossEntropyLoss
from torch.nn.functional import dropout2d, pad
from torch.nn.functional import relu, softmax
from torch.optim import SGD
import numpy as np
from abc import ABCMeta, abstractmethod


def same_padding(input, kernel_size, stride=None, dilation=1):
    if type(kernel_size) is int:
        kernel_size = (kernel_size, kernel_size)
    if type(stride) is int:
        stride = (stride, stride)
    if type(dilation) is int:
        dilation = (dilation, dilation)

    input_rows = input.size(0)
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                       (kernel_size[0] - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)

    input_cols = input.size(1)
    out_cols = (input_cols + stride[1] - 1) // stride[1]
    padding_cols = max(0, (out_cols - 1) * stride[1] +
                       (kernel_size[1] - 1) * dilation[1] + 1 - input_cols)
    cols_odd = (padding_cols % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])
    input = pad(input, [padding_rows // 2, padding_rows // 2,
                        padding_cols // 2, padding_cols // 2])
    return input


class TrainableModule(metaclass=ABCMeta):
    def __init__(self, training=False, batch_size=32):
        self.training = training
        self.batch_size = batch_size
        self.train_loader = None
        self.test_loader = None
        self.n_steps = 0

    def set_dataloder(self, method, data_loader):
        if method == 'train':
            self.train_loader = data_loader
            self.n_steps = 1+len(data_loader)//self.batch_size
        elif method == 'test' or method == 'validate':
            self.test_loader = data_loader

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_criterion(self, criterion):
        self.criterion = criterion

    @abstractmethod
    def save_module(self, path):
        pass

    @abstractmethod
    def train_module(self, epochs):
        pass

    @abstractmethod
    def evaluate(self, data_loader):
        pass

    @abstractmethod
    def predict(self):
        pass


class ActionConvNet(torch.nn.Module, TrainableModule):
    def __init__(self, training=False):
        torch.nn.Module.__init__(self)
        TrainableModule.__init__(self, training=training)

        self.conv1 = Conv2d(in_channels=3,
                            out_channels=32,
                            kernel_size=3,
                            stride=1)
        self.max_pool1 = MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = Conv2d(in_channels=32,
                            out_channels=32,
                            kernel_size=3,
                            stride=1)
        self.max_pool2 = MaxPool2d(kernel_size=3, stride=2)
        self.bn1 = BatchNorm2d(32, eps=1e-3, momentum=0.99,
                               track_running_stats=False)
        # track_running_stats=False is better
        self.dropout1 = Dropout2d(0.5)

        self.conv3 = Conv2d(in_channels=32,
                            out_channels=64,
                            kernel_size=3,
                            stride=1)
        self.max_pool3 = MaxPool2d(kernel_size=3, stride=2)

        self.conv4 = Conv2d(in_channels=64,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,)
        self.max_pool4 = MaxPool2d(kernel_size=3, stride=2)

        self.bn2 = BatchNorm2d(64, eps=1e-3, momentum=0.99,
                               track_running_stats=False)
        self.dropout2 = Dropout2d(0.5)
        self.fc1 = Linear(in_features=256, out_features=256)
        self.fc2 = Linear(in_features=256, out_features=20)

    def forward(self, x):
        # input, kernal_size, padding
        x = same_padding(x, kernel_size=3, stride=1)
        x = relu(self.conv1(x))
        x = self.max_pool1(x)
        x = same_padding(x, kernel_size=3, stride=1)
        x = relu(self.conv2(x))
        x = self.max_pool2(x)

        x = self.bn1(x)
        x = self.dropout1(x)

        x = same_padding(x, kernel_size=3, stride=1)
        x = relu(self.conv3(x))
        x = self.max_pool3(x)
        x = same_padding(x, kernel_size=3, stride=1)
        x = relu(self.conv4(x))
        x = self.max_pool4(x)

        x = self.bn2(x)
        x = self.dropout2(x)

        x = x.view(-1, self.get_n_features(x))
        x = relu(self.fc1(x))
        x = self.fc2(x)  # TODO:正则化？
        return x

    def get_n_features(self, x):
        size = np.array(x.size())
        n_features = np.prod(size[1:])

        return n_features

    def train_module(self, epochs):
        for epoch in range(epochs):
            self.train(True)
            running_loss = 0.0
            n_correct_preds, n_total_preds = 0, 0
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                categorical_outputs = self(inputs)
                outputs = torch.argmax(categorical_outputs, 1)

                loss = self.criterion(categorical_outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                n_correct_preds += torch.sum(labels == outputs).item()
                n_total_preds += labels.numel()
                total_accuracy = n_correct_preds / n_total_preds

                if i % self.n_steps == self.n_steps-1:    # print every 2000 mini-batches
                    self.show_statistics(
                        'train', epoch, running_loss, total_accuracy)
                    running_loss = 0.0

            self.show_statistics('both', epoch, running_loss, total_accuracy)

        print('Finished Training')

    def show_statistics(self, method, epoch, loss, accuracy):
        tr_info = 'epoch: %d loss: %.3f acc: %.3f' % (
            epoch + 1,
            loss / self.n_steps,
            accuracy)
        if method == 'train':
            print(tr_info)
        elif method == 'both':
            val_loss, val_acc = self.evaluate(self.test_loader)
            val_info = 'val_loss: %.3f val_acc: %.3f' % (
                val_loss, val_acc
            )
            print(' '.join([tr_info, val_info]))

    def evaluate(self, data_loader):
        self.eval()
        val_loss = 0
        n_correct_val, n_total_val = 0, 0
        n_steps = 1+len(data_loader)//self.batch_size
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inputs, labels = data
                categorical_outputs = self(inputs)
                outputs = torch.argmax(categorical_outputs, 1)

                loss = self.criterion(categorical_outputs, labels)
                val_loss += loss.item()
                n_correct_val += torch.sum(labels == outputs).item()
                n_total_val += labels.numel()

            total_accuracy = n_correct_val/n_total_val
        return val_loss / len(data_loader), total_accuracy

    def predict(self):
        pass

    def save_module(self, path):
        pass


if __name__ == "__main__":
    batch_size = 32

    train_data = MSRAction3D(root='data/MSRAction3D',
                             method='train', resize_isize=(52, 52, 3))
    test_data = MSRAction3D(root='data/MSRAction3D',
                            method='test', resize_isize=(52, 52, 3))

    train_loader = DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size)

    convnet = ActionConvNet(training=True)
    print(convnet)

    convnet.set_dataloder('train', train_loader)
    convnet.set_dataloder('test', test_loader)
    convnet.set_criterion(CrossEntropyLoss())
    convnet.set_optimizer(SGD(convnet.parameters(),
                              lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True))
    convnet.train_module(epochs=100)
