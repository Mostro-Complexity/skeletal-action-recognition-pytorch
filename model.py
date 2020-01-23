import torch
from dataset import MSRAction3D
from torch.utils.data.dataloader import DataLoader
from torch.nn import Conv2d, MaxPool2d, Linear, BatchNorm2d, Dropout2d
from torch.nn import CrossEntropyLoss, BatchNorm1d
from torch.nn.functional import dropout2d, pad
from torch.nn.functional import relu, softmax
from torch.optim import SGD
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal
import numpy as np
from abc import ABCMeta, abstractmethod
from tqdm import tqdm


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
    def __init__(self, predict_mode=False, batch_size=32):
        self.predict_mode = predict_mode
        self.batch_size = batch_size
        self.train_loader = None
        self.test_loader = None
        self.n_steps = 0

    def set_dataloader(self, method, data_loader):
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
    def train_module(self, epochs, l2_regular_lambda=1e-2):
        pass

    @abstractmethod
    def evaluate(self, data_loader):
        pass

    @abstractmethod
    def predict(self):
        pass


class ActionConvNet(torch.nn.Module, TrainableModule):
    def __init__(self, input_isize=(60, 60, 3), predict_mode=False):
        torch.nn.Module.__init__(self)
        TrainableModule.__init__(self, predict_mode=predict_mode)

        self.input_isize = input_isize
        self.n_hidden = 256
        self.conv1 = Conv2d(in_channels=input_isize[2],
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
        self.fc1 = Linear(in_features=256, out_features=self.n_hidden)
        self.fc2 = Linear(in_features=self.n_hidden, out_features=20)

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
        x = x.view(-1, self.get_hidden_size(x))
        x = relu(self.fc1(x))
        if self.predict_mode:
            x = self.fc2(x)  # TODO:正则化？
        return x

    def get_hidden_size(self, x):
        size = np.array(x.size())
        n_features = np.prod(size[1:])

        return n_features

    def train_module(self, epochs, l2_regular_lambda=1e-2):
        assert self.predict_mode, "You shouldn't use this method in predict mode"

        for epoch in range(epochs):
            self.train(True)
            train_loss = 0.0
            n_correct_preds, n_total_preds = 0, 0
            train_loader = tqdm(self.train_loader, ncols=90, unit='batchs')
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                categorical_outputs = self(inputs)
                outputs = torch.argmax(categorical_outputs, 1)

                loss = self.criterion(categorical_outputs, labels)
                loss += l2_regular_lambda*(torch.norm(self.fc1.weight, p=2) +
                                           torch.norm(self.fc2.weight, p=2))
                loss.backward()
                self.optimizer.step()

                # print statistics
                train_loss += loss.item()
                n_correct_preds += torch.sum(labels == outputs).item()
                n_total_preds += labels.numel()
                total_accuracy = n_correct_preds / n_total_preds

                if i % self.n_steps == self.n_steps-1:    # print every 2000 mini-batches
                    self.show_statistics(
                        train_loader, 'train', epoch, train_loss, total_accuracy)
                    train_loss = 0.0

            self.show_statistics(train_loader, 'both',
                                 epoch, train_loss, total_accuracy)

        print('Finished Training')

    def show_statistics(self, train_loader, method, epoch, loss, accuracy):
        tr_info = 'epoch: %d loss: %.3f acc: %.3f' % (
            epoch + 1,
            loss / self.n_steps,
            accuracy)
        train_loader.set_description(tr_info)
        if method == 'both':
            val_loss, val_acc = self.evaluate(self.test_loader)
            val_info = 'val_loss: %.3f val_acc: %.3f' % (
                val_loss, val_acc
            )
            print(val_info)

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
        torch.save(self, path)

    def load_module(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint)


class MultiModalFusion(torch.nn.Module, TrainableModule):
    def __init__(self, path=None):
        torch.nn.Module.__init__(self)
        TrainableModule.__init__(self)
        if path is not None:
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint)
        self.n_rank = 1
        self.n_outputs = 20
        self.convnet1 = ActionConvNet(predict_mode=True)
        self.convnet2 = ActionConvNet(predict_mode=True)
        self.convnet3 = ActionConvNet(input_isize=(60, 60, 1),
                                      predict_mode=True)

        self.convnet1.set_criterion(CrossEntropyLoss())
        self.convnet1.set_optimizer(SGD(self.convnet1.parameters(),
                                        lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True))
        self.convnet2.set_criterion(CrossEntropyLoss())
        self.convnet2.set_optimizer(SGD(self.convnet2.parameters(),
                                        lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True))
        self.convnet3.set_criterion(CrossEntropyLoss())
        self.convnet3.set_optimizer(SGD(self.convnet3.parameters(),
                                        lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True))

        n_hidden_1 = self.convnet1.n_hidden
        n_hidden_2 = self.convnet2.n_hidden
        n_hidden_3 = self.convnet3.n_hidden

        self.bn1 = BatchNorm1d(n_hidden_1, eps=1e-3, momentum=0.99,
                               track_running_stats=False)
        self.bn2 = BatchNorm1d(n_hidden_2, eps=1e-3, momentum=0.99,
                               track_running_stats=False)
        self.bn3 = BatchNorm1d(n_hidden_3, eps=1e-3, momentum=0.99,
                               track_running_stats=False)

        # abs_pos_factor = Parameter(
        #     torch.FloatTensor(self.n_rank, n_hidden_1, self.n_outputs))
        # lp_ang_factor = Parameter(
        #     torch.FloatTensor(self.n_rank, n_hidden_2, self.n_outputs))
        # ll_ang_factor = Parameter(
        #     torch.FloatTensor(self.n_rank, n_hidden_3, self.n_outputs))
        self.fc1 = Linear(256, 256, bias=False)
        self.fc2 = Linear(256, 20)

        # fusion_weights = Parameter(torch.FloatTensor(1, self.n_rank))
        # fusion_bias = Parameter(torch.FloatTensor(1, self.n_outputs))

        # self.abs_pos_factor = xavier_normal(abs_pos_factor)
        # self.lp_ang_factor = xavier_normal(lp_ang_factor)
        # self.ll_ang_factor = xavier_normal(ll_ang_factor)
        # self.fusion_weights = xavier_normal(fusion_weights)
        # self.fusion_bias = xavier_normal(fusion_bias)

    def forward(self, abs_pos, lp_ang, ll_ang):
        self.convnet1.predict_mode = False
        abs_pos = self.convnet1(abs_pos)
        self.convnet2.predict_mode = False
        lp_ang = self.convnet2(lp_ang)
        self.convnet3.predict_mode = False
        ll_ang = self.convnet3(ll_ang)

        # abs_pos = torch.matmul(abs_pos, self.abs_pos_factor)
        # lp_ang = torch.matmul(lp_ang, self.lp_ang_factor)
        # ll_ang = torch.matmul(ll_ang, self.ll_ang_factor)

        fusion_output = abs_pos*lp_ang*ll_ang
        # fusion_output = fusion_output.permute(1, 0, 2)
        # fusion_output = torch.matmul(self.fusion_weights, fusion_output)
        # fusion_output = fusion_output.squeeze()+self.fusion_bias
        # fusion_output = fusion_output.view(-1, self.n_outputs)
        fusion_output = self.fc2(fusion_output)
        return fusion_output

    def train_module(self, epochs, l2_regular_lambda=1e-2):
        self.convnet1.train_module(epochs=100)
        self.convnet2.train_module(epochs=100)
        self.convnet3.train_module(epochs=100)
        self.convnet1.predict_mode = False
        self.convnet2.predict_mode = False
        self.convnet3.predict_mode = False
        print("Start training fusion model.")
        for epoch in range(epochs):
            self.train(True)
            train_loss = 0.0
            n_correct_preds, n_total_preds = 0, 0
            train_loader = tqdm(self.train_loader, ncols=90, unit='batchs')
            for i, data in enumerate(train_loader, 0):
                abs_pos, lp_ang, ll_ang, labels = data
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                categorical_outputs = self(abs_pos, lp_ang, ll_ang)
                outputs = torch.argmax(categorical_outputs, 1)

                loss = self.criterion(categorical_outputs, labels)
                loss += l2_regular_lambda * \
                    torch.norm(self.fusion_weights, p=2)
                loss.backward()
                self.optimizer.step()

                # print statistics
                train_loss += loss.item()
                n_correct_preds += torch.sum(labels == outputs).item()
                n_total_preds += labels.numel()
                total_accuracy = n_correct_preds / n_total_preds

                if i % self.n_steps == self.n_steps-1:    # print every 2000 mini-batches
                    self.show_statistics(
                        train_loader, 'train', epoch, train_loss, total_accuracy)
                    train_loss = 0.0

            self.show_statistics(train_loader, 'both',
                                 epoch, train_loss, total_accuracy)

        print('Finished training')

    def show_statistics(self, train_loader, method, epoch, loss, accuracy):
        tr_info = 'epoch: %d loss: %.3f acc: %.3f' % (
            epoch + 1,
            loss / self.n_steps,
            accuracy)
        train_loader.set_description(tr_info)
        if method == 'both':
            val_loss, val_acc = self.evaluate(self.test_loader)
            val_info = 'val_loss: %.3f val_acc: %.3f' % (
                val_loss, val_acc
            )
            print(val_info)

    def evaluate(self, data_loader):
        self.eval()
        val_loss = 0
        n_correct_val, n_total_val = 0, 0
        n_steps = 1+len(data_loader)//self.batch_size
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                abs_pos, lp_ang, ll_ang, labels = data
                categorical_outputs = self(abs_pos, lp_ang, ll_ang)
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
        torch.save(self.state_dict(), path)

    def load_module(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint)


if __name__ == "__main__":
    pass
