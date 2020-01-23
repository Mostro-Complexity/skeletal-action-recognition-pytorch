from dataset import MSRAction3D
from torch.utils.data.dataloader import DataLoader
from torch.nn import CrossEntropyLoss
from model import ActionConvNet, MultiModalFusion
from torch.optim import SGD
from preprocessing import MappingPipeline, MultiMappingPipeline


def training_single_convnet():
    batch_size = 32

    train_data = MSRAction3D(root='data/MSRAction3D',
                             method='train', resize_isize=(52, 52, 3))
    test_data = MSRAction3D(root='data/MSRAction3D',
                            method='test', resize_isize=(52, 52, 3))
    processed_tr_data = MappingPipeline(train_data)
    processed_te_data = MappingPipeline(test_data)

    train_loader = DataLoader(
        dataset=processed_tr_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=processed_te_data, batch_size=batch_size)

    convnet = ActionConvNet(predict_mode=True)
    print(convnet)

    convnet.set_dataloader('train', train_loader)
    convnet.set_dataloader('test', test_loader)
    convnet.set_criterion(CrossEntropyLoss())
    convnet.set_optimizer(SGD(convnet.parameters(),
                              lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True))
    convnet.train_module(epochs=100)
    convnet.save_module('model/rbg_mapping.pth')


def training_multi_convnet():
    batch_size = 32

    train_data = MSRAction3D(root='data/MSRAction3D',
                             method='train', resize_isize=(52, 52, 3))
    test_data = MSRAction3D(root='data/MSRAction3D',
                            method='test', resize_isize=(52, 52, 3))
    processed_tr_data = MultiMappingPipeline(train_data)
    processed_te_data = MultiMappingPipeline(test_data)

    train_loader = DataLoader(
        dataset=processed_tr_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=processed_te_data, batch_size=batch_size)

    multimodal = MultiModalFusion()
    print(multimodal)
    # total
    multimodal.set_dataloader('train', train_loader)
    multimodal.set_dataloader('test', test_loader)
    multimodal.set_criterion(CrossEntropyLoss())
    multimodal.set_optimizer(SGD(multimodal.parameters(),
                                 lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True))

    # subnet1
    processed_tr_data = MultiMappingPipeline(train_data, method='abs_pos')
    processed_te_data = MultiMappingPipeline(test_data, method='abs_pos')
    train_loader = DataLoader(
        dataset=processed_tr_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        dataset=processed_te_data, batch_size=batch_size, shuffle=True)
    multimodal.convnet1.set_dataloader('train', train_loader)
    multimodal.convnet1.set_dataloader('test', test_loader)

    # subnet2
    processed_tr_data = MultiMappingPipeline(train_data, method='lp_ang')
    processed_te_data = MultiMappingPipeline(test_data, method='lp_ang')
    train_loader = DataLoader(
        dataset=processed_tr_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        dataset=processed_te_data, batch_size=batch_size, shuffle=True)
    multimodal.convnet2.set_dataloader('train', train_loader)
    multimodal.convnet2.set_dataloader('test', test_loader)

    processed_tr_data = MultiMappingPipeline(train_data, method='ll_ang')
    processed_te_data = MultiMappingPipeline(test_data, method='ll_ang')
    train_loader = DataLoader(
        dataset=processed_tr_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        dataset=processed_te_data, batch_size=batch_size, shuffle=True)
    multimodal.convnet3.set_dataloader('train', train_loader)
    multimodal.convnet3.set_dataloader('test', test_loader)

    multimodal.train_module(epochs=100)
    multimodal.save_module('model/multi_modal.pth.tar')
    multimodal.convnet1.save_module('model/abs_pos.pth.tar')
    multimodal.convnet2.save_module('model/lp_ang.pth.tar')
    multimodal.convnet3.save_module('model/ll_ang.pth.tar')

def test_load_convnet():
    batch_size = 32
    test_data = MSRAction3D(root='data/MSRAction3D',
                            method='test', resize_isize=(52, 52, 3))
    processed_te_data = MultiMappingPipeline(test_data)
    test_loader = DataLoader(
        dataset=processed_te_data, batch_size=batch_size, shuffle=True)
    multimodal = MultiModalFusion(path='model/multi_modal.pth.tar')
    multimodal.set_criterion(CrossEntropyLoss())
    multimodal.set_optimizer(SGD(multimodal.parameters(),
                                 lr=0.001, momentum=0.9, weight_decay=1e-4, nesterov=True))
    loss, acc = multimodal.evaluate(test_loader)
    print('loss:%f  acc:%f' % (loss, acc))


if __name__ == "__main__":
    # training_single_convnet()
    training_multi_convnet()
    # test_load_convnet()
