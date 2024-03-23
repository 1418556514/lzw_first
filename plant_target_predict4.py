 # coding=UTF-8
#!/usr/local/bin/python
import os, sys
import torchvision
import pylab
import torch
from matplotlib import pyplot as plt
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import os
from torchvision import transforms, models, datasets
import pandas as pd
import random
import warnings
from sklearn.metrics import roc_auc_score
import time

warnings.filterwarnings('ignore')


def trans_nt(seq):
    """对序列进行编码"""
    # CG对越多热稳定性越高。因为CG对互相形成三个氢键，更加稳定
    nt_dic = {'A': 1.0, 'C': 2.0, 'G': -2.0, 'T': -1.0, 'U': -1.0}
    nt_l = []
    for i in seq:
        nt_l.append(nt_dic[i])

    return nt_l


def get_matrix(test):
    """将序列转化为图像矩阵"""

    # 两个C(chaneel),顺序CHW。最大的miRNA长度24，设置30补0；最大的gene_seq长度7000，设置8000补0
    miRNA_encode = trans_nt(test['miRNA_seq'])
    for i in range(30 - len(miRNA_encode)):
        miRNA_encode.append(0)
    c1_value = np.array([miRNA_encode for i in range(8000)]).T

    gene_encode = trans_nt(test['gene_seq'])
    for i in range(8000 - len(gene_encode)):
        gene_encode.append(0)
    c2_value = np.array([gene_encode for i in range(30)])
    c3_value = ((c1_value + c2_value) == 0)
    res = np.array([c1_value, c2_value, c3_value])

    return res


class MyDataset(torch.utils.data.Dataset):
    """构建数据集"""

    def __init__(self, ori_data):
        self.ori_data = ori_data
        # self.req_type=req_type

    def __getitem__(self, item):
        # 改成存了再读取速度会更快，因为从利用率上看CPU要远高于GPU，每次迭代取数处理耗时较久(但是占用空间太大，放弃，一条数据3.5MB)
        # return torch.load(f'./test/{self.ori_data.iloc[item]["match_id"]}.pth'), self.ori_data.iloc[item]['result']
        # if self.req_type=='train':
        return torch.from_numpy(get_matrix(self.ori_data.iloc[item])), self.ori_data.iloc[item]['result']
        # else:
        #     return

    def __len__(self):
        return self.ori_data.shape[0]


def get_resnet_model(input_channel, out_size):
    """构建restnet网络"""
    # 使用resnet网络
    network = models.resnet18()
    # 改第一层
    network.conv1 = torch.nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # 获取全连接层输入特征
    num_ftrs = network.fc.in_features
    # 从新加全连接层，从新设置，输出1
    network.fc = torch.nn.Sequential(torch.nn.Linear(num_ftrs, out_size),
                                     torch.nn.Sigmoid())  # dim=0表示对列运算（1是对行运算），且元素和为1；

    return network


def get_acc(predict_res, labels_res):
    """返回准确率等指标"""
    TP, TN, FP, FN = 0, 0, 0, 0
    predict_res = predict_res.squeeze()
    for i in range(predict_res.shape[0]):
        if (predict_res[i] >= 0.5) and (labels_res[i] == 1):
            TP += 1
        elif (predict_res[i] < 0.5) and (labels_res[i] == 0):
            TN += 1
        elif (predict_res[i] >= 0.5) and (labels_res[i] == 0):
            FP += 1
        else:
            FN += 1
    #
    # return  [TP,TN,FP,FN]
    return [roc_auc_score(labels_res.cpu().detach().numpy(), predict_res.cpu().detach().numpy()),
            {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}]


def model_test(test_data_loaders, test_model):
    """进行模型测试"""

    with torch.no_grad():
        # tem是每隔20个batch输出此段准确率
        # s_tem, t_tem = 0, 0
        # s0, t0 = 0, 0
        predict_all, labels_all = torch.tensor([]), torch.tensor([])
        for i, data in enumerate(test_data_loaders, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = test_model(inputs.float()).squeeze()
            predict_all = torch.concat([predict_all, outputs.cpu()], 0)
            labels_all = torch.concat([labels_all, labels.cpu()], 0)

    return predict_all, labels_all


def model_train(ori_model, epoch_num, optimizer, data_loaders, criterion, dtest_data_loaders=None,
                extra_model_name='common'):
    """训练模型"""

    # 分别存储每个epoch的验证集指标结果，模型，训练loss,测试loss
    model_dic, train_acc_dic, test_acc_dic, train_loss_dic, test_loss_dic = {}, {}, {}, {}, {}
    for epoch in range(epoch_num):
        print(f'epoch-start-{epoch}:{time.asctime()}')
        running_loss = 0.0
        for i, data in enumerate(data_loaders, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # 清空梯度
            optimizer.zero_grad()
            # outputs=1 if tp_resnet_model(inputs)>=0.5 else 0
            outputs = ori_model(inputs.float())
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # print(get_acc(outputs, labels))
            # print(f'loss:{loss}')
            # 训练过程的显示
            if i % 50 == 49:
                print(get_acc(outputs, labels))
                print('[%d, %5d] loss: %.3f' %
                      (epoch, i + 1, running_loss / 50))
                running_loss = 0.0
        print(f'epoch-train-end-{epoch}:{time.asctime()}')

        # 保存每个epoch训练的模型、训练结果和测试结果
        torch.save(ori_model.state_dict(), f'./models/{extra_model_name}_{epoch}_model.PTH')
        model_dic[epoch] = ori_model

        if epoch > 8:
            predict_train_tem, labels_train_tem = model_test(data_loaders, ori_model)
            torch.save(predict_train_tem, f'./predicted/{extra_model_name}_{epoch}_train_predictions.pth')
            torch.save(labels_train_tem, f'./predicted/{extra_model_name}_{epoch}_train_labels.pth')

            train_acc_dic[epoch] = get_acc(predict_train_tem, labels_train_tem)
            train_loss_dic[epoch] = criterion(predict_train_tem, labels_train_tem.float()).item()
            print('*' * 50)
            print(train_acc_dic)
            print(train_loss_dic)
        if dtest_data_loaders:
            predict_test_tem, labels_test_tem = model_test(dtest_data_loaders, ori_model)
            torch.save(predict_test_tem, f'./predicted/{extra_model_name}_{epoch}_predictions.pth')
            torch.save(labels_test_tem, f'./predicted/{extra_model_name}_{epoch}_labels.pth')
            test_acc_dic[epoch] = get_acc(predict_test_tem, labels_test_tem)
            test_loss_dic[epoch] = criterion(predict_test_tem, labels_test_tem.float()).item()
            print(test_acc_dic)
            print(test_loss_dic)

    return model_dic, train_acc_dic, train_loss_dic, test_acc_dic, test_loss_dic


# 这里使用的阳性数据剔除了与测试集重复的数据
more_data_p = pd.read_csv('contrast_p_train_data.csv')
more_data_n = pd.read_csv('contrast_n_train_0.05_data.csv')
need_col = ['miRNA_seq', 'gene_seq', 'result']
train_data = pd.concat([more_data_p[need_col], more_data_n[need_col]]).reset_index(drop=True)

# 构建可迭代的数据装载器
train_data_loaders = torch.utils.data.DataLoader(MyDataset(train_data), batch_size=16, shuffle=True)

# 构建resnet网络
tp_resnet_model = get_resnet_model(3, 1)

# 指定设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tp_resnet_model = tp_resnet_model.to(device)
# 实例化损失函数类
train_criterion = torch.nn.BCELoss()
# 设置优化器
ori_optimizer = torch.optim.Adam(tp_resnet_model.parameters(), lr=0.01)
# 设置每3轮学习率减半
train_optimizer = torch.optim.lr_scheduler.StepLR(ori_optimizer, step_size=3, gamma=0.5)

name = 'contrast'
# name='tt0'
model_dic, train_acc_dic, train_loss_dic, test_acc_dic, test_loss_dic = \
    model_train(tp_resnet_model, 8, ori_optimizer, train_data_loaders, train_criterion, extra_model_name=name)
torch.save(model_dic, f'./dicts/{name}_model_dic.pth')
torch.save(train_acc_dic, f'./dicts/{name}_train_acc_dic.pth')
torch.save(train_loss_dic, f'./dicts/{name}_train_loss_dic.pth')
torch.save(test_acc_dic, f'./dicts/{name}_test_acc_dic.pth')
torch.save(test_loss_dic, f'./dicts/{name}_test_loss_dic.pth')
