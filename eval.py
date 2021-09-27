# -*- coding: utf-8 -*-
"""
@Time   : 2020/6/4

@Author : Shen Fang
"""
import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from data.traffic_dataset import TrafficDataset
from model.gat import GAT
from test import test


num_nodes = 307
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def eval():
    # Loading Dataset
    test_data = TrafficDataset(data_path=["dataset/PeMS_04/PeMS04.csv", "dataset/PeMS_04/PeMS04.npz"], num_nodes=num_nodes, divide_days=[45, 14],
                               time_interval=5, history_length=6,
                               train_mode="test")

    test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=32)

    # Loading Model
    my_net = GAT(nfeat=6, nhid=6, nclass=1, dropout=0., alpha=0.1, nheads=2)
    my_net.load_state_dict(torch.load('checkpoints/1.pth'))
    my_net = my_net.to(device)
    criterion = nn.MSELoss()

    # Test Model
    # visualizing results; evaluating model using MAE, MAPE, and RMSE.
    test(my_net, criterion, device, num_nodes, test_data, test_loader)

if __name__ == '__main__':
    eval()
