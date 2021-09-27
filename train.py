# -*- coding: utf-8 -*-
"""
@Time   : 2020/6/4

@Author : Shen Fang
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from data.traffic_dataset import TrafficDataset
from model.gat import GAT
# from test import test


num_epochs = 50
num_nodes = 307
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def train():
    # Loading Dataset
    train_data = TrafficDataset(data_path=["dataset/PeMS_04/PeMS04.csv", "dataset/PeMS_04/PeMS04.npz"], num_nodes=num_nodes, divide_days=[45, 14],
                                time_interval=5, history_length=6,
                                train_mode="train")

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=32)

    test_data = TrafficDataset(data_path=["dataset/PeMS_04/PeMS04.csv", "dataset/PeMS_04/PeMS04.npz"], num_nodes=num_nodes, divide_days=[45, 14],
                               time_interval=5, history_length=6,
                               train_mode="test")

    test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=32)

    # Loading Model
    my_net = GAT(nfeat=6, nhid=6, nclass=1, dropout=0., alpha=0.1, nheads=2)

    my_net = my_net.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=my_net.parameters())

    # Train model
    my_net.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        start_time = time.time()
        for data in train_loader:  # ["graph": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]]
            adj = data['graph'].to(device)
            flow_x = data['flow_x'].to(device)
            B, N, _, _ = flow_x.size()
            flow_x = flow_x.view(B, N, -1)

            my_net.zero_grad()
            predict_value = my_net(flow_x, adj).to(torch.device("cpu"))  # [B, N, 1, D],由于标签flow_y在cpu中，所以最后的预测值要放回到cpu中
            loss = criterion(predict_value, data["flow_y"])  # 计算损失，切记这个loss不是标量
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        end_time = time.time()

        print("num_epochs: {:04d}, Loss: {:02.4f}, Time: {:02.2f} mins".format(epoch, 1000 * epoch_loss / len(train_data),
                                                                          (end_time-start_time)/60))
        torch.save(my_net.state_dict(), "checkpoints/1.pth")

    # Test Model
    # visualizing results; evaluating model using MAE, MAPE, and RMSE.
    # test(my_net, criterion, device, num_nodes, test_data, test_loader)

if __name__ == '__main__':
    train()
