from data.traffic_dataset import TrafficDataset
from utils.metrics import Metric
import numpy as np
import h5py
import torch


def test(model, criterion, device, num_nodes, test_data, test_loader):
    model.eval()  # 打开测试模式
    with torch.no_grad():  # 关闭梯度
        MAE, MAPE, RMSE = [], [], []  # 定义三种指标的列表
        Target = np.zeros([num_nodes, 0, 1])  # [N, T, D],T=1 ＃ 目标数据的维度，用０填充
        Predict = np.zeros_like(Target)  # [N, T, D],T=1 # 预测数据的维度

        total_loss = 0.0
        for data in test_loader:  # 一次把一个batch的测试数据取出来

            # 下面得到的预测结果实际上是归一化的结果，有一个问题是我们这里使用的三种评价标准以及可视化结果要用的是逆归一化的数据
            adj = data['graph'].to(device)
            flow_x = data['flow_x'].to(device)
            B, N, _, _ = flow_x.size()
            flow_x = flow_x.view(B, N, -1)

            model.zero_grad()
            predict_value = model(flow_x, adj).to(torch.device("cpu"))  # [B, N, 1, D],由于标签flow_y在cpu中，所以最后的预测值要放回到cpu中
            loss = criterion(predict_value, data["flow_y"])  # 计算损失，切记这个loss不是标量
            total_loss += loss.item()  # 所有的batch的loss累加
            # 下面实际上是把预测值和目标值的batch放到第二维的时间维度，这是因为在测试数据的时候对样本没有shuffle，
            # 所以每一个batch取出来的数据就是按时间顺序来的，因此放到第二维来表示时间是合理的.
            predict_value = predict_value.transpose(0, 2).squeeze(0)  # [1, N, B(T), D] -> [N, B(T), D] -> [N, T, D]
            target_value = data["flow_y"].transpose(0, 2).squeeze(0)  # [1, N, B(T), D] -> [N, B(T), D] -> [N, T, D]

            performance, data_to_save = compute_performance(predict_value, target_value, test_loader)  # 计算模型的性能，返回评价结果和恢复好的数据

            # 下面这个是每一个batch取出的数据，按batch这个维度进行串联，最后就得到了整个时间的数据，也就是
            # [N, T, D] = [N, T1+T2+..., D]
            Predict = np.concatenate([Predict, data_to_save[0]], axis=1)
            Target = np.concatenate([Target, data_to_save[1]], axis=1)

            MAE.append(performance[0])
            MAPE.append(performance[1])
            RMSE.append(performance[2])

        print("Test Loss: {:02.4f}".format(1000 * total_loss / len(test_data)))

    # 三种指标取平均
    print("Performance:  MAE {:2.2f}    {:2.2f}%    {:2.2f}".format(np.mean(MAE), np.mean(MAPE * 100), np.mean(RMSE)))

    result_file = "GAT_result.h5"
    file_obj = h5py.File(result_file, "w")  # 将预测值和目标值保存到文件中，因为要多次可视化看看结果

    file_obj["predict"] = Predict  # [N, T, D]
    file_obj["target"] = Target  # [N, T, D]


def compute_performance(prediction, target, data):  # 计算模型性能
    # 下面的try和except实际上在做这样一件事：当训练+测试模型的时候，数据肯定是经过dataloader的，所以直接赋值就可以了
    # 但是如果将训练好的模型保存下来，然后测试，那么数据就没有经过dataloader，是dataloader型的，需要转换成dataset型。
    try:
        dataset = data.dataset  # 数据为dataloader型，通过它下面的属性.dataset类变成dataset型数据
    except:
        dataset = data  # 数据为dataset型，直接赋值

    # 下面就是对预测和目标数据进行逆归一化，recover_data()函数在上一小节的数据处理中
    prediction = TrafficDataset.inverse_normalize_data(dataset.data_norm_base[0], dataset.data_norm_base[1], prediction.numpy())
    target = TrafficDataset.inverse_normalize_data(dataset.data_norm_base[0], dataset.data_norm_base[1], target.numpy())

    # 对三种评价指标写了一个类，这个类封装在另一个文件中，在后面
    mae, mape, rmse = Metric.total(target.reshape(-1), prediction.reshape(-1))  # 变成常向量才能计算这三种指标

    performance = [mae, mape, rmse]
    recovered_data = [prediction, target]

    return performance, recovered_data  # 返回评价结果，以及恢复好的数据（为可视化准备的）
