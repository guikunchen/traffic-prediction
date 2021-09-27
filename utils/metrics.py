import numpy as np
import h5py


class Metric(object):
    def __init__(self):
        pass

    @staticmethod
    def mae_(target, output):
        return np.mean(np.abs(target - output))

    @staticmethod
    def mape_(target, output):
        return np.mean(np.abs(target - output) / (target + 1)) # 加1是因为target有可能为0，当然只要不太大，加几都行

    @staticmethod
    def rmse_(target, output):
        return np.sqrt(np.mean(np.power(target - output, 2)))

    @staticmethod
    def total(target, output):
        mae = Metric.mae_(target, output)
        mape = Metric.mape_(target, output)
        rmse = Metric.rmse_(target, output)

        return mae, mape, rmse


