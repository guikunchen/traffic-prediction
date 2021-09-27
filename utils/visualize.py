import matplotlib.pyplot as plt
import numpy as np
import h5py


def visualize_result(h5_file, nodes_id, time_se, visualize_file):
    file_obj = h5py.File(h5_file, "r") # 获得文件对象，这个文件对象有两个keys："predict"和"target"
    prediction = file_obj["predict"][:][:, :, 0]  # [N, T],切片，最后一维取第0列，所以变成二维了，要是[:, :, :1]那么维度不会缩减
    target = file_obj["target"][:][:, :, 0]  # [N, T],同上
    file_obj.close()

    plot_prediction = prediction[nodes_id][time_se[0]: time_se[1]]  # [T1]，将指定节点的，指定时间的数据拿出来
    plot_target = target[nodes_id][time_se[0]: time_se[1]]  # [T1]，同上

    plt.figure()
    plt.grid(True, linestyle="-.", linewidth=0.5)
    plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), plot_prediction, ls="-", marker=" ", color="r")
    plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), plot_target, ls="-", marker=" ", color="b")

    plt.legend(["prediction", "target"], loc="upper right")

    plt.axis([0, time_se[1] - time_se[0],
              np.min(np.array([np.min(plot_prediction), np.min(plot_target)])),
              np.max(np.array([np.max(plot_prediction), np.max(plot_target)]))])

    plt.savefig(visualize_file + ".png")


def visualize_dataset():
    def get_flow(file_name):
        flow_data = np.load(file_name)
        flow_data = flow_data['data'].transpose([1, 0, 2])[:, :, 0][:, :, np.newaxis]  # [N, T, D]  D = 1

    traffic_data = get_flow("PeMS_04/PeMS04.npz")
    node_id = 226
    print(traffic_data.shape)

    plt.plot(traffic_data[: 24 * 12, node_id, 0])
    plt.savefig("node_{:3d}_1.png".format(node_id))

    plt.plot(traffic_data[: 24 * 12, node_id, 1])
    plt.savefig("node_{:3d}_2.png".format(node_id))

    plt.plot(traffic_data[: 24 * 12, node_id, 2])
    plt.savefig("node_{:3d}_3.png".format(node_id))

if __name__ == '__main__':
    # 可视化，在下面的 Evaluation()类中，这里是对应的GAT算法运行的结果，进行可视化\
    visualize_result(h5_file="/home/cgk/projects/graph-cnn/GAT_result.h5",
    nodes_id = 226, time_se = [0, 24 * 12 * 2],  # 是节点的时间范围
    visualize_file = "gat_node_120")