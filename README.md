# Traffic Prediction Based On GAT

Traffic Prediction Based On Graph Attention Networks.

# Dataset
The datasets are collected by the Caltrans Performance Measurement System (PEMS-04)

​Number of Nodes: 307 detectors

​Date: Jan to Feb in 2018 (2018.1.1——2018.2.28)

​Features: flow, occupy, speed.

Every node has three fetures while the distribution of two features remains stable.

Therefore, “flow” is the only feature we considered.

# Models
## GAT
This model is similar to https://arxiv.org/abs/1710.10903

# Reference
Parts of this project is based on https://github.com/LeronQ/GCN_predict-Pytorch