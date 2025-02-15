import numpy as np
import torch
import random
def encoding_mask_noise_diff(num_nodes, mask_rate, difficulty, epoch, max_epoch):
    proportion = 0.5

    # 根据难度排序
    sorted_indices = torch.argsort(difficulty)

    # 计算 g(t)
    g_t = min(1, proportion + (1 - proportion) * (epoch / max_epoch))

    # 根据 g_t 计算 mask_set 的大小
    mask_set_len = int(g_t * num_nodes)
    mask_set = sorted_indices[:mask_set_len]

    # 计算需要掩码的节点数量
    num_mask_nodes = int(mask_rate * num_nodes)

    # 随机选择 mask_nodes
    perm = torch.randperm(mask_set_len)
    mask_nodes_indices = perm[:num_mask_nodes]
    mask_nodes = mask_set[mask_nodes_indices]

    # 获取未掩码节点
    mask_nodes_set = torch.isin(sorted_indices, mask_nodes)
    keep_nodes = sorted_indices[~mask_nodes_set]  # 选择不在掩码节点中的节点


    return (mask_nodes, keep_nodes)


# num_nodes = 20
# mask_rate = 0.5
# difficulty = [random.random() for _ in range(20)]
# difficulty = torch.tensor(difficulty)
# print(difficulty)
# res = encoding_mask_noise_diff(num_nodes, mask_rate, difficulty, 1, 100)

proportion = 0.3
max_epoch = 100
for epoch in range(1, max_epoch + 1):
    g_t = min(1, proportion + (1 - proportion) * (epoch / max_epoch))

    print(g_t)
