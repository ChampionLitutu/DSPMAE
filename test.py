# 这个python的代码
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
from scipy.optimize import linear_sum_assignment

def global_contrastive_loss(Q_original, Q_masked):
    """
    计算原始图和掩码图聚类分布的对比损失
    参数：
        Q_original: 原始图的聚类分布，形状 (N, K)
        Q_masked: 掩码图的聚类分布，形状 (N, K)
    返回：
        对比损失值
    """
    # 使用 KL 散度作为对比损失
    loss = F.kl_div(F.log_softmax(Q_masked, dim=-1),
                    F.softmax(Q_original, dim=-1),
                    reduction="batchmean")
    return loss


# 定义聚类函数
def cluster_features(features, num_clusters=10):
    """
    对输入特征进行聚类
    参数：
        features: 输入特征，形状 (N, D)
        num_clusters: 聚类的簇数
    返回：
        聚类分布 Q，形状 (N, K)
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features.cpu().detach().numpy())
    cluster_assignments = kmeans.labels_  # (N,)
    print("-----------------cluster_assignments---------------")
    print(cluster_assignments)
    Q = torch.zeros(features.size(0), num_clusters, device=features.device)
    Q[torch.arange(features.size(0)), cluster_assignments] = 1  # One-hot
    return Q


# 掩码和原图对比训练
def train_contrastive(model, graph, masked_graph, optimizer, num_clusters=10):
    """
    训练函数，包含掩码图和原图的对比
    参数：
        model: GNN 模型
        graph: 原始图数据
        masked_graph: 掩码图数据
        optimizer: 优化器
        num_clusters: 聚类簇数
    返回：
        损失值
    """
    model.train()
    optimizer.zero_grad()

    # 从 GNN 中获取特征
    x_original = model(graph)  # 原始图特征 (N, D)
    x_masked = model(masked_graph)  # 掩码图特征 (N, D)

    # 对特征进行聚类
    Q_original = cluster_features(x_original, num_clusters)  # 原始图聚类分布
    Q_masked = cluster_features(x_masked, num_clusters)  # 掩码图聚类分布

    # 计算全局对比损失
    loss = global_contrastive_loss(Q_original, Q_masked)
    loss.backward()
    optimizer.step()

    return loss.item()

def align_clusters(labels1, labels2):
    """
    对齐两组聚类标签。
    参数：
        labels1: 第一次聚类标签，形状 (N,)
        labels2: 第二次聚类标签，形状 (N,)
    返回：
        对齐后的 labels2
    """
    num_clusters = max(labels1.max(), labels2.max()) + 1
    cost_matrix = np.zeros((num_clusters, num_clusters), dtype=np.int64)

    for i in range(num_clusters):
        for j in range(num_clusters):
            cost_matrix[i, j] = -np.sum((labels1 == i) & (labels2 == j))  # 交集数量，取负作为代价

    # 使用匈牙利算法求解最佳匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 构建映射表并重新分配标签
    label_mapping = {col: row for row, col in zip(row_ind, col_ind)}
    aligned_labels2 = np.array([label_mapping[label] for label in labels2])
    print(aligned_labels2)
    return aligned_labels2
# 示例特征矩阵 (5 samples, 3 features)
features = torch.rand(5, 3)  # 随机生成 5x3 的特征矩阵
num_clusters = 2  # 定义 2 个聚类簇

# 调用函数
# Q = cluster_features(features, num_clusters)
label1 = [0, 0, 0, 0, 1]
label2 = [1, 1, 1, 0, 0]
label1 = np.array(label1)
label2 = np.array(label2)
align_clusters(label1, label2)

