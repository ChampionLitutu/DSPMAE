import logging
import numpy as np
from torch_geometric import edge_index
from tqdm import tqdm
import torch
import torch_geometric
from collections import Counter
from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from graphmae.datasets.data_util import load_dataset
from graphmae.evaluation import node_classification_evaluation
# from graphmae.evaluation import node_classification_evaluation, test_classify, test_nc, label_classification
from graphmae.evaluation import node_clustering
from graphmae.models import build_model
from datetime import datetime
from logger import Logger
import os
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

def pretrain(model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f,
             max_epoch_f, linear_prob, difficulty, logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    # pr = getPagerank(graph, device)
    # degrees = select_low_degree_nodes(graph, device)
    # degrees = getPagerank(graph, device)
    epoch_iter = tqdm(range(max_epoch))
    acc_list = []
    clu_acc_list = []
    max_acc = 0.0
    best_nmi = 0
    for epoch in epoch_iter:
        model.train()
        loss, loss_dict = model(graph, x, difficulty, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")


        if (epoch + 1) % 200 == 0 and epoch != max_epoch:
            acc, _ = node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f,
                                                    device, linear_prob, mute=True, logger=logger)
            acc_list.append(acc)
            if acc > max_acc:
                max_acc = acc
    # logger.info(f"# all_epoch_max_acc: {max(acc_list):.4f}")
    # print(f"# all_epoch_max_acc: {max(acc_list):.4f}")
    # return best_model
    return model


def getLogger(dataset, args):
    current_time = datetime.now()
    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置日志级别

    # 移除现有的所有处理器（包括 StreamHandler）
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # file_handler = logging.FileHandler(f'log/{dataset}_momentum_{args}_{time_str}.log')
    file_handler = logging.FileHandler(f'log/1Layer_{dataset}_{args}_GraphMAE_{time_str}.log')
    file_handler.setLevel(logging.INFO)  # 设置处理器级别
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 添加文件处理器到日志器
    logger.addHandler(file_handler)
    return logger


def compute_difficulty(pseudo_labels, edge_index):
    """
    计算每个节点的难度 D_local(u)
    pseudo_labels: 每个节点的伪标签张量 [num_nodes]
    edge_index: 图的边索引，形状为 [2, num_edges]，表示节点的连接关系

    返回 D_local(u) 的张量
    """
    edge_index = torch.stack(edge_index, dim=0)
    num_nodes = pseudo_labels.shape[0]
    difficulty = torch.zeros(num_nodes)

    # 构建邻接矩阵（邻居信息）
    adj = torch_geometric.utils.to_dense_adj(edge_index)[0]  # 转换为稠密邻接矩阵

    for u in range(num_nodes):
        # 获取节点 u 的邻居
        neighbors = adj[u].nonzero(as_tuple=False).squeeze()  # 获取 u 的邻居节点索引
        if neighbors.ndimension() == 0:
            neighbors = neighbors.unsqueeze(0)

        if len(neighbors) == 0:
            continue  # 如果没有邻居，跳过

        # 获取邻居节点的伪标签
        neighbor_labels = pseudo_labels[neighbors]
        if not isinstance(neighbor_labels, np.ndarray):
            neighbor_labels = np.array([neighbor_labels])
        # 计算每个类别的概率 P_c(u)
        label_counts = Counter(neighbor_labels)  # 统计各个标签的数量
        # label_counts = Counter(neighbor_labels.cpu().numpy())  # 统计各个标签的数量
        total_neighbors = len(neighbors)

        # 计算 P_c(u) 对于每个类别 c
        P_c_u = {c: count / total_neighbors for c, count in label_counts.items()}

        # 计算 D_local(u) = - Σ P_c(u) log(P_c(u))
        D_local_u = 0.0
        for c, P_c in P_c_u.items():
            D_local_u -= P_c * torch.log(torch.tensor(P_c))  # 使用 log(P_c(u))

        # 存储计算结果
        difficulty[u] = D_local_u

    return difficulty


def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler
    momentum = args.momentum
    acc_list = []
    estp_acc_list = []
    clu_acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []


    seeds = [i for i in range(1)]

    logger = getLogger(dataset_name, args.prompt_num)
    print(f"Random seed used: {seeds}")
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)


        graph, (num_features, num_classes) = load_dataset(dataset_name)
        args.num_features = num_features
        args.num_classes = num_classes

        logging.info("Saveing Model ...")
        model_pretrain = build_model(args)
        model_pretrain.load_state_dict(torch.load("checkpoint_cora0.pt", weights_only=True))
        model_pretrain = model_pretrain.to(device)
        x = graph.ndata["feat"]

        with torch.no_grad():
            features = model_pretrain.embed(graph.to(device), x.to(device))
        kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(features.cpu().detach().numpy())  # 聚类结果
        edge_index = graph.edges()
        difficulty = compute_difficulty(labels, edge_index)

        logger.info(args)

        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None

        model = pretrain(model, graph, x, optimizer, max_epoch, device, scheduler, num_classes, lr_f,
                         weight_decay_f, max_epoch_f, linear_prob, difficulty, logger)
        model = model.cpu()

        model = model.to(device)
        model.eval()

        final_acc, estp_acc = node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f,
                                                             max_epoch_f, device, linear_prob, logger=logger)
        # acc_list.append(acc)
        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)
        clu_acc_final = node_clustering(model, graph, x, num_classes, 'final', device)
        clu_acc_list.append(clu_acc_final[0])
        # logger.info(f"# final_acc: {acc}")
        logger.info(f"# final_acc: {final_acc:.4f}")
        logger.info(f"# early-stopping_acc: {estp_acc:.4f}")
        logger.info(f"# clu_acc_final: {clu_acc_final}")
        nmi_list.append(clu_acc_final[1])
        ari_list.append(clu_acc_final[2])
        f1_list.append(clu_acc_final[3])
        # if logger is not None:
        #     logger.finish()
        # tSNE(model, x, graph, device)
    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"acc_list:  {acc_list}")
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")
    print(f"clu_acc_final:  {clu_acc_list}")
    print(f"nmi_acc_final:  {nmi_list}")
    print(f"ari_acc_final:  {ari_list}")
    print(f"f1_acc_final:  {f1_list}")
    print(f"# clu_acc: {np.mean(clu_acc_list):.4f}±{np.std(clu_acc_list):.4f}")
    print(f"# nmi_acc: {np.mean(nmi_list):.4f}±{np.std(nmi_list):.4f}")
    print(f"# ari_acc: {np.mean(ari_list):.4f}±{np.std(ari_list):.4f}")
    print(f"# f1_acc: {np.mean(f1_list):.4f}±{np.std(f1_list):.4f}")
    logging.info(f"# clu_acc: {np.mean(clu_acc_list):.4f}±{np.std(clu_acc_list):.4f}")
    logging.info(f"# nmi_acc: {np.mean(nmi_list):.4f}±{np.std(nmi_list):.4f}")
    logging.info(f"# ari_acc: {np.mean(ari_list):.4f}±{np.std(ari_list):.4f}")
    logging.info(f"# f1_acc: {np.mean(f1_list):.4f}±{np.std(f1_list):.4f}")
    logging.info(f"acc_list:  {acc_list}")
    logging.info(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    logging.info(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")
    logging.info("----------------------------------------------------------------")



if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")

    print(args)
    main(args)
    # args.start_epoch = 54
    # main(args)
