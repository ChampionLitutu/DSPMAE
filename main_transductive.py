import logging
import numpy as np
from tqdm import tqdm
import torch

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
from graphmae.models import build_model
from datetime import datetime
from logger import Logger
import os
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def getPagerank(g, device):
    # 直接从数据中获取边列表和节点数
    edge_index = g.edges()  # 获取边的元组 (src, dst)
    num_nodes = g.num_nodes()  # 获取总节点数

    # 创建一个 NetworkX 图
    nx_graph = nx.Graph()

    # 将边加入 NetworkX 图中
    # edge_index 是一个元组，包含起始节点和目标节点
    edge_list = zip(edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy())

    nx_graph.add_edges_from(edge_list)  # 将所有边加入图中

    # 计算 PageRank 值
    pr = nx.pagerank(nx_graph)
    pagerank_values = [pr[node] for node in range(num_nodes)]

    # 将 PageRank 值转换为 tensor 并发送到指定设备
    pagerank_values_tensor = torch.tensor(pagerank_values, device=device)

    return pagerank_values_tensor


def select_low_degree_nodes(g, device):
    # Step 1: 获取图的边索引并构建 NetworkX 图
    edge_index = g.edges()  # 边的元组 (src, dst)
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(zip(edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()))

    # Step 2: 计算每个节点的度数并转为 PyTorch 张量
    degrees = torch.tensor([degree for _, degree in nx_graph.degree()], device=device)

    # Step 3: 找出度数最低的 50% 节点的索引
    # num_nodes_to_select = degrees.size(0) // 2  # 取前 50%
    # _, low_degree_indices = torch.topk(degrees, k=num_nodes_to_select, largest=False)
    min_val = degrees.min()
    max_val = degrees.max()

    # 避免除以零的情况
    if max_val == min_val:
        return torch.zeros_like(degrees)

    normalized_degrees = (degrees - min_val) / (max_val - min_val)
    return normalized_degrees


def pretrain(model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f,
             max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    # pr = getPagerank(graph, device)
    # degrees = select_low_degree_nodes(graph, device)
    # degrees = getPagerank(graph, device)
    epoch_iter = tqdm(range(max_epoch))
    acc_list = []
    max_acc = 0.0
    for epoch in epoch_iter:
        model.train()

        loss, loss_dict = model(graph, x)
        # loss, loss_dict = model(graph, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        # logger.info(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        # if logger is not None:
        #     loss_dict["lr"] = get_current_lr(optimizer)
        #     logger.note(loss_dict, step=epoch)
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

    file_handler = logging.FileHandler(f'log/{dataset}_secondTest_epoch_{args}_{time_str}.log')
    file_handler.setLevel(logging.INFO)  # 设置处理器级别
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 添加文件处理器到日志器
    logger.addHandler(file_handler)
    return logger


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

    acc_list = []
    estp_acc_list = []

    # # 模型保存的路径
    # model_dir = "save_model"
    # model_name = f"{dataset_name}Model_{time_str}.pth"
    # save_path = os.path.join(model_dir, model_name)l
    import random
    seeds = []
    seeds = [i for i in range(10)]
    # for i in range(5):
    #     seeds.append(3)

    logger = getLogger(dataset_name, max_epoch)

    print(f"Random seed used: {seeds}")
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)
        graph, (num_features, num_classes) = load_dataset(dataset_name)
        args.num_features = num_features
        logger.info(args)

        # if logs:
        #     logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        # else:
        #     logger = None
        args.num_classes = num_classes
        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None

        x = graph.ndata["feat"]
        if not load_model:
            model = pretrain(model, graph, x, optimizer, max_epoch, device, scheduler, num_classes, lr_f,
                             weight_decay_f, max_epoch_f, linear_prob, logger)
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")

        model = model.to(device)
        model.eval()

        final_acc, estp_acc = node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f,
                                                             max_epoch_f, device, linear_prob, logger=logger)
        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)

        logger.info(f"# final_acc: {final_acc:.4f}")
        logger.info(f"# early-stopping_acc: {estp_acc:.4f}")
        # if logger is not None:
        #     logger.finish()
        # tSNE(model, x, graph, device)
    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"acc_list:  {acc_list}")
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")

    logging.info(f"acc_list:  {acc_list}")
    logging.info(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    logging.info(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")
    logging.info("----------------------------------------------------------------")


def tSNE(model, x, graph, device):
    # 假设以下是编码后的节点表示 (embeddings) 和对应的节点标签 (labels)
    # embeddings: (N, d), N是节点数，d是编码维度
    # labels: (N,)，表示节点的类别
    # 请替换成你的实际数据
    with torch.no_grad():
        embeddings = model.embed(graph.to(device), x.to(device))
    labels = graph.ndata["label"]
    embeddings = embeddings.cpu().numpy()
    # 使用 T-SNE 降维到 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=100, learning_rate=200, n_iter=2000)

    embeddings_2d = tsne.fit_transform(embeddings)

    # 可视化
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10', s=10, alpha=0.8
    )
    plt.colorbar(scatter, label="Node Classes")
    plt.title("My-test")
    plt.xlabel("T-SNE Dimension 1")
    plt.ylabel("T-SNE Dimension 2")
    plt.show()

    # Press the green button in the gutter to run the script.

if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")

    print(args)
    main(args)
    # args.start_epoch = 54
    # main(args)
