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
# from graphmae.evaluation import node_classification_evaluation, test_classify, test_nc, label_classification
from graphmae.models import build_model
from datetime import datetime
from logger import Logger
import os
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import dgl
import torch.nn as nn
import dgl.function as fn
from sklearn.metrics import roc_auc_score, accuracy_score, adjusted_rand_score, confusion_matrix, f1_score
import copy
class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            return g.edata["score"][:, 0]

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    )
    return roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy())

def link_prediction_for_transductive(model, graph, x, lr_lp, weight_decay_lp, max_epoch, device, test_ratio=0.1, mute=False):
    model.eval()
    decoder = DotPredictor()
    decoder.to(device)
    optimizer = torch.optim.Adam(
        [{'params': decoder.parameters()}], lr=lr_lp, weight_decay=weight_decay_lp
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    graph = graph.to(device)
    x = x.to(device)

    num_nodes = graph.number_of_nodes()
    u, v = graph.edges()
    num_edges = graph.number_of_edges()
    eids = torch.randperm(num_edges).to(device)
    test_size = int(eids.shape[0] * test_ratio)
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    num_samples = num_edges
    neg_u, neg_v = dgl.sampling.global_uniform_negative_sampling(graph, num_samples, exclude_self_loops=True)
    neg_eids = torch.randperm(num_edges)
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    train_g = dgl.remove_edges(graph, eids[:test_size])
    train_g = dgl.add_self_loop(train_g)
    # test_g = graph.edge_subgraph(eids[:test_size])
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=num_nodes)
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=num_nodes)
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=num_nodes)
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=num_nodes)

    best_test_auc = 0
    best_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model.embed(train_g, train_g.ndata["feat"])
        pos_score = decoder(train_pos_g, out)
        neg_score = decoder(train_neg_g, out)
        loss = criterion(pos_score, neg_score)
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            out = model.embed(train_g, train_g.ndata["feat"])
            pos_score_test = decoder(test_pos_g, out)
            neg_score_test = decoder(test_neg_g, out)
            test_auc = compute_auc(pos_score_test, neg_score_test)
            pos_score_train = decoder(train_pos_g, out)
            neg_score_train = decoder(train_neg_g, out)
            train_auc = compute_auc(pos_score_train, neg_score_train)
            test_loss = criterion(pos_score_test, neg_score_test)
        if test_auc >= best_test_auc:
            best_test_auc = test_auc
            # print('best', best_test_auc)
            best_epoch = epoch
            best_model = copy.deepcopy(decoder)

        if not mute:
            epoch_iter.set_description(
                f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, train_auc:{train_auc: .4f}, "
                f"test_loss:{test_loss.item(): .4f}, test_auc:{test_auc: .4f}")

        best_model.eval()
        with torch.no_grad():
            pos_score = best_model(test_pos_g, out)
            neg_score = best_model(test_neg_g, out)
            test_auc = compute_auc(pos_score, neg_score)
        if mute:
            print(
                f"# IGNORE: --- TestAUC: {test_auc:.4f}, early-stopping-TestAUC: {best_test_auc:.4f} in epoch {best_epoch} --- ")
        else:
            print(
                f"--- TestAUC: {test_auc:.4f}, early-stopping-TestAUC: {best_test_auc:.4f} in epoch {best_epoch} --- ")

        # (final_acc, es_acc, best_acc)
        return test_auc, best_test_auc

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
            test_auc, best_test_auc = link_prediction_for_transductive(model, graph, x, lr_f, weight_decay_f, 70,
                                             device, test_ratio=0.1, mute=True)
            print("test_auc: {}, best_test_auc:{}".format(test_auc, best_test_auc))
            acc_list.append(test_auc)
            if test_auc > max_acc:
                max_acc = test_auc
    logger.info(f"# all_epoch_max_acc: {max(acc_list):.4f}")
    print(f"# all_epoch_max_acc: {max(acc_list):.4f}")
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
    file_handler = logging.FileHandler(f'log/Link_Prediction_{dataset}_GraphMAE_{time_str}.log')
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
    momentum = args.momentum
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

    logger = getLogger(dataset_name, args.prompt_num)

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

        # with torch.no_grad():
        #     feature = model.embed(graph.to(device), x.to(device))
        # final_acc, estp_acc = test_classify(feature.cpu(), graph.ndata["label"])


        # with torch.no_grad():
        #     emb = model.embed(graph.to(device), x.to(device))
        # train_mask = graph.ndata["train_mask"]
        # val_mask = graph.ndata["val_mask"]
        # test_mask = graph.ndata["test_mask"]
        # labels = graph.ndata["label"]
        # # acc = label_classification(emb, train_mask, val_mask, test_mask, labels)

        final_acc, estp_acc = link_prediction_for_transductive(model, graph, x, lr_f, weight_decay_f, 70,
                                                               device, test_ratio=0.1, mute=True)
        # acc_list.append(acc)
        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)

        # logger.info(f"# final_acc: {acc}")
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
