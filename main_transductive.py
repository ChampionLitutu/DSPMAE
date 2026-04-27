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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import seaborn as sns


def pretrain(model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f,
             max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)
    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        model.train()

        # loss, loss_dict = model(graph, x, epoch)
        loss, loss_dict = model(graph, x)

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
        # if epoch == 0 or epoch == max_epoch / 2:
        #     tSNE(model, x, graph, device)
        # if (epoch + 1) % 200 == 0 and epoch != max_epoch:
        #     acc, _ = node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f,
        #                                             device, linear_prob, mute=True, logger=logger)
        #     acc_list.append(acc)
        #
        #     # clu_acc = node_clustering(model, graph, x, num_classes, epoch, device)
        #
        #     # clu_acc_list.append(clu_acc)
        #     if acc > max_acc:
        #         max_acc = acc
        # if epoch  == max_epoch / 2:
        #     tSNE(model, x, graph, device)
        # if epoch == 0:
        #     tSNE(model, x, graph, device)
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
    file_handler = logging.FileHandler(f'log/{dataset}_{time_str}_{args}__GraphMAE.log')
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
    loss_lamda = args.loss_lamda
    acc_list = []
    estp_acc_list = []
    clu_acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []

    # # 模型保存的路径
    # model_dir = "save_model"
    # model_name = f"{dataset_name}Model_{time_str}.pth"
    # save_path = os.path.join(model_dir, model_name)l
    import random
    # seeds = [0]
    seeds = [i for i in range(10)]
    # for i in range(5):
    #     seeds.append(3)

    logger = getLogger(dataset_name, args.mask_rate)
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
            torch.save(model.state_dict(), "checkpoint_cora0.pt")

        model = model.to(device)
        model.eval()

        with torch.no_grad():
            feature = model.embed(graph.to(device), x.to(device))
        # final_acc, estp_acc = test_classify(feature.cpu(), graph.ndata["label"])


        final_acc, estp_acc = node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f,
                                                             max_epoch_f, device, linear_prob, logger=logger)
        # acc_list.append(acc)
        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)
        # clu_acc_final = node_clustering(model, graph, x, num_classes, 'final', device)
        # clu_acc_list.append(clu_acc_final[0])
        # logger.info(f"# final_acc: {acc}")
        logger.info(f"# final_acc: {final_acc:.4f}")
        logger.info(f"# early-stopping_acc: {estp_acc:.4f}")
        # logger.info(f"# clu_acc_final: {clu_acc_final}")
        # nmi_list.append(clu_acc_final[1])
        # ari_list.append(clu_acc_final[2])
        # f1_list.append(clu_acc_final[3])
        # if logger is not None:
        #     logger.finish()
        tSNE(model, x, graph, device)
        print("show")
        # plot_hypersphere_distribution(feature, "test")
    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"acc_list:  {acc_list}")
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")
    # print(f"clu_acc_final:  {clu_acc_list}")
    # print(f"nmi_acc_final:  {nmi_list}")
    # print(f"ari_acc_final:  {ari_list}")
    # print(f"f1_acc_final:  {f1_list}")
    # print(f"# clu_acc: {np.mean(clu_acc_list):.4f}±{np.std(clu_acc_list):.4f}")
    # print(f"# nmi_acc: {np.mean(nmi_list):.4f}±{np.std(nmi_list):.4f}")
    # print(f"# ari_acc: {np.mean(ari_list):.4f}±{np.std(ari_list):.4f}")
    # print(f"# f1_acc: {np.mean(f1_list):.4f}±{np.std(f1_list):.4f}")
    #
    # logging.info(f"clu_acc_final:  {clu_acc_list}")
    # logging.info(f"nmi_acc_final:  {nmi_list}")
    # logging.info(f"ari_acc_final:  {ari_list}")
    # logging.info(f"f1_acc_final:  {f1_list}")
    # logging.info(f"# clu_acc: {np.mean(clu_acc_list):.4f}±{np.std(clu_acc_list):.4f}")
    # logging.info(f"# nmi_acc: {np.mean(nmi_list):.4f}±{np.std(nmi_list):.4f}")
    # logging.info(f"# ari_acc: {np.mean(ari_list):.4f}±{np.std(ari_list):.4f}")
    # logging.info(f"# f1_acc: {np.mean(f1_list):.4f}±{np.std(f1_list):.4f}")
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
    tsne = TSNE(n_components=2, random_state=42, perplexity=150, learning_rate=200, n_iter=5000)

    embeddings_2d = tsne.fit_transform(embeddings)
    labels = labels.cpu().numpy()
    # 可视化
    plt.figure(figsize=(10, 8))
    # scatter = plt.scatter(
    #     embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10', s=10, alpha=0.8
    # )
    plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10', s=10, alpha=0.8
    )
    # plt.colorbar(scatter, label="Node Classes")


    # plt.title("Cora")
    # plt.xlabel("T-SNE Dimension 1")
    # plt.ylabel("T-SNE Dimension 2")
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
