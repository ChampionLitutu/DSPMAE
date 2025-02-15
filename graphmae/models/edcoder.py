from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor

from .gin import GIN
from .gat import GAT
from .gcn import GCN
from .dot_gat import DotGAT
from .loss_func import sce_loss
from graphmae.utils import create_norm, drop_edge
import torch.nn.functional as F
from torch_geometric.utils import scatter, softmax
from typing import Callable, Optional, Tuple, Union
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
import dgl
from torch_geometric.utils import add_self_loops, negative_sampling, degree
from torch.nn import MultiheadAttention
def topk(
    x: Tensor,
    ratio: Optional[Union[float, int]],
    min_score: Optional[float] = None,
    tol: float = 1e-7,
    largest: bool = True
) -> Tensor:
    # 判断是否需要基于 min_score 过滤节点
    if min_score is not None:
        # 确保不会过滤掉所有节点，设置一个容差
        scores_max = x.max() - tol  # 找到节点分数的最大值
        scores_min = min(scores_max, min_score)  # 设定最小保留分数

        # 获取满足分数条件的节点索引
        perm = (x > scores_min).nonzero().view(-1)
    # 若指定了保留比例或数量 ratio
    elif ratio is not None:
        # 计算所需的保留节点数 k
        num_nodes = x.size(0)  # 节点总数
        if ratio >= 1:
            k = int(ratio)  # 如果 ratio 是整数，则直接表示保留节点数
            k = min(k, num_nodes)  # 确保不超过节点总数
        else:
            k = int(np.ceil(ratio * num_nodes))

        # 计算 top-k 节点的索引
        _, perm = x.topk(k=k, largest=largest, sorted=True)  # 按分数从高到低选取前 k 个节点
    else:
        raise ValueError("必须指定 'min_score' 或 'ratio' 参数之一")

    return perm

def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "mlp":
        # * just for decoder
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError

    return mod


class TransformerLayer(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super(TransformerLayer, self).__init__()
        self.attention = MultiheadAttention(embed_dim=in_channels, num_heads=num_heads)
        self.fc = nn.Linear(in_channels, in_channels)  # Keep dimensions consistent

    def forward(self, x):
        # x.shape: (num_nodes, in_channels)
        x = x.unsqueeze(1)  # Add batch dimension for MultiheadAttention
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.squeeze(1)  # Remove batch dimension
        return F.relu(self.fc(attn_output))  # Pass through a fully connected layer

class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gcn",
            decoder_type: str = "gcn",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            start_epoch1: int = 300,
            start_epoch2: int = 500,
            alpha: float = 0.25,
            momentum: float = 0.996,
            alpha_l2: int = 2,
            prompt_num: int = 2
         ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden

        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden

        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.enc_re_mask_token = nn.Parameter(torch.zeros(1, dec_in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

        self.mse_loss_fn = nn.MSELoss()
        self.projector = nn.Sequential(
            nn.Linear(num_hidden, 256),
            nn.PReLU(),
            nn.Linear(256, num_hidden),
        )

        self.loss_weight = nn.Parameter(torch.tensor(1.0))
        self.loss_weight2 = nn.Parameter(torch.tensor(1.0))
        self.dimReductionMLP = nn.Linear(num_hidden, 256)  # 第一层线性层
        self.dimReductionMLP2 = nn.Linear(256, 128)  # 第二层线性层
        # self._start_epoch1 = start_epoch1
        # self._start_epoch2 = start_epoch2
        self._alpha = alpha

        self.projector_ema = nn.Sequential(
            nn.Linear(num_hidden, 256),
            nn.PReLU(),
            nn.Linear(256, num_hidden),
        )

        self.encoder_ema = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )
        self.encoder_ema.load_state_dict(self.encoder.state_dict())
        self.projector_ema.load_state_dict(self.projector.state_dict())
        self.predictor = nn.Sequential(
            nn.PReLU(),
            nn.Linear(num_hidden, num_hidden)
        )
        for p in self.encoder_ema.parameters():
            p.requires_grad = False
            p.detach_()
        for p in self.projector_ema.parameters():
            p.requires_grad = False
            p.detach_()
        # self._delayed_ema_epoch = 0
        self.print_num_parameters()

        # self._momentum = 0.997
        self._momentum = momentum
        self.alpha_l2 = alpha_l2
        self.prompt_num = prompt_num
        self.edge_decoder = EdgeDecoder(num_hidden, 64, num_layers = 2, dropout= 0.2)
        # self._mu = nn.Parameter(torch.tensor(1.0))
        # self._nu = nn.Parameter(torch.tensor(1.0))
        # self.std_expander = nn.Sequential(nn.Linear(num_hidden, num_hidden),
        #                                   nn.PReLU())
        self.negative_sampler = negative_sampling
        self.transformer = TransformerLayer(num_hidden, num_heads=4)
        print("prompt_num:{}".format(prompt_num))
        # self.reset_parameters_for_token()
    def print_num_parameters(self):
        num_encoder_params = [p.numel() for p in self.encoder.parameters() if  p.requires_grad]
        num_decoder_params = [p.numel() for p in self.decoder.parameters() if  p.requires_grad]
        num_params = [p.numel() for p in self.parameters() if p.requires_grad]

        print(f"num_encoder_params: {sum(num_encoder_params)}, num_decoder_params: {sum(num_decoder_params)}, num_params_in_total: {sum(num_params)}")

    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.enc_re_mask_token)
        # nn.init.xavier_normal_(self.dec_mask_token)
        nn.init.xavier_normal_(self.encoder_to_decoder.weight, gain=1.414)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            # out_x = x.clone()
            # token_nodes = mask_nodes
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[token_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()
        return use_g, out_x, (mask_nodes, keep_nodes)

    # def encoding_mask_noise2(self, x, mask_rate, scores, epoch, max_epoch, curMode):
    def encoding_mask_noise2(self, x, mask_rate, scores, epoch, max_epoch):
        num_nodes = x.shape[0]
        if self._start_epoch1 < epoch < self._start_epoch2:
        # if self._start_epoch1  epoch:
            throw_nodes_first = topk(scores, mask_rate, largest=True).to(x.device)  # 选取得分最高的关注节点
            tmp_scores = torch.tensor(np.random.uniform(0, 1, num_nodes)).to(x.device)  # 生成随机分数
            tmp_scores[throw_nodes_first] += self._alpha  # 将关注节点的分数加上alpha
        # elif 300> epoch > self._start_epoch2:
        #     throw_nodes_first = topk(scores, mask_rate, largest=False).to(x.device)  # 选取得分最高的关注节点
        #     tmp_scores = torch.tensor(np.random.uniform(0, 1, num_nodes)).to(x.device)  # 生成随机分数
        #     tmp_scores[throw_nodes_first] += self._alpha  # 将关注节点的分数加上alpha
        else:
            tmp_scores = torch.tensor(np.random.uniform(0, 1, num_nodes)).to(x.device)  # 生成随机分数

        keep_nodes = topk(-tmp_scores, 1 - mask_rate, largest=True).to(x.device)  # 保留得分最低的1 - mask_rate的节点
        all_indices = torch.arange(num_nodes).to(x.device)
        mask_nodes = all_indices[~torch.isin(all_indices, keep_nodes)].to(x.device)
        if self._replace_rate > 0:
            num_mask_nodes = len(mask_nodes)
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        return out_x, (mask_nodes, keep_nodes)

    # def encoding_mask_noise_diff(self, g, x, mask_rate, difficulty , epoch, max_epoch):
    #     num_nodes = x.shape[0]
    #     proportion = 0.5
    #     sorted_indices = torch.argsort(difficulty).to(x.device)  # 根据难度排序
    #     g_t = min(1, proportion + (1 - proportion) * (epoch / max_epoch))
    #
    #     mask_set_len = int(g_t * num_nodes)
    #     mask_set = sorted_indices[:mask_set_len]
    #     num_mask_nodes = int(mask_rate * num_nodes)
    #
    #     perm = torch.randperm(mask_set_len, device=x.device)
    #     mask_nodes_indices = perm[: num_mask_nodes].to(x.device)
    #     mask_nodes = mask_set[mask_nodes_indices].to(x.device)
    #     keep_nodes = sorted_indices[~torch.isin(sorted_indices, mask_nodes)]
    #
    #     if self._replace_rate > 0:
    #         num_noise_nodes = int(self._replace_rate * num_mask_nodes)
    #         perm_mask = torch.randperm(num_mask_nodes, device=x.device)
    #         token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
    #         noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
    #         noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]
    #
    #         out_x = x.clone()
    #         out_x[token_nodes] = 0.0
    #         out_x[noise_nodes] = x[noise_to_be_chosen]
    #     else:
    #         # out_x = x.clone()
    #         # token_nodes = mask_nodes
    #         out_x = x.clone()
    #         token_nodes = mask_nodes
    #         out_x[token_nodes] = 0.0
    #
    #     out_x[token_nodes] += self.enc_mask_token
    #     use_g = g.clone()
    #     return use_g, out_x, (mask_nodes, keep_nodes)
    def encoding_mask_noise_diff(self, g, x, mask_rate, difficulty, epoch, max_epoch):
        num_nodes = x.shape[0]

        # 计算 g(t)
        max_difficulty = torch.max(difficulty)
        min_difficulty = torch.min(difficulty)
        # g_t = min_difficulty + (max_difficulty - min_difficulty) * (epoch / max_epoch)  # 假设epoch最多训练100次

        proportion = 0.5
        g_t = min(1, proportion + (1 - proportion) * (epoch / max_epoch))
        mask_nodes = torch.nonzero(difficulty >= g_t).squeeze()

        num_mask_nodes = int(mask_rate * num_nodes)
        if mask_nodes.ndimension() == 0:
            # 如果是标量，将其转换为 1D 张量
            mask_nodes = mask_nodes.unsqueeze(0)

        if mask_nodes.shape[0] > num_mask_nodes:
            # 如果选择的难度节点数量多于需要掩码的数量，从中随机选择
            mask_nodes = mask_nodes[torch.randperm(mask_nodes.shape[0])[:num_mask_nodes]].to(x.device)
        else:
            # 如果选择的难度节点数量少于需要掩码的数量，补充选择难度较低的节点
            remaining_indices = torch.nonzero(difficulty < g_t).squeeze()
            additional_masked_indices = remaining_indices[
                torch.randperm(len(remaining_indices))[:(num_mask_nodes - mask_nodes.shape[0])]]
            mask_nodes = torch.cat([mask_nodes, additional_masked_indices]).to(x.device)

        all_nodes = torch.arange(num_nodes, device=x.device)
        keep_nodes = all_nodes[~torch.isin(all_nodes, mask_nodes)]
        if self._replace_rate > 0:
            # 计算噪声节点数量
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)

            # 随机选择掩码节点中的 token_nodes 和 noise_nodes
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]

            # 选择噪声节点
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            # 处理输出
            out_x = x.clone()
            out_x[token_nodes] = 0.0  # 将掩码节点的特征置为0
            out_x[noise_nodes] = x[noise_to_be_chosen]  # 用噪声节点替换

        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[token_nodes] = 0.0  # 将掩码节点的特征置为0

        # 添加掩码标记
        out_x[token_nodes] += self.enc_mask_token

        # 克隆图数据（如果需要对图进行进一步修改）
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    def encoding_mask_noise_diff_less_proportion(self, g, x, mask_rate, difficulty, epoch, max_epoch):
        num_nodes = x.shape[0]

        # 计算 g(t)
        max_difficulty = torch.max(difficulty)
        min_difficulty = torch.min(difficulty)
        mid_difficulty= difficulty[int(num_nodes / 2)]
        g_t = mid_difficulty + (max_difficulty - mid_difficulty) * (epoch / max_epoch)  # 假设epoch最多训练100次

        # proportion = 0.4
        # g_t = min(1, proportion + (1 - proportion) * (epoch / max_epoch))
        # # g_t = g_t if g_t < 0.5 else 0.5
        mask_nodes = torch.nonzero(difficulty >= g_t).squeeze()
        print(mask_nodes.shape)
        num_mask_nodes = int(g_t * num_nodes)
        if mask_nodes.ndimension() == 0:
            # 如果是标量，将其转换为 1D 张量
            mask_nodes = mask_nodes.unsqueeze(0)

        if mask_nodes.shape[0] > num_mask_nodes:
            # 如果选择的难度节点数量多于需要掩码的数量，从中随机选择
            mask_nodes = mask_nodes[torch.randperm(mask_nodes.shape[0])[:num_mask_nodes]].to(x.device)
        else:
            # 如果选择的难度节点数量少于需要掩码的数量，补充选择难度较低的节点
            remaining_indices = torch.nonzero(difficulty < g_t).squeeze()
            additional_masked_indices = remaining_indices[
                torch.randperm(len(remaining_indices))[:(num_mask_nodes - mask_nodes.shape[0])]]
            mask_nodes = torch.cat([mask_nodes, additional_masked_indices]).to(x.device)

        all_nodes = torch.arange(num_nodes, device=x.device)
        keep_nodes = all_nodes[~torch.isin(all_nodes, mask_nodes)]
        if self._replace_rate > 0:
            # 计算噪声节点数量
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)

            # 随机选择掩码节点中的 token_nodes 和 noise_nodes
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]

            # 选择噪声节点
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            # 处理输出
            out_x = x.clone()
            out_x[token_nodes] = 0.0  # 将掩码节点的特征置为0
            out_x[noise_nodes] = x[noise_to_be_chosen]  # 用噪声节点替换

        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[token_nodes] = 0.0  # 将掩码节点的特征置为0

        # 添加掩码标记
        out_x[token_nodes] += self.enc_mask_token

        # 克隆图数据（如果需要对图进行进一步修改）
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)
    # def forward(self, g, x, difficulty, epoch):
    #     loss = self.mask_attr_prediction_diff(g, x, difficulty, epoch)
    #     loss_item = {"loss": loss.item()}
    #     return loss, loss_item

    def forward(self, g, x):
        loss = self.mask_attr_prediction(g, x)
        # loss = self.mask_attr_prediction(g, x, pr, epoch,  max_epoch)
        loss_item = {"loss": loss.item()}
        return loss, loss_item


    def align_clusters_by_centroids(self, centroids1, centroids2):
        """
        使用质心对齐两个聚类标签。
        参数：
            centroids1: 第一次聚类质心，形状 (K, D)
            centroids2: 第二次聚类质心，形状 (K, D)
        返回：
            对齐的簇标签映射
        """
        cost_matrix = np.linalg.norm(centroids1[:, None, :] - centroids2[None, :, :], axis=2)  # 欧氏距离
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 构造标签映射
        label_mapping = {col: row for row, col in zip(row_ind, col_ind)}
        return label_mapping

    def compute_nonmatching_contrastive_loss(self, features_original, features_masked, num_clusters=10,
                                             distance_metric='euclidean'):
        """
        计算两视图中不同簇节点之间的对比损失。
        参数：
            features_original: 原始图的特征，形状 (N, D)
            features_masked: 掩码图的特征，形状 (N, D)
            num_clusters: 聚类簇的数量
            distance_metric: 特征距离度量方式（默认欧几里得距离）。
        返回：
            非同簇节点的对比损失
        """

        def perform_kmeans(features, num_clusters):
            """
            对特征执行 K-Means 聚类，并返回聚类结果和质心。
            """
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(features.cpu().detach().numpy())  # 聚类结果
            centroids = torch.tensor(kmeans.cluster_centers_, device=features.device)  # 聚类质心
            return labels, centroids

        def align_labels(labels_source, centroids_source, centroids_target):
            """
            使用匈牙利算法对齐聚类标签。
            """
            cost_matrix = torch.cdist(centroids_source, centroids_target)  # 质心距离矩阵
            row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
            mapping = {col: row for row, col in zip(row_ind, col_ind)}
            return torch.tensor([mapping[label] for label in labels_source], device=features_original.device)

        # Step 1: 对原图和掩码图分别进行 K-Means 聚类
        labels_original, centroids_original = perform_kmeans(features_original, num_clusters)
        labels_masked, centroids_masked = perform_kmeans(features_masked, num_clusters)

        # Step 2: 对齐掩码图的簇标签
        labels_masked_aligned = align_labels(labels_masked, centroids_masked, centroids_original)

        # 转换为 torch.Tensor 类型
        labels_original = torch.tensor(labels_original, device=features_original.device)
        labels_masked = labels_masked_aligned

        # Step 3: 创建不同簇的掩码
        different_cluster_mask = (labels_original != labels_masked)  # 不同簇掩码

        # Step 4: 计算节点特征距离
        if distance_metric == 'euclidean':
            distance = F.pairwise_distance(features_original, features_masked, p=2)  # 欧几里得距离
        elif distance_metric == 'cosine':
            distance = 1 - F.cosine_similarity(features_original, features_masked)
        else:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")

        # Step 5: 计算不同簇节点对的损失
        loss = torch.mean(different_cluster_mask.float() * distance ** 2)

        return loss

    def perform_kmeans(features, num_clusters):
        """
        对特征执行 K-Means 聚类，并返回聚类结果和质心。
        """
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(features.cpu().detach().numpy())  # 聚类结果
        centroids = torch.tensor(kmeans.cluster_centers_, device=features.device)  # 聚类质心
        return labels, centroids

    def mask_attr_prediction_origin(self, g, x):
        num_nodes = g.num_nodes()

        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)

        use_g = pre_use_g
        enc_rep, all_hidden = self.encoder(pre_use_g, use_x, return_hidden=True)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)
        #
        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "liear"):
            recon = self.decoder(rep)
        else:
            recon = self.decoder(use_g, rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)


        return loss

    def mask_attr_prediction_addMaskOriginNode(self, g, x):
        num_nodes = g.num_nodes()

        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)

        use_g = pre_use_g

        with torch.no_grad():
            enc_rep_nomask, _ = self.encoder(g, x, return_hidden=True)

        enc_rep_nomask_reduced = self.dimReductionMLP2(F.relu(self.dimReductionMLP(enc_rep_nomask)))  # 通过第一层并应用ReLU激活

        with torch.no_grad():
            edge_nodeID1, edge_nodeID2 = self.batch_top_k_cosine_similarity(enc_rep_nomask_reduced, 10240, mask_nodes, k=1)

        pre_use_g_add_edge = use_g.clone()

        # pre_use_g_add_edge.add_edges(edge_nodeID2, edge_nodeID1)

        new_node_features = x[mask_nodes]
        pre_use_g_add_edge.add_nodes(len(mask_nodes), data={'feat': new_node_features})  # 新增节点并赋特征

        # 新节点索引
        new_node_ids = torch.arange(pre_use_g_add_edge.num_nodes() - len(mask_nodes), pre_use_g_add_edge.num_nodes(), device=g.device)
        pre_use_g_add_edge.add_edges(new_node_ids, mask_nodes)
        pre_use_g_add_edge.add_edges(new_node_ids, new_node_ids)
        use_x = torch.cat([use_x, new_node_features], dim=0)

        enc_rep, all_hidden = self.encoder(pre_use_g_add_edge, use_x, return_hidden=True)
        enc_rep = enc_rep[:num_nodes]

        # loss_cluster = self.compute_nonmatching_contrastive_loss(enc_rep_nomask, enc_rep, self.num_classes)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)
        #
        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "liear"):
            recon = self.decoder(rep)
        else:
            # recon = self.decoder(pre_use_g_add_edge, rep)
            recon = self.decoder(use_g, rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)
        # loss = self.criterion(x_rec, x_init) + self.loss_weight * loss_cluster


        return loss

    def mask_attr_prediction4(self, g, x):
        num_nodes = g.num_nodes()
        # if int(max_epoch * self._start_rate1) < epoch < int(max_epoch * self._start_rate2):
        # use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise2(x, self._mask_rate, pr, epoch, max_epoch)
        # pre_use_g = g.clone()

        # if self._start_epoch1 < epoch < self._start_epoch2:
        # if epoch % 100 == 1:
        #     use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise2(x, 0.1, pr, epoch, max_epoch)
        #     pre_use_g = g.clone()
        # else:
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)

        # pre_use_g2, use_x2, (mask_nodes2, keep_nodes2) = self.encoding_mask_noise(g, x, self._mask_rate)

        use_g = pre_use_g

        # enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)
        with torch.no_grad():
            enc_rep_nomask, _ = self.encoder(g, x, return_hidden=True)
        # enc_rep_nomask = self.encoder_ema(g, x, )
        # enc_rep_nomask_reduced = self.dimReductionMLP2(F.relu(self.dimReductionMLP(enc_rep_nomask)))  # 通过第一层并应用ReLU激活
        with torch.no_grad():
            edge_nodeID1, edge_nodeID2 = self.batch_top_k_cosine_similarity(x, 2048, mask_nodes, k=1)
            # edge_nodeID12, edge_nodeID22 = self.batch_top_k_cosine_similarity(x, 2048, mask_nodes2, k=1)

        pre_use_g_add_edge = use_g.clone()
        # pre_use_g_add_edge2 = use_g.clone()

        pre_use_g_add_edge.add_edges(edge_nodeID2, edge_nodeID1)
        # pre_use_g_add_edge.add_edges(edge_nodeID1, edge_nodeID2)
        # pre_use_g_add_edge.add_edges(edge_nodeID1, edge_nodeID2)
        # pre_use_g_add_edge2.add_edges(edge_nodeID22, edge_nodeID12)
        #
        enc_rep, all_hidden = self.encoder(pre_use_g_add_edge, use_x, return_hidden=True)
        # with torch.no_grad():
        #     enc_rep2, all_hidden2 = self.encoder(pre_use_g_add_edge2, use_x2, return_hidden=True)

        # latent_target = self.projector_ema(enc_rep_nomask[keep_nodes])
        # #
        # latent_pred = self.projector(enc_rep[keep_nodes])
        # latent_pred = self.predictor(latent_pred)
        # loss_latent = sce_loss(latent_pred, latent_target, 1)


        # loss_latent = sce_loss(enc_rep, enc_rep_nomask)
        # loss_latent = sce_loss(enc_rep[keep_nodes], enc_rep_nomask[keep_nodes])

        # loss_cluster = self.compute_nonmatching_contrastive_loss(enc_rep_nomask, enc_rep, self.num_classes)
        # loss_cluster = self.compute_global_contrastive_loss(enc_rep_nomask, enc_rep, self.num_classes)
        # ---- attribute reconstruction ----
        # rep = self.encoder_to_decoder(final_enc_rep)
        rep = self.encoder_to_decoder(enc_rep)
        #
        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "liear"):
            recon = self.decoder(rep)
        else:
            recon = self.decoder(use_g, rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        # loss = self.criterion(x_rec, x_init) + self.loss_weight * loss_cluster
        # loss = self.criterion(x_rec, x_init) + self.loss_weight * loss_cluster
        loss = self.criterion(x_rec, x_init)
        # loss = self.criterion(x_rec, x_init) + self.loss_weight * loss_latent

        # self.ema_update()
        return loss
    def mask_attr_prediction(self, g, x):
        num_nodes = g.num_nodes()

        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)
        use_g = pre_use_g

        enc_rep_nomask = self.encoder_ema(g, x, )

        enc_rep_nomask_reduced = self.dimReductionMLP2(F.relu(self.dimReductionMLP(enc_rep_nomask)))
        with torch.no_grad():
            edge_nodeID1, edge_nodeID2 = self.batch_top_k_cosine_similarity(enc_rep_nomask_reduced, 2048,
                                                                            mask_nodes, k=self.prompt_num)
        pre_use_g_add_edge = use_g.clone()
        pre_use_g_add_edge.add_edges(edge_nodeID2, edge_nodeID1)
        enc_rep, all_hidden = self.encoder(g, use_x, return_hidden=True)


        latent_target = self.projector_ema(enc_rep_nomask[keep_nodes])
        latent_pred = self.projector(enc_rep[keep_nodes])
        loss_latent = sce_loss(latent_pred, latent_target, self.alpha_l2)

        # mmd_loss = self.compute_mmd(latent_target, latent_pred, sigma=1.0)
        mmd_loss = self.compute_mmd(latent_target, latent_pred, sigma=1.0)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "liear"):
            recon = self.decoder(rep)
        else:
            recon = self.decoder(use_g, rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        # loss = self.criterion(x_rec, x_init) + self.loss_weight * loss_latent
        loss = self.criterion(x_rec, x_init) + self.loss_weight * loss_latent + 0.65 * mmd_loss
        self.ema_update()
        return loss
    def mask_attr_prediction_diff(self, g, x, difficulty, epoch):
        num_nodes = g.num_nodes()

        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise_diff_less_proportion(g, x, self._mask_rate,
                                                                                   difficulty, epoch, 1500)


        use_g = pre_use_g

        enc_rep_nomask = self.encoder_ema(g, x, )

        enc_rep_nomask_reduced = self.dimReductionMLP2(F.relu(self.dimReductionMLP(enc_rep_nomask)))
        with torch.no_grad():
            edge_nodeID1, edge_nodeID2 = self.batch_top_k_cosine_similarity(enc_rep_nomask_reduced, 2048,
                                                                            mask_nodes, k=self.prompt_num)
        pre_use_g_add_edge = use_g.clone()
        pre_use_g_add_edge.add_edges(edge_nodeID2, edge_nodeID1)
        enc_rep, all_hidden = self.encoder(g, use_x, return_hidden=True)


        latent_target = self.projector_ema(enc_rep_nomask[keep_nodes])
        latent_pred = self.projector(enc_rep[keep_nodes])
        loss_latent = sce_loss(latent_pred, latent_target, self.alpha_l2)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "liear"):
            recon = self.decoder(rep)
        else:
            recon = self.decoder(use_g, rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init) + self.loss_weight * loss_latent
        self.ema_update()
        return loss
    def mask_attr_prediction_edge(self, g, x):
        num_nodes = g.num_nodes()

        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)
        use_g = pre_use_g

        enc_rep_nomask = self.encoder_ema(g, x, )
        enc_rep_nomask_reduced = self.dimReductionMLP2(F.relu(self.dimReductionMLP(enc_rep_nomask)))
        with torch.no_grad():
            edge_nodeID1, edge_nodeID2 = self.batch_top_k_cosine_similarity(enc_rep_nomask_reduced, 2048,
                                                                            mask_nodes, k=self.prompt_num)
        pre_use_g_add_edge = use_g.clone()
        pre_use_g_add_edge.add_edges(edge_nodeID2, edge_nodeID1)
        enc_rep, all_hidden = self.encoder(pre_use_g_add_edge, use_x, return_hidden=True)

        ng, (dsrc, ddst) = self.drop_edge(g, 0.99, True)
        masked_edges = torch.stack((dsrc, ddst), dim=0)

        # aug_edge_index, _ = add_self_loops(g.edge())
        # neg_edges = self.negative_sampler(
        #     torch.stack(g.edges()),
        #     num_nodes=num_nodes,
        #     num_neg_samples=dsrc.size(0),
        # ).view_as(masked_edges)
        # ng_edge = torch.cat((dsrc, ddst), dim=0)
        with torch.no_grad():
            edge_nodeID1, edge_nodeID2 = self.batch_top_k_cosine_similarity(enc_rep_nomask_reduced, 2048,
                                                                            dsrc, k=1)
        ng.add_edges(edge_nodeID2, ddst)

        # z = self.encoder(ng, x)
        z = self.encoder(g, x)


        # ******************* loss for edge reconstruction *********************
        pos_out = self.edge_decoder(z, masked_edges, sigmoid=False)
        # neg_out = self.edge_decoder(z, neg_edges, sigmoid=False)
        # loss_edge = ce_loss(pos_out, neg_out)
        loss_edge = ce_loss(pos_out)


        latent_target = self.projector_ema(enc_rep_nomask[keep_nodes])
        latent_pred = self.projector(enc_rep[keep_nodes])
        loss_latent = sce_loss(latent_pred, latent_target, self.alpha_l2)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "liear"):
            recon = self.decoder(rep)
        else:
            recon = self.decoder(use_g, rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init) + self.loss_weight * loss_latent + self.loss_weight2 * loss_edge
        self.ema_update()
        return loss
    def mask_attr_prediction6(self, g, x):
        num_nodes = g.num_nodes()

        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)
        use_g = pre_use_g

        enc_rep_nomask = self.encoder_ema(g, x, )
        pre_use_g_add_edge = use_g.clone()

        # pre_use_g_add_edge.add_edges(edge_nodeID2, edge_nodeID1)

        enc_rep, all_hidden = self.encoder(pre_use_g_add_edge, use_x, return_hidden=True)

        latent_target = self.projector_ema(enc_rep_nomask[keep_nodes])
        latent_pred = self.projector(enc_rep[keep_nodes])


        loss_latent = sce_loss(latent_pred, latent_target, self.alpha_l2)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0


        if self._decoder_type in ("mlp", "liear"):
            recon = self.decoder(rep)
        else:
            recon = self.decoder(use_g, rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init) + self.loss_weight * loss_latent

        self.ema_update()
        return loss

    #消融实验
    def mask_attr_prediction_ablation(self, g, x):
        num_nodes = g.num_nodes()

        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)
        use_g = pre_use_g

        enc_rep_nomask = self.encoder_ema(g, x, )
        enc_rep_nomask_reduced = self.dimReductionMLP2(F.relu(self.dimReductionMLP(enc_rep_nomask)))  # 通过第一层并应用ReLU激活
        with torch.no_grad():
            edge_nodeID1, edge_nodeID2 = self.batch_top_k_cosine_similarity(enc_rep_nomask_reduced, 2048, mask_nodes, k=self.prompt_num)
        pre_use_g_add_edge = use_g.clone()

        pre_use_g_add_edge.add_edges(edge_nodeID2, edge_nodeID1)

        enc_rep, all_hidden = self.encoder(pre_use_g_add_edge, use_x, return_hidden=True)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0
        if self._decoder_type in ("mlp", "liear"):
            recon = self.decoder(rep)
        else:
            recon = self.decoder(use_g, rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)

        self.ema_update()
        return loss

    def mask_edge(self, graph, mask_prob):
        E = graph.num_edges()

        mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
        masks = torch.bernoulli(1 - mask_rates)
        mask_idx = masks.nonzero().squeeze(1)
        return mask_idx

    def drop_edge(self, graph, drop_rate, return_edges=False):
        if drop_rate <= 0:
            return graph

        n_node = graph.num_nodes()
        edge_mask = self.mask_edge(graph, drop_rate)
        src = graph.edges()[0]
        dst = graph.edges()[1]

        nsrc = src[edge_mask]
        ndst = dst[edge_mask]

        ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
        ng = ng.add_self_loop()

        dsrc = src[~edge_mask]
        ddst = dst[~edge_mask]

        if return_edges:
            return ng, (dsrc, ddst)
        return ng

    def ema_update(self):
        def update(student, teacher):
            with torch.no_grad():
            # m = momentum_schedule[it]  # momentum parameter
                m = self._momentum
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        update(self.encoder, self.encoder_ema)
        update(self.projector, self.projector_ema)
    def batch_top_k_cosine_similarity(self, x, batch_size, mask_nodes,  k=1):
        # 提取 mask_nodes 中的特征
        x_masked = x[mask_nodes]
        M = x_masked.shape[0]  # mask_nodes 中的节点数

        edge_nodeID1 = []
        edge_nodeID2 = []

        # 预先计算每个节点的特征模长，用于归一化
        x_norm = x / x.norm(dim=1, keepdim=True)
        x_masked_norm = x_masked / x_masked.norm(dim=1, keepdim=True)

        # 分批计算
        with torch.no_grad():
            for i in range(0, M, batch_size):

                end_i = min(i + batch_size, M)
                x_masked_batch = x_masked_norm[i:end_i]
                batch_node_ids = mask_nodes[i:end_i]

                # 批量余弦相似度计算：避免循环操作
                similarity_matrix = x_masked_batch @ x_norm.T

                # 筛选 top-k 相似节点（排除自身）
                top_k_values, top_k_indices = torch.topk(similarity_matrix, k=k + 1, dim=1)

                # 排除自身
                mask_self = (top_k_indices != batch_node_ids.view(-1, 1))
                all_true = mask_self.all(dim=1)

                # 针对全 True 行随机设置一个为 False
                random_indices = torch.randint(0, mask_self.size(1), (all_true.sum().item(),))
                mask_self[all_true.nonzero().squeeze(), random_indices] = False

                # 筛选最终 top-k 节点
                filtered_indices = top_k_indices[mask_self].view(similarity_matrix.size(0), -1)[:, :k]

                # 构建边的起点和终点
                edge_nodeID1.extend(batch_node_ids.repeat_interleave(k).tolist())
                edge_nodeID2.extend(filtered_indices.flatten().tolist())

        return edge_nodeID1, edge_nodeID2

    def batch_top_k_dot_similarity(self, x, batch_size, mask_nodes,  k=1):
        # 提取 mask_nodes 中的特征
        x_masked = x[mask_nodes]
        M = x_masked.shape[0]  # mask_nodes 中的节点数

        edge_nodeID1 = []
        edge_nodeID2 = []

        # 分批计算
        with torch.no_grad():
            for i in range(0, M, batch_size):
                end_i = min(i + batch_size, M)
                x_masked_batch = x_masked[i:end_i]
                batch_node_ids = mask_nodes[i:end_i]

                # 批量点积计算：计算 x_masked_batch 和 x 之间的点积
                similarity_matrix = x_masked_batch @ x.T  # 使用点积而非余弦相似度

                # 筛选 top-k 相似节点（排除自身）
                top_k_values, top_k_indices = torch.topk(similarity_matrix, k=k + 1, dim=1)

                # 排除自身
                mask_self = (top_k_indices != batch_node_ids.view(-1, 1))
                all_true = mask_self.all(dim=1)

                # 针对全 True 行随机设置一个为 False
                random_indices = torch.randint(0, mask_self.size(1), (all_true.sum().item(),))
                mask_self[all_true.nonzero().squeeze(), random_indices] = False

                # 筛选最终 top-k 节点
                filtered_indices = top_k_indices[mask_self].view(similarity_matrix.size(0), -1)[:, :k]

                # 构建边的起点和终点
                edge_nodeID1.extend(batch_node_ids.repeat_interleave(k).tolist())
                edge_nodeID2.extend(filtered_indices.flatten().tolist())

        return edge_nodeID1, edge_nodeID2
    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

    def reconstruct_adj_mse(self, g, emb):
        adj = g.adj().to_dense()
        adj = adj.to(emb.device)
        res_adj = (emb @ emb.t())
        res_adj = F.sigmoid(res_adj)
        relative_distance = (adj * res_adj).sum() / (res_adj * (1 - adj)).sum()
        cri = torch.nn.MSELoss()
        res_loss = cri(adj, res_adj) + F.binary_cross_entropy_with_logits(adj, res_adj)
        loss = res_loss + relative_distance

        return loss

    def std_loss(self, z):
        z = self.std_expander(z)
        z = F.normalize(z, dim=1)
        std_z = torch.sqrt(z.var(dim=0) + 1e-4)
        std_loss = torch.mean(F.relu(1 - std_z))
        return std_loss
    def gaussian_kernel(self, x, y, sigma=1.0):
        """
        计算高斯核函数
        :param x: 张量，形状为 (n_samples, n_features)
        :param y: 张量，形状为 (m_samples, n_features)
        :param sigma: 高斯核的带宽参数
        :return: 高斯核矩阵
        """
        x_norm = torch.sum(x**2, dim=1, keepdim=True)  # (n_samples, 1)
        y_norm = torch.sum(y**2, dim=1, keepdim=True)  # (m_samples, 1)
        dist = x_norm + y_norm.T - 2 * torch.matmul(x, y.T)  # (n_samples, m_samples)
        return torch.exp(-dist / (2 * sigma**2))

    def compute_mmd(self, x, y, sigma=1.0):
        """
        计算最大均值差异（MMD）
        :param x: 张量，形状为 (n_samples, n_features)
        :param y: 张量，形状为 (m_samples, n_features)
        :param sigma: 高斯核的带宽参数
        :return: MMD 值
        """
        # 计算核矩阵
        k_xx = self.gaussian_kernel(x, x, sigma)
        k_yy = self.gaussian_kernel(y, y, sigma)
        k_xy = self.gaussian_kernel(x, y, sigma)

        # 计算 MMD
        n = x.shape[0]
        m = y.shape[0]
        mmd = (k_xx.sum() / (n * n) +  # 第一项
              k_yy.sum() / (m * m) -  # 第二项
              2 * k_xy.sum() / (n * m))  # 第三项
        return mmd
def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    else:
        raise ValueError("Unknown activation")

def ce_loss(pos_out):
    pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
    # neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
    # return pos_loss + neg_loss
    return pos_loss
class EdgeDecoder(nn.Module):
    """Simple MLP Edge Decoder"""

    def __init__(
        self, in_channels, hidden_channels, out_channels=1,
        num_layers=2, dropout=0.5, activation='relu'
    ):

        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, z, edge, sigmoid=True, reduction=False):
        x = z[edge[0]] * z[edge[1]]

        if reduction:
            x = x.mean(1)

        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)

        x = self.mlps[-1](x)

        if sigmoid:
            return x.sigmoid()
        else:
            return x

