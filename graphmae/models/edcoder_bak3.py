from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn

from .gin import GIN
from .gat import GAT
from .gcn import GCN
from .dot_gat import DotGAT
from .loss_func import sce_loss
from graphmae.utils import create_norm, drop_edge
import torch.nn.functional as F
from sklearn.random_projection import SparseRandomProjection
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
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

        # self.self_attention = nn.MultiheadAttention(embed_dim=num_hidden, num_heads=8, batch_first=True)
        self.loss_weight = nn.Parameter(torch.tensor(1.0))
        self.loss_weight2 = nn.Parameter(torch.tensor(1.0))
        self.loss_weight3 = nn.Parameter(torch.tensor(1.0))

        self.mse_loss_fn = nn.MSELoss()
        self.projector = nn.Sequential(
            nn.Linear(num_hidden, 256),
            nn.PReLU(),
            nn.Linear(256, num_hidden),
        )

        self.dimReductionMLP = nn.Linear(num_hidden, 256)  # 第一层线性层
        self.dimReductionMLP2 = nn.Linear(256, 128)  # 第二层线性层

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

        # use_g = g.clone()
        # # 添加新节点
        # new_node_feat = x[mask_nodes]
        # out_x = torch.cat((out_x, new_node_feat), dim=0)
        # use_g.add_nodes(len(mask_nodes), data={'feat': new_node_feat})
        # new_nodes_start_idx = num_nodes
        # neighbors_list = [use_g.successors(i) for i in mask_nodes]  # 邻居列表
        # all_neighbors = torch.cat(neighbors_list)  # 将所有邻居合并成一个张量
        #
        # # 生成所有 mask_nodes 对应的新节点的索引
        # num_neighbors = [len(neighbors) for neighbors in neighbors_list]  # 每个 mask_node 的邻居数量
        # new_node_indices = torch.arange(new_nodes_start_idx, new_nodes_start_idx + len(mask_nodes), device=g.device)
        #
        # # 扩展新节点的索引，使每个邻居对应正确的新节点索引
        # expanded_new_node_indices = torch.cat([torch.full((n,), new_node_indices[i], device=g.device)
        #                                        for i, n in enumerate(num_neighbors)])
        #
        # # 将邻居与新节点相连（双向边）
        # use_g.add_edges(all_neighbors, expanded_new_node_indices)
        # use_g.add_edges(expanded_new_node_indices, all_neighbors)
        # out_x[mask_nodes] = 0.0

        # out_x[token_nodes] += self.enc_mask_token

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()
        return use_g, out_x, (mask_nodes, keep_nodes)



    def forward(self, g, x):
        # ---- attribute reconstruction ----
        loss = self.mask_attr_prediction(g, x)
        # loss = self.mask_attr_prediction2(g, x)
        loss_item = {"loss": loss.item()}
        return loss, loss_item

    def batch_cosine_similarity(self, x, batch_size):
        N = x.shape[0]  # 总节点数
        similarity_matrix = torch.zeros((N, N)).to(x.device)  # 初始化相似度矩阵

        # 分块计算余弦相似度
        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)  # 处理最后不足 batch_size 的情况
            for j in range(0, N, batch_size):
                end_j = min(j + batch_size, N)  # 处理最后不足 batch_size 的情况
                # 对小块进行余弦相似度计算
                similarity_matrix[i:end_i, j:end_j] = F.cosine_similarity(
                    x[i:end_i].unsqueeze(1),
                    x[j:end_j].unsqueeze(0),
                    dim=-1
                )
        return similarity_matrix



    def random_projection(self, x, n_components=10):
        """
        使用随机投影对数据进行降维。

        参数:
            x (torch.Tensor): 原始输入数据，形状为 (num_samples, num_features)
            n_components (int): 要保留的维度（降维后的维度）

        返回:
            torch.Tensor: 降维后的数据，形状为 (num_samples, n_components)
        """
        # 将Torch Tensor 转换为 numpy array
        x_np = x.cpu().detach().numpy()

        # 使用 SparseRandomProjection 进行降维
        rp = SparseRandomProjection(n_components=n_components)
        x_reduced = rp.fit_transform(x_np)

        # 返回降维后的结果，并转换回 Torch Tensor
        return torch.tensor(x_reduced)

    def mask_attr_prediction2(self, g, x):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)

        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g

        enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "liear"):
            recon = self.decoder(rep)
        else:
            recon = self.decoder(pre_use_g, rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)
        return loss
    def mask_attr_prediction(self, g, x):
        num_nodes = g.num_nodes()
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)

        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g

        enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)

        enc_rep_nomask, _ = self.encoder(g, x, return_hidden=True)
        # enc_rep_nomask_reduced = self.random_projection(enc_rep_nomask, n_components=128)
        # enc_rep_nomask_reduced = enc_rep_nomask_reduced.to(device=x.device)
        enc_rep_nomask_reduced = self.dimReductionMLP2(F.relu(self.dimReductionMLP(enc_rep_nomask)))  # 通过第一层并应用ReLU激活
        # enc_rep_nomask_reduced = enc_rep_nomask

        similarity_matrix = self.batch_cosine_similarity(enc_rep_nomask, batch_size=2048)
        # similarity_matrix = F.cosine_similarity(enc_rep_nomask_reduced.unsqueeze(1), enc_rep_nomask_reduced.unsqueeze(0), dim=-1)
        # similarity_matrix = F.cosine_similarity(enc_rep_nomask.unsqueeze(1), enc_rep_nomask.unsqueeze(0), dim=-1)

        similarity_matrix = similarity_matrix.to(device=x.device)
        # Step 2: 为每个节点找到与其相似度最高的 k 个节点
        # similarity_matrix 是 (num_nodes, num_nodes) 的矩阵
        # 我们对每个节点取出相似度最高的 k 个节点 (排除自身)
        k = 1
        edge_nodeID1 = []
        edge_nodeID2 = []
        for i in mask_nodes:
            # 将第 i 个节点与所有其他节点的相似度按从高到低排序，得到 k 个最相似的节点
            top_k_indices = similarity_matrix[i].topk(k+1).indices  # k+1 是因为自身相似度为 1，需要排除自己
            top_k_indices = top_k_indices[top_k_indices != i][:k]  # 排除自己并只取前 k 个

            # 将当前节点与最相似的 k 个节点连接起来
            for j in top_k_indices:
                edge_nodeID1.append(i.item())  # 将 (i, j) 加入边列表
                edge_nodeID2.append(j.item())  # 将 (i, j) 加入边列表
        # batch_size = 1024  # 根据显存限制设定合适的 batch_size
        # num_nodes = similarity_matrix.size(0)

        # for batch_start in range(0, len(keep_nodes), batch_size):
        #     batch_nodes = keep_nodes[batch_start:batch_start + batch_size]
        #
        #     # 对于每批次节点，逐一计算相似度，并找到最相似的 k 个节点
        #     for i in batch_nodes:
        #         top_k_indices = similarity_matrix[i].topk(k + 1).indices
        #         top_k_indices = top_k_indices[top_k_indices != i][:k]
        #
        #         # 更新边的列表
        #         for j in top_k_indices:
        #             edge_nodeID1.append(i.item())
        #             edge_nodeID2.append(j.item())

        pre_use_g_add_edge = pre_use_g.clone()
        pre_use_g_add_edge.add_edges(edge_nodeID1, edge_nodeID2)
        enc_rep, all_hidden = self.encoder(pre_use_g_add_edge, use_x, return_hidden=True)

        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)
        #
        # enc_init = enc_rep_nomask[mask_nodes]
        # enc_rec = enc_rep[mask_nodes]
        # loss_graph = self.mse_loss_fn(enc_init, enc_rec)
        #
        # enc_rep_attention, _ = self.self_attention(enc_rep, enc_rep, enc_rep)
        # final_enc_rep = enc_rep_attention.clone()
        # final_enc_rep[keep_nodes] = enc_rep[keep_nodes]

        # rep_init = enc_rep[mask_nodes]
        # rep_rec = final_enc_rep[mask_nodes]
        #
        # loss_H = self.mse_loss_fn(rep_init, rep_rec)
        #
        # latent_pred1 = self.projector(enc_rep[mask_nodes])
        # latent_pred2 = self.projector(enc_rep_nomask[mask_nodes])
        # loss_projector = self.mse_loss_fn(latent_pred1, latent_pred2)

        # ---- attribute reconstruction ----
        # rep = self.encoder_to_decoder(final_enc_rep)
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "liear") :
            recon = self.decoder(rep)
        else:
            recon = self.decoder(pre_use_g, rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        # loss = self.criterion(x_rec, x_init) + self.loss_weight2 * loss_graph + self.loss_weight * loss_projector
        # loss = self.criterion(x_rec, x_init) + self.loss_weight * loss_H + self.loss_weight2 * loss_graph
        # loss = (self.criterion(x_rec, x_init) + self.loss_weight * loss_H + self.loss_weight2 * loss_graph
        #         + self.loss_weight3 * loss_projector)
        # loss = (self.criterion(x_rec, x_init) + self.loss_weight2 * loss_graph
        #         + self.loss_weight3 * loss_projector)
        # loss = self.criterion(x_rec, x_init) + self.loss_weight * loss_H
        loss = self.criterion(x_rec, x_init)
        return loss

    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
