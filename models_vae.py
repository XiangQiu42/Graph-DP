import torch
import torch.nn as nn
from layers import GraphConvolution, GraphAggregation, MultiGraphConvolutionLayers, MultiDenseLayer

import numpy as np
import util_dir.mode_graphvae as model_graphvae
import torch.nn.init as init
import torch.nn.functional as F
import scipy.optimize
from torch.autograd import Variable


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout_rate):
        super(Generator, self).__init__()
        self.multi_dense_layer = MultiDenseLayer(z_dim, conv_dims, torch.nn.Tanh())

        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes

        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropoout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.multi_dense_layer(x)
        edges_logits = self.edges_layer(output).view(-1, self.edges, self.vertexes, self.vertexes)
        edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2)) / 2
        edges_logits = self.dropoout(edges_logits.permute(0, 2, 3, 1))

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropoout(nodes_logits.view(-1, self.vertexes, self.nodes))

        return edges_logits, nodes_logits


class EncoderVAE(nn.Module):
    """VAE encoder sharing part."""

    def __init__(self, conv_dim, m_dim, b_dim, z_dim, with_features=False, f_dim=0, dropout_rate=0.):
        super(EncoderVAE, self).__init__()

        graph_conv_dim, aux_dim, linear_dim = conv_dim
        # discriminator
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, b_dim, with_features, f_dim, dropout_rate)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1] + m_dim, aux_dim, torch.nn.Tanh(), with_features, f_dim,
                                          dropout_rate)
        self.multi_dense_layer = MultiDenseLayer(aux_dim, linear_dim, torch.nn.Tanh(), dropout_rate=dropout_rate)
        self.emb_mean = nn.Linear(linear_dim[-1], z_dim)
        self.emb_logvar = nn.Linear(linear_dim[-1], z_dim)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, adj, hidden, node, activation=None):
        adj = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        h = self.gcn_layer(node, adj, hidden)
        h = self.agg_layer(h, node, hidden)
        h = self.multi_dense_layer(h)
        h_mu = self.emb_mean(h)
        h_logvar = self.emb_logvar(h)
        h = self.reparameterize(h_mu, h_logvar)
        return h, h_mu, h_logvar


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, conv_dim, m_dim, b_dim, dropout):
        super(Discriminator, self).__init__()

        graph_conv_dim, aux_dim, linear_dim = conv_dim
        # discriminator
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, b_dim)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1] + m_dim, aux_dim, torch.nn.Tanh())
        self.multi_dense_layer = MultiDenseLayer(aux_dim, linear_dim, torch.nn.Tanh())

        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, adj, hidden, node, activation=None):
        adj = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        h = self.gcn_layer(node, adj, hidden)
        h = self.agg_layer(h, node, hidden)
        h = self.multi_dense_layer(h)

        output = self.output_layer(h)
        output = activation(output) if activation is not None else output

        return output, h


class GraphVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, max_num_nodes, pool='sum'):
        """
        Args:
            input_dim: input feature dimension for node.
            hidden_dim: hidden dim for 2-layer gcn.
            latent_dim: dimension of the latent representation of graph.
        """
        super(GraphVAE, self).__init__()
        self.conv1 = model_graphvae.GraphConv(input_dim=input_dim, output_dim=hidden_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.conv2 = model_graphvae.GraphConv(input_dim=hidden_dim, output_dim=hidden_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.act = nn.ReLU()

        output_dim = max_num_nodes * (max_num_nodes + 1) // 2
        # self.vae = model.MLP_VAE_plain(hidden_dim, latent_dim, output_dim)
        self.vae = model_graphvae.MLP_VAE_plain(input_dim * input_dim, latent_dim, output_dim)
        # self.feature_mlp = model.MLP_plain(latent_dim, latent_dim, output_dim)

        self.max_num_nodes = max_num_nodes
        for m in self.modules():
            if isinstance(m, model_graphvae.GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.pool = pool

    def recover_adj_lower(self, l):
        # NOTE: Assumes 1 per minibatch
        adj = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        adj[torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1] = l
        return adj

    def recover_full_adj_from_lower(self, lower):
        diag = torch.diag(torch.diag(lower, 0))
        return lower + torch.transpose(lower, 0, 1) - diag

    def edge_similarity_matrix(self, adj, adj_recon, matching_features,
                               matching_features_recon, sim_func):
        S = torch.zeros(self.max_num_nodes, self.max_num_nodes,
                        self.max_num_nodes, self.max_num_nodes)
        for i in range(self.max_num_nodes):
            for j in range(self.max_num_nodes):
                if i == j:
                    for a in range(self.max_num_nodes):
                        S[i, i, a, a] = adj[i, i] * adj_recon[a, a] * \
                                        sim_func(matching_features[i], matching_features_recon[a])
                        # with feature not implemented
                        # if input_features is not None:
                else:
                    for a in range(self.max_num_nodes):
                        for b in range(self.max_num_nodes):
                            if b == a:
                                continue
                            S[i, j, a, b] = adj[i, j] * adj[i, i] * adj[j, j] * \
                                            adj_recon[a, b] * adj_recon[a, a] * adj_recon[b, b]
        return S

    def mpm(self, x_init, S, max_iters=50):
        x = x_init
        for it in range(max_iters):
            x_new = torch.zeros(self.max_num_nodes, self.max_num_nodes)
            for i in range(self.max_num_nodes):
                for a in range(self.max_num_nodes):
                    x_new[i, a] = x[i, a] * S[i, i, a, a]
                    pooled = [torch.max(x[j, :] * S[i, j, a, :])
                              for j in range(self.max_num_nodes) if j != i]
                    neigh_sim = sum(pooled)
                    x_new[i, a] += neigh_sim
            norm = torch.norm(x_new)
            x = x_new / norm
        return x

    def deg_feature_similarity(self, f1, f2):
        return 1 / (abs(f1 - f2) + 1)

    def permute_adj(self, adj, curr_ind, target_ind):
        ''' Permute adjacency matrix.
          The target_ind (connectivity) should be permuted to the curr_ind position.
        '''
        # order curr_ind according to target ind
        ind = np.zeros(self.max_num_nodes, dtype=np.int)
        ind[target_ind] = curr_ind
        adj_permuted = torch.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_permuted[:, :] = adj[ind, :]
        adj_permuted[:, :] = adj_permuted[:, ind]
        return adj_permuted

    def pool_graph(self, x):
        if self.pool == 'max':
            out, _ = torch.max(x, dim=1, keepdim=False)
        elif self.pool == 'sum':
            out = torch.sum(x, dim=1, keepdim=False)
        return out

    def forward(self, input_features, adj):
        # x = self.conv1(input_features, adj)
        # x = self.bn1(x)
        # x = self.act(x)
        # x = self.conv2(x, adj)
        # x = self.bn2(x)

        # pool over all nodes
        # graph_h = self.pool_graph(x)
        graph_h = input_features.view(-1, self.max_num_nodes * self.max_num_nodes)
        # vae
        h_decode, z_mu, z_lsgms = self.vae(graph_h)
        out = F.sigmoid(h_decode)
        out_tensor = out.cpu().data
        recon_adj_lower = self.recover_adj_lower(out_tensor)
        recon_adj_tensor = self.recover_full_adj_from_lower(recon_adj_lower)

        # set matching features be degree
        out_features = torch.sum(recon_adj_tensor, 1)

        adj_data = adj.cpu().data[0]
        adj_features = torch.sum(adj_data, 1)

        S = self.edge_similarity_matrix(adj_data, recon_adj_tensor, adj_features, out_features,
                                        self.deg_feature_similarity)

        # initialization strategies
        init_corr = 1 / self.max_num_nodes
        init_assignment = torch.ones(self.max_num_nodes, self.max_num_nodes) * init_corr
        # init_assignment = torch.FloatTensor(4, 4)
        # init.uniform(init_assignment)
        assignment = self.mpm(init_assignment, S)
        # print('Assignment: ', assignment)

        # matching
        # use negative of the assignment score since the alg finds min cost flow
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(-assignment.numpy())
        print('row: ', row_ind)
        print('col: ', col_ind)
        # order row index according to col index
        # adj_permuted = self.permute_adj(adj_data, row_ind, col_ind)
        adj_permuted = adj_data
        adj_vectorized = adj_permuted[torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1].squeeze_()
        adj_vectorized_var = Variable(adj_vectorized).cuda()

        # print(adj)
        # print('permuted: ', adj_permuted)
        # print('recon: ', recon_adj_tensor)
        adj_recon_loss = self.adj_recon_loss(adj_vectorized_var, out[0])
        print('recon: ', adj_recon_loss)
        print(adj_vectorized_var)
        print(out[0])

        loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
        loss_kl /= self.max_num_nodes * self.max_num_nodes  # normalize
        print('kl: ', loss_kl)

        loss = adj_recon_loss + loss_kl

        return loss

    def forward_test(self, input_features, adj):
        self.max_num_nodes = 4
        adj_data = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        adj_data[:4, :4] = torch.FloatTensor([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]])
        adj_features = torch.Tensor([2, 3, 3, 2])

        adj_data1 = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        adj_data1 = torch.FloatTensor([[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
        adj_features1 = torch.Tensor([3, 3, 2, 2])
        S = self.edge_similarity_matrix(adj_data, adj_data1, adj_features, adj_features1,
                                        self.deg_feature_similarity)

        # initialization strategies
        init_corr = 1 / self.max_num_nodes
        init_assignment = torch.ones(self.max_num_nodes, self.max_num_nodes) * init_corr
        # init_assignment = torch.FloatTensor(4, 4)
        # init.uniform(init_assignment)
        assignment = self.mpm(init_assignment, S)
        # print('Assignment: ', assignment)

        # matching
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(-assignment.numpy())
        print('row: ', row_ind)
        print('col: ', col_ind)

        permuted_adj = self.permute_adj(adj_data, row_ind, col_ind)
        print('permuted: ', permuted_adj)

        adj_recon_loss = self.adj_recon_loss(permuted_adj, adj_data1)
        print(adj_data1)
        print('diff: ', adj_recon_loss)

    def adj_recon_loss(self, adj_truth, adj_pred):
        return F.binary_cross_entropy(adj_truth, adj_pred)
