"""Role-aware physical hypergraph and causal relational temporal encoder."""

import math
import torch
from torch import nn
from .physics import incidence


class RoleHypergraph(nn.Module):
    def __init__(self, dimension, roles=True):
        super().__init__()
        h, r = incidence()
        self.register_buffer('h', h)
        self.register_buffer('roles', r)
        self.use_roles = roles
        self.transforms = nn.ModuleList([nn.Linear(dimension, dimension, bias=False) for _ in range(3)])
        self.edge_query = nn.Parameter(torch.randn(7, dimension) / math.sqrt(dimension))
        self.role_bias = nn.Parameter(torch.zeros(3))
        self.output = nn.Linear(dimension, dimension)
        self.norm = nn.LayerNorm(dimension)

    def forward(self, nodes, intensity):
        transformed = torch.stack([layer(nodes) for layer in self.transforms], dim=-2)
        roles = self.roles if self.use_roles else torch.zeros_like(self.roles)
        selector = torch.nn.functional.one_hot(roles, 3).float()
        values = torch.einsum('btnrd,ner->btned', transformed, selector)
        scores = torch.einsum('btned,ed->btne', values, self.edge_query)
        scores = scores / math.sqrt(nodes.shape[-1]) + self.role_bias[roles]
        scores = scores + .2 * intensity.abs().unsqueeze(-1)
        scores = scores.masked_fill(self.h[None, None] == 0, -1e4)
        attention = scores.softmax(dim=-2)
        edges = torch.einsum('btne,btned->bted', attention, values)
        reverse = self.h / self.h.sum(-1, keepdim=True).clamp_min(1)
        message = torch.einsum('ne,bted->btnd', reverse, edges)
        return self.norm(nodes + self.output(message))


class ForecastModel(nn.Module):
    def __init__(self, channels, cfg, method='PHGT', local=False, variant='full'):
        super().__init__()
        d = cfg['forecast']['dimension']
        self.method, self.local, self.variant = method, local, variant
        self.use_physics = local and method == 'PHGT' and variant != 'no_jacobian'
        assignment = torch.zeros(channels, 18)
        for c in range(channels):
            assignment[c, c % 18] = 1
        assignment /= assignment.sum(0).clamp_min(1)
        self.register_buffer('assignment', assignment)
        self.node_encoder = nn.Linear(6, d)
        self.hypergraph = RoleHypergraph(d, roles=variant != 'no_roles')
        self.static_layers = nn.ModuleList([nn.Linear(d, d) for _ in range(cfg['forecast']['relational_layers'])])
        self.dynamic_layers = nn.ModuleList([nn.Linear(d, d) for _ in range(cfg['forecast']['relational_layers'])])
        self.position = nn.Parameter(torch.randn(1, 16, d) * .02)
        layer = nn.TransformerEncoderLayer(d, cfg['forecast']['heads'], 2*d, dropout=.0,
                                          batch_first=True, norm_first=False, activation='gelu')
        self.temporal = nn.TransformerEncoder(layer, cfg['forecast']['temporal_layers'], enable_nested_tensor=False)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.multiscale = nn.ModuleList([nn.Conv1d(d, d, kernel_size=k) for k in (1, 3, 5)])
        self.action = nn.Linear(7, d)
        self.step_embedding = nn.Parameter(torch.randn(1, 4, d) * .02)
        self.decoder = nn.Sequential(nn.Linear(2*d, d), nn.SiLU(), nn.Linear(d, channels+7))

    def forward(self, batch, mean, scale):
        x, mask = batch['x'], batch['mask']
        b, t = x.shape[:2]
        node_values = (x @ self.assignment).reshape(b, t, 6, 3)
        node_mask = (mask @ self.assignment).reshape(b, t, 6, 3)
        nodes = self.node_encoder(torch.cat([node_values, node_mask], dim=-1))
        if self.method in ('PHGT', 'CHNN_proxy'):
            nodes = self.hypergraph(nodes, batch['hist_jac'] / 30)
        else:
            neighbor = (nodes + torch.roll(nodes, 1, dims=2) + torch.roll(nodes, -1, dims=2)) / 3
            nodes = nodes + neighbor
        sequence = nodes.mean(2)
        if self.method == 'PHGT' and self.variant != 'no_relational':
            for static, dynamic in zip(self.static_layers, self.dynamic_layers):
                previous = torch.cat([sequence[:, :1], sequence[:, :-1]], dim=1)
                score = sequence @ sequence.transpose(1, 2) / math.sqrt(sequence.shape[-1])
                causal = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
                score = score.masked_fill(causal, -1e4).softmax(-1)
                sequence = sequence + .2 * (static(previous) + dynamic(score @ sequence))
        if self.method in ('PHGT', 'PI_STHNN_proxy'):
            causal = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
            z = self.temporal(sequence + self.position, mask=causal)[:, -1]
        elif self.method == 'multiscale_graph_proxy':
            z = torch.stack([conv(torch.nn.functional.pad(sequence.transpose(1, 2), (conv.kernel_size[0]-1, 0)))[:, :, -1]
                             for conv in self.multiscale]).mean(0)
        else:
            z = self.gru(sequence)[0][:, -1]
        condition = self.action(torch.cat([batch['actions'].unsqueeze(-1), batch['jac'] / 30], dim=-1)) + self.step_embedding
        decoded = self.decoder(torch.cat([z[:, None].expand(-1, 4, -1), condition], dim=-1))
        prediction = x[:, -1:, :] + decoded[..., :x.shape[-1]]
        mass = torch.nn.functional.softplus(decoded[..., -7]) * 8
        raw_tau = decoded[..., -6:] * 100
        physical_tau = batch['jac'] * mass.unsqueeze(-1)
        tau = .5 * (raw_tau + physical_tau) if self.use_physics else raw_tau
        if self.use_physics:
            prediction = prediction.clone()
            prediction[..., 9:11] = .5 * (prediction[..., 9:11] + (physical_tau[..., 1:3] - mean[9:11]) / scale[9:11])
        return {'prediction': prediction, 'mass': mass, 'tau': tau,
                'raw_tau': raw_tau, 'physical_tau': physical_tau}
