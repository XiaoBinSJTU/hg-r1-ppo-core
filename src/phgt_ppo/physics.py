"""Assumed serial-chain geometry and gravitational load mapping."""

import numpy as np
import torch


def _rotation(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def unit_load_map(poses, lengths, offset):
    maps, wrenches = [], []
    for pose in np.asarray(poses):
        rotation = np.eye(3)
        position = np.zeros(3)
        origins, axes = [], []
        for q, length, axis in zip(pose, lengths, [2, 1, 1, 0, 1, 0]):
            origins.append(position.copy())
            axes.append(rotation[:, axis].copy())
            rotation = rotation @ _rotation(axis, q)
            position = position + rotation @ np.array([length, 0., 0.])
        force = np.array([0., 0., -9.81])
        moment = np.cross(rotation @ np.array([offset, 0., 0.]), force)
        jacobian = np.stack([np.r_[np.cross(a, position - o), a]
                             for a, o in zip(axes, origins)], axis=1)
        maps.append(jacobian.T @ np.r_[force, moment])
        wrenches.append(np.r_[rotation.T @ force, rotation.T @ moment])
    return np.asarray(maps, dtype='float32'), np.asarray(wrenches, dtype='float32')


def incidence(device=None):
    h = torch.zeros(6, 7, device=device)
    roles = torch.zeros(6, 7, dtype=torch.long, device=device)
    for edge in range(5):
        h[:edge + 2, edge] = 1
        roles[0, edge] = 0
        roles[1:edge + 1, edge] = 1
        roles[edge + 1, edge] = 2
    h[[0, 1, 2], 5] = 1
    h[[2, 3, 4, 5], 6] = 1
    roles[:, 5:] = 1
    return h, roles
