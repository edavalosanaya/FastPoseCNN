import typing

import torch

class LogitData(typing.TypedDict, total=False):
    mask: torch.Tensor
    quaternion: torch.Tensor
    scales: torch.Tensor
    z: torch.Tensor
    xy: torch.Tensor

class CategoricalData(typing.TypedDict, total=False):
    mask: torch.Tensor
    quaternion: torch.Tensor
    scales: torch.Tensor
    z: torch.Tensor
    xy: torch.Tensor

class AggData(typing.TypedDict, total=False):
    # Meta Data
    class_ids: torch.Tensor
    sample_ids: torch.Tensor

    # Feature Data
    instance_masks: torch.Tensor
    quaternion: torch.Tensor
    scales: torch.Tensor
    z: torch.Tensor
    xy: torch.Tensor
    R: torch.Tensor
    T: torch.Tensor
    RT: torch.Tensor

class MatchedData(typing.TypedDict, total=False):
    # Meta Data
    class_ids: torch.Tensor
    sample_ids: torch.Tensor
    symmetric_ids: torch.Tensor

    instance_masks: torch.Tensor
    quaternion: torch.Tensor
    scales: torch.Tensor
    z: torch.Tensor
    xy: torch.Tensor
    R: torch.Tensor
    T: torch.Tensor
    RT: torch.Tensor