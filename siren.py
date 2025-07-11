# Credit for the SIREN Code to https://github.com/scart97/Siren-fastai2/blob/master/siren.py

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import List
import math


def siren_init_tensor(tensor, use_this_fan_in=None):
  """ Siren initalization of a tensor. To initialize a nn.Module use 'apply_siren_init'. 
    It's equivalent to torch.nn.init.kaiming_uniform_ with mode = 'fan_in'
    and the same gain as the 'ReLU' nonlinearity """
  if use_this_fan_in is not None:
    fan_in = use_this_fan_in
  else:
    fan_in = nn.init._calculate_correct_fan(tensor, "fan_in")
  bound = math.sqrt(6.0 / fan_in)
  with torch.no_grad():
    return tensor.uniform_(-bound, bound)

def siren_init_model(layer: nn.Module):
  """ applies siren initialization to a layer """
  siren_init_tensor(layer.weight)
  if layer.bias is not None:
    fan_in = nn.init._calculate_correct_fan(layer.weight, "fan_in")
    siren_init_tensor(layer.bias, use_this_fan_in=fan_in)


class SirenActivation(nn.Module):
  """ Siren activation https://arxiv.org/abs/2006.09661 """

  def __init__(self, w0=1):
    """ w0 comes from the end of section 3
      it should be 30 for the first layer
      and 1 for the rest """
    super().__init__()
    self.w0 = torch.tensor(w0)

  def forward(self, x):
    return torch.sin(self.w0 * x)

  def extra_repr(self):
    return "w0={}".format(self.w0)


def SirenLayer(in_features, out_features, bias=True, w0=1):
  """ Siren Layer - it's a modified linear layer with sine activation """
  layer = nn.Sequential(nn.Linear(in_features, out_features, bias), SirenActivation(w0))
  siren_init_model(layer[0])
  return layer

def Siren(dimensions: List[int]):
  """ Siren model as presented in the paper. It's a sequence of linear layers followed by the Siren activation """
  first_layer = SirenLayer(dimensions[0], dimensions[1], w0=30)
  other_layers = []
  for dim0, dim1 in zip(dimensions[1:-1], dimensions[2:]):
    other_layers.append(SirenLayer(dim0, dim1))
  return nn.Sequential(first_layer, *other_layers)
