"""Data Processing tool functions

This module contains methods process data / class outputs.

The module structure is the following:

- The ``tensor_ConcatFromDict`` function to concatenate N tensors
  contained into a dictionnary with str as key arguments following
  the mentionned dimension as argument of the function

  Example:
  Dict = {'1' : torch.Tensor(1x3x10x10) , '2' : torch.Tensor(1x3x10x10)}

  tensor_ConcatFromDict(ds = Dict, inds = ['1','2'], dim = 0)
    |_ returning : torch.Tensor(2x3x10x10)

- ``DatabaseTorch`` implements is a derived class that use PyTorch to
  load images from a folder containing a dataset and convert those image
  into torch.Tensor format
"""

import torch
from typing import Dict, List

def tensor_ConcatFromDict(ds : Dict[str, torch.Tensor],
                          inds : List[str],
                          dim : int = 0) -> torch.Tensor:
    """Function performing torch.Tensor Concatenation

    Parameters
    ---------
    ds : Dict[str, torch.Tensor]
        The Collection of inputs.

    inds : List[str]
        The Selected keys which data will be concatenated.

    dim: int
        The dimension of concatenation.

    Returns
    -------
    _storage : torch.Tensor
        The resulting Tensor concatenated
    """
    # the storer
    _storage = ds[inds[0]]
    # the update / concatenation
    for indx in range(1,len(inds)):
        torch.cat((_storage, ds[inds[indx]]), dim)

    return _storage