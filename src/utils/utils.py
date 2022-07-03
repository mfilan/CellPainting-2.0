from typing import Any, Union

import numpy as np
import torch


def to_numpy(vector: Union[torch.Tensor, np.ndarray]) -> Any:
    if isinstance(vector, torch.Tensor):
        return vector.detach().cpu().numpy()
    return vector


def to_list(**kwargs):
    return list(kwargs.values())
