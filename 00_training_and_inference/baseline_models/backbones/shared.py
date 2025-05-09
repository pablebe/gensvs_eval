import functools
import numpy as np

import torch
import torch.nn as nn

from baseline_models.util.registry import Registry


BackboneRegistry = Registry("Backbone")


