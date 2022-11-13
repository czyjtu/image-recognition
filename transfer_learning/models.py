from typing import Callable 
import torch as th 
import pretrainedmodels
from pathlib import Path 


def get_xception() -> th.nn.Module:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    model = pretrainedmodels.xception()
    model.last_linear = th.nn.Linear(2048, 3)
    model.last_linear
    return model 


def get_my_network() -> th.nn.Module:
    model_path = Path(__file__).absolute().parent / "pretrained/model8_gap"

    import cnn.models as cnn_models
    model: th.nn.Sequential = cnn_models.MODEL_REGISTRY["model8_gap"]()
    model.load_state_dict(th.load(model_path, map_location=th.device('cpu')))
    n_features = model[-2].in_features
    model[-2] = th.nn.Linear(n_features, 3)
    return model 


MODEL_REGISTRY: dict[str, Callable[[], th.nn.Module]] = {}

MODEL_REGISTRY["xception"] = get_xception
MODEL_REGISTRY["prev_lab"] = get_my_network
