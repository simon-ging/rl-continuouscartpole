from torch import optim
from torch.optim.rmsprop import RMSprop

from src.opt.adamw import AdamW


def get_optimizer(config_opt, parameters):
    name = config_opt["name"]
    if name == "adam":
        opt = optim.Adam(
            parameters, lr=config_opt["lr"],
            betas=(config_opt["momentum"], config_opt["beta2"]),
            eps=config_opt["eps"], weight_decay=config_opt["weight_decay"],
            amsgrad=config_opt["amsgrad"])
    elif name == "adamw":
        opt = AdamW(
            parameters, lr=config_opt["lr"],
            betas=(config_opt["momentum"], config_opt["beta2"]),
            eps=config_opt["eps"], weight_decay=config_opt["weight_decay"],
            amsgrad=config_opt["amsgrad"])
    elif name == "sgd":
        opt = optim.SGD(
            parameters, lr=config_opt["lr"], momentum=config_opt["momentum"],
            dampening=config_opt["dampening"],
            weight_decay=config_opt["weight_decay"],
            nesterov=config_opt["nesterov"])
    elif name == "rmsprop":
        opt = RMSprop(
            parameters, lr=config_opt["lr"], alpha=config_opt["alpha"],
            eps=config_opt["eps"], weight_decay=config_opt["weight_decay"],
            momentum=config_opt["momentum"], centered=config_opt["centered"])
    else:
        raise NotImplementedError("optimizer {} not implemented".format(name))
    return opt
