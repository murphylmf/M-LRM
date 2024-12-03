from dataclasses import dataclass
import importlib

import torch.nn as nn
from omegaconf import OmegaConf

from lrm.utils.misc import get_device, load_module_weights
from lrm.utils.typing import *


def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    scfg = OmegaConf.merge(OmegaConf.structured(fields), cfg)
    return scfg


def find_class(cls_string):
    module_string = ".".join(cls_string.split(".")[:-1])
    cls_name = cls_string.split(".")[-1]
    module = importlib.import_module(module_string, package=None)
    cls = getattr(module, cls_name)
    return cls


class Configurable:
    @dataclass
    class Config:
        pass

    def __init__(self, cfg: Optional[dict] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)


class Updateable:
    def do_update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ):
        for attr in self.__dir__():
            if attr.startswith("_"):
                continue
            try:
                module = getattr(self, attr)
            except:
                continue  # ignore attributes like property, which can't be retrived using getattr?
            if isinstance(module, Updateable):
                module.do_update_step(
                    epoch, global_step, on_load_weights=on_load_weights
                )
        self.update_step(epoch, global_step, on_load_weights=on_load_weights)

    def do_update_step_end(self, epoch: int, global_step: int):
        for attr in self.__dir__():
            if attr.startswith("_"):
                continue
            try:
                module = getattr(self, attr)
            except:
                continue  # ignore attributes like property, which can't be retrived using getattr?
            if isinstance(module, Updateable):
                module.do_update_step_end(epoch, global_step)
        self.update_step_end(epoch, global_step)

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # override this method to implement custom update logic
        # if on_load_weights is True, you should be careful doing things related to model evaluations,
        # as the models and tensors are not guarenteed to be on the same device
        pass

    def update_step_end(self, epoch: int, global_step: int):
        pass


def update_if_possible(module: Any, epoch: int, global_step: int) -> None:
    if isinstance(module, Updateable):
        module.do_update_step(epoch, global_step)


def update_end_if_possible(module: Any, epoch: int, global_step: int) -> None:
    if isinstance(module, Updateable):
        module.do_update_step_end(epoch, global_step)


class BaseObject(Updateable):
    @dataclass
    class Config:
        pass

    cfg: Config  # add this to every subclass of BaseObject to enable static type checking

    def __init__(
        self, cfg: Optional[Union[dict, DictConfig]] = None, *args, **kwargs
    ) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self.device = get_device()
        self.configure(*args, **kwargs)

    def configure(self, *args, **kwargs) -> None:
        pass


class BaseModule(nn.Module, Updateable):
    @dataclass
    class Config:
        weights: Optional[str] = None
        freeze: Optional[bool] = False

    cfg: Config  # add this to every subclass of BaseModule to enable static type checking

    def __init__(
        self, cfg: Optional[Union[dict, DictConfig]] = None, *args, **kwargs
    ) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self.device = get_device()
        self._non_modules = {}
        self.configure(*args, **kwargs)
        if self.cfg.weights is not None:
            # format: path/to/weights:module_name
            if ":" in self.cfg.weights:
                weights_path, module_name = self.cfg.weights.split(":")
            else:
                weights_path, module_name = self.cfg.weights, None
            state_dict = load_module_weights(
                weights_path, module_name=module_name, map_location="cpu"
            )
            self.load_state_dict(state_dict, strict=False)
            # self.do_update_step(
            #     epoch, global_step, on_load_weights=True
            # )  # restore states

        if self.cfg.freeze:
            for params in self.parameters():
                params.requires_grad = False

    def configure(self, *args, **kwargs) -> None:
        pass

    def register_non_module(self, name: str, module: nn.Module) -> None:
        # non-modules won't be treated as model parameters
        if name in self._non_modules:
            raise ValueError(f"Non-module {name} already exists!")
        self._non_modules[name] = module

    def non_module(self, name: str):
        return self._non_modules.get(name, None)
