from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Union

import torch
from loguru import logger
from omegaconf import ListConfig, OmegaConf
from torch import Tensor, nn
from einops import rearrange

from yolo.config.config import ModelConfig, YOLOLayer
from yolo.tools.dataset_preparation import prepare_weight
from yolo.utils.module_utils import get_layer_map
from yolo.model.module import EarlyExitMultiheadDetection, MultiheadDetection, EarlyExitSampler


class YOLO(nn.Module):
    """
    A preliminary YOLO (You Only Look Once) model class still under development.

    Parameters:
        model_cfg: Configuration for the YOLO model. Expected to define the layers,
                   parameters, and any other relevant configuration details.
    """

    def __init__(self, model_cfg: ModelConfig, class_num: int = 80):
        super(YOLO, self).__init__()
        self.num_classes = class_num
        self.layer_map = get_layer_map()  # Get the map Dict[str: Module]
        self.model: List[YOLOLayer] = nn.ModuleList()
        self.reg_max = getattr(model_cfg.anchor, "reg_max", 16)
        self.build_model(model_cfg.model)
        self.model_cfg = model_cfg.model

        # For early stop
        self.confidence = model_cfg.early_exit.confidence
        if model_cfg.early_exit.dynamic == 'entropy':
            self.early_exit_func = self.early_exit_entropy
            self.confidence *= torch.log2(torch.tensor(class_num))
        elif model_cfg.early_exit.dynamic == 'confidence':
            self.early_exit_func = self.early_exit_confidence
        self.softmax = nn.Softmax(dim=-1)
        self.specified_layer = None
        if getattr(model_cfg.early_exit, "specified_layer", False):
            self.specified_layer = model_cfg.early_exit.specified_layer

        self.early_exit_layer_num = 0



    def add_early_exit(self):
        # E1
        param1 = {
            'in_channels':self.model[0].out_c,
            'out_channels': 256,
            'output_size': 80
        }
        param2 = {
            'in_channels':self.model[0].out_c,
            'out_channels': 512,
            'output_size': 40
        }
        param3 = {
            'in_channels':self.model[0].out_c,
            'out_channels': 1024,
            'output_size': 20
        }
        param4 = {
            'in_channels':[256, 512, 1024], 
            'num_classes': 80
        }
        sampler1 = self.create_layer('EarlyExitSampler', -1, {}, **param1)
        sampler2 = self.create_layer('EarlyExitSampler', 1, {}, **param2)
        sampler3 = self.create_layer('EarlyExitSampler', 1, {}, **param3)
        ex = self.create_layer('EarlyExitMultiheadDetection', [2, 3, 4], {'output': True}, **param4)
        sampler1.usable = True
        sampler2.usable = True
        sampler3.usable = True
        self.model.insert(1, ex)
        self.model.insert(1, sampler3)
        self.model.insert(1, sampler2)
        self.model.insert(1, sampler1)

        for layer in self.model:
            if layer.layer_type != 'EarlyExitSampler' and layer.layer_type != 'EarlyExitMultiheadDetection':
                if isinstance(layer.source, list):
                    new_source = []
                    for source in layer.source:
                        if source >= 2:
                            new_source.append(source + 4)
                        else:
                            new_source.append(source)
                    layer.source = new_source
                elif layer.source >= 2:
                    layer.source += 4

        # E2
        param1 = {
            'in_channels':self.model[5].out_c,
            'out_channels': 256,
            'output_size': 80
        }
        param2 = {
            'in_channels':self.model[5].out_c,
            'out_channels': 512,
            'output_size': 40
        }
        param3 = {
            'in_channels':self.model[5].out_c,
            'out_channels': 1024,
            'output_size': 20
        }
        param4 = {
            'in_channels':[256, 512, 1024], 
            'num_classes': 80
        }
        sampler1 = self.create_layer('EarlyExitSampler', 6, {}, **param1)
        sampler2 = self.create_layer('EarlyExitSampler', 6, {}, **param2)
        sampler3 = self.create_layer('EarlyExitSampler', 6, {}, **param3)
        ex = self.create_layer('EarlyExitMultiheadDetection', [7, 8, 9], {'output': True}, **param4)
        sampler1.usable = True
        sampler2.usable = True
        sampler3.usable = True
        self.model.insert(6, ex)
        self.model.insert(6, sampler3)
        self.model.insert(6, sampler2)
        self.model.insert(6, sampler1)

        for layer in self.model:
            if layer.layer_type != 'EarlyExitSampler' and layer.layer_type != 'EarlyExitMultiheadDetection':
                if isinstance(layer.source, list):
                    new_source = []
                    for source in layer.source:
                        if source >= 7:
                            new_source.append(source + 4)
                        else:
                            new_source.append(source)
                    layer.source = new_source
                elif layer.source >= 7:
                    layer.source += 4

        # E3
        param1 = {
            'in_channels':self.model[10].out_c,
            'out_channels': 256,
            'output_size': 80
        }
        param2 = {
            'in_channels':self.model[10].out_c,
            'out_channels': 512,
            'output_size': 40
        }
        param3 = {
            'in_channels':self.model[10].out_c,
            'out_channels': 1024,
            'output_size': 20
        }
        param4 = {
            'in_channels':[256, 512, 1024], 
            'num_classes': 80
        }
        sampler1 = self.create_layer('EarlyExitSampler', 11, {}, **param1)
        sampler2 = self.create_layer('EarlyExitSampler', 11, {}, **param2)
        sampler3 = self.create_layer('EarlyExitSampler', 11, {}, **param3)
        ex = self.create_layer('EarlyExitMultiheadDetection', [12, 13, 14], {'output': True}, **param4)
        sampler1.usable = True
        sampler2.usable = True
        sampler3.usable = True
        self.model.insert(11, ex)
        self.model.insert(11, sampler3)
        self.model.insert(11, sampler2)
        self.model.insert(11, sampler1)

        for layer in self.model:
            if layer.layer_type != 'EarlyExitSampler' and layer.layer_type != 'EarlyExitMultiheadDetection':
                if isinstance(layer.source, list):
                    new_source = []
                    for source in layer.source:
                        if source >= 12:
                            new_source.append(source + 4)
                        else:
                            new_source.append(source)
                    layer.source = new_source
                elif layer.source >= 12:
                    layer.source += 4

        # E4
        param1 = {
            'in_channels':self.model[16].out_c,
            'out_channels': 256,
            'output_size': 80
        }
        param2 = {
            'in_channels':self.model[16].out_c,
            'out_channels': 512,
            'output_size': 40
        }
        param3 = {
            'in_channels':self.model[16].out_c,
            'out_channels': 1024,
            'output_size': 20
        }
        param4 = {
            'in_channels':[256, 512, 1024], 
            'num_classes': 80
        }
        sampler1 = self.create_layer('EarlyExitSampler', 17, {}, **param1)
        sampler2 = self.create_layer('EarlyExitSampler', 17, {}, **param2)
        sampler3 = self.create_layer('EarlyExitSampler', 17, {}, **param3)
        ex = self.create_layer('EarlyExitMultiheadDetection', [18, 19, 20], {'output': True}, **param4)
        sampler1.usable = True
        sampler2.usable = True
        sampler3.usable = True
        self.model.insert(17, ex)
        self.model.insert(17, sampler3)
        self.model.insert(17, sampler2)
        self.model.insert(17, sampler1)

        for layer in self.model:
            if layer.layer_type != 'EarlyExitSampler' and layer.layer_type != 'EarlyExitMultiheadDetection':
                if isinstance(layer.source, list):
                    new_source = []
                    for source in layer.source:
                        if source >= 18:
                            new_source.append(source + 4)
                        else:
                            new_source.append(source)
                    layer.source = new_source
                elif layer.source >= 18:
                    layer.source += 4
        


    def build_model(self, model_arch: Dict[str, List[Dict[str, Dict[str, Dict]]]]):
        self.layer_index = {}
        output_dim, layer_idx = [3], 1
        logger.info(f"🚜 Building YOLO")
        # print("\n#############################################################")
        # print(model_arch)
        # print("#############################################################\n")
        for arch_name in model_arch:
            if model_arch[arch_name]:
                logger.info(f"  🏗️  Building {arch_name}")
            for layer_idx, layer_spec in enumerate(model_arch[arch_name], start=layer_idx):
                layer_type, layer_info = next(iter(layer_spec.items()))
                layer_args = layer_info.get("args", {})

                # Get input source
                source = self.get_source_idx(
                    layer_info.get("source", -1), layer_idx)

                # Find in channels
                if any(module in layer_type for module in ["Conv", "ELAN", "ADown", "AConv", "CBLinear"]):
                    layer_args["in_channels"] = output_dim[source]
                if "Detection" in layer_type or "Segmentation" in layer_type:
                    layer_args["in_channels"] = [output_dim[idx]
                                                 for idx in source]
                    layer_args["num_classes"] = self.num_classes
                    layer_args["reg_max"] = self.reg_max

                # create layers
                layer = self.create_layer(
                    layer_type, source, layer_info, **layer_args)
                self.model.append(layer)

                if layer.tags:
                    if layer.tags in self.layer_index:
                        raise ValueError(
                            f"Duplicate tag '{layer_info['tags']}' found.")
                    self.layer_index[layer.tags] = layer_idx

                out_channels = self.get_out_channels(
                    layer_type, layer_args, output_dim, source)
                output_dim.append(out_channels)
                setattr(layer, "out_c", out_channels)
            layer_idx += 1

    def forward(self, x):
        # print("\n#############################################################")
        # print(self.training)

        # print("#############################################################\n")

        if self.training:
            y = {0: x}
            output = []
            for index, layer in enumerate(self.model, start=1):
                
                if isinstance(layer.source, list):
                    model_input = [y[idx] for idx in layer.source]
                else:
                    model_input = y[layer.source]
                x = layer(model_input)

                # Get all of the outputs of heads
                if isinstance(layer, EarlyExitMultiheadDetection) or isinstance(layer, MultiheadDetection):
                    output.append(x)

                y[-1] = x
                if layer.usable:
                    y[index] = x

            return output
        else:
            early_exit_layer_counter = 0
            y = {0: x}
            output = dict()
            for index, layer in enumerate(self.model, start=1):
                # print("\n#############################################################")
                # print(layer, layer.source)
                # print(y.keys())
                # print(index)
                # print("#############################################################\n")
                if isinstance(layer.source, list):
                    model_input = [y[idx] for idx in layer.source]
                else:
                    model_input = y[layer.source]
                x = layer(model_input)

                # Gate of early exit
                if isinstance(layer, EarlyExitMultiheadDetection):
                    confidence = self.early_exit_func(x)
                    early_exit_layer_counter += 1

                    if self.specified_layer:
                        # print(self.specified_layer)
                        if early_exit_layer_counter == self.specified_layer:
                            print(
                                f"Early exit in {early_exit_layer_counter} of {self.early_exit_layer_num} early exit layer!")
                            return x

                    if confidence:
                        print(
                            f"Early exit in {early_exit_layer_counter} of {self.early_exit_layer_num} early exit layer!")
                        return x

                y[-1] = x
                if layer.usable:
                    y[index] = x
                if layer.output:
                    output[layer.tags] = x
            print("Exit in the last output layer!")
            return output

    def early_exit_entropy(self, x: Tensor) -> bool:
        entropy_list = []
        preds_cls = []
        for _, predict in enumerate(x):
            pred_cls, _, _ = predict
            preds_cls.append(rearrange(pred_cls, "B C h w -> B (h w) C"))
        preds_cls = torch.concat(preds_cls, dim=1)
        preds_cls = self.softmax(preds_cls)
        entropy = torch.mean(
            torch.sum((- preds_cls) * torch.log2(preds_cls), dim=1))
        # entropy_list.append(entropy)
        # entropy = torch.mean(torch.tensor(entropy_list))
        # print(entropy)
        if entropy < self.confidence:
            print(
                f"Current entropy: {entropy:.2f}   Threshold: {self.confidence:.2f}")
            return True
        return False

    def early_exit_confidence(self, x: Tensor) -> bool:
        preds_cls = []
        for _, predict in enumerate(x):
            pred_cls, _, _ = predict
            preds_cls.append(rearrange(pred_cls, "B C h w -> B (h w) C"))
        preds_cls = torch.concat(preds_cls, dim=1)
        preds_cls = self.softmax(preds_cls)
        confidence = torch.max(preds_cls, dim=-1)[0]
        confidence = torch.mean(confidence)
        if confidence > self.confidence:
            print(
                f"Current confidence: {confidence:.2f}   Threshold: {self.confidence:.2f}")
            return True
        return False

    def get_out_channels(self, layer_type: str, layer_args: dict, output_dim: list, source: Union[int, list]):
        if hasattr(layer_args, "out_channels"):
            return layer_args["out_channels"]
        if layer_type == "CBFuse":
            return output_dim[source[-1]]
        if isinstance(source, int):
            return output_dim[source]
        if isinstance(source, list):
            return sum(output_dim[idx] for idx in source)

    def get_source_idx(self, source: Union[ListConfig, str, int], layer_idx: int):
        if isinstance(source, ListConfig):
            return [self.get_source_idx(index, layer_idx) for index in source]
        if isinstance(source, str):
            source = self.layer_index[source]
        if source < -1:
            source += layer_idx
        if source > 0:  # Using Previous Layer's Output
            self.model[source - 1].usable = True
        return source

    def create_layer(self, layer_type: str, source: Union[int, list], layer_info: Dict, **kwargs) -> YOLOLayer:
        if layer_type in self.layer_map:
            layer = self.layer_map[layer_type](**kwargs)
            setattr(layer, "layer_type", layer_type)
            setattr(layer, "source", source)
            setattr(layer, "in_c", kwargs.get("in_channels", None))
            setattr(layer, "output", layer_info.get("output", False))
            setattr(layer, "tags", layer_info.get("tags", None))
            setattr(layer, "usable", 0)
            return layer
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    def save_load_weights(self, weights: Union[Path, OrderedDict]):
        """
        Update the model's weights with the provided weights.

        args:
            weights: A OrderedDict containing the new weights.
        """
        if isinstance(weights, Path):
            weights = torch.load(weights, map_location=torch.device("cpu"))
        if "model_state_dict" in weights:
            weights = weights["model_state_dict"]

        model_state_dict = self.model.state_dict()

        # TODO1: autoload old version weight
        # TODO2: weight transform if num_class difference

        error_dict = {"Mismatch": set(), "Not Found": set()}
        for model_key, model_weight in model_state_dict.items():
            if model_key not in weights:
                error_dict["Not Found"].add(tuple(model_key.split(".")[:-2]))
                continue
            if model_weight.shape != weights[model_key].shape:
                error_dict["Mismatch"].add(tuple(model_key.split(".")[:-2]))
                continue
            model_state_dict[model_key] = weights[model_key]

        for error_name, error_set in error_dict.items():
            for weight_name in error_set:
                logger.warning(
                    f"⚠️ Weight {error_name} for key: {'.'.join(weight_name)}")

        print("\n\n+++++++++++++++++++++++++++++++++++++++++++++++\n\n")
        self.model.load_state_dict(model_state_dict)
        # for i in model_state_dict.keys():
        #     print(i)
        # exit()


def frozen_weight(model):
    for param in model.parameters():
        param.requires_grad = False
    print("Model weights have been frozen!")


def add_early_exit(model):
    model.add_early_exit()
    model.early_exit_layer_num = sum(1 for layer in model.modules(
        ) if isinstance(layer, EarlyExitMultiheadDetection))
    print("Early exiting have been added!")


def create_model(model_cfg: ModelConfig, weight_path: Union[bool, Path] = True, class_num: int = 80) -> YOLO:
    """Constructs and returns a model from a Dictionary configuration file.

    Args:
        config_file (dict): The configuration file of the model.

    Returns:
        YOLO: An instance of the model defined by the given configuration.
    """
    OmegaConf.set_struct(model_cfg, False)
    model = YOLO(model_cfg, class_num)
    if weight_path:
        if weight_path == True:
            weight_path = Path("weights") / f"{model_cfg.name}.pt"
        elif isinstance(weight_path, str):
            weight_path = Path(weight_path)

        if not weight_path.exists():
            logger.info(f"🌐 Weight {weight_path} not found, try downloading")
            prepare_weight(weight_path=weight_path)
        if weight_path.exists():
            model.save_load_weights(weight_path)
            logger.info("✅ Success load model & weight")
    else:
        logger.info("✅ Success load model")

    frozen_weight(model)
    add_early_exit(model)
    # print(model.model[0:7])

    return model
