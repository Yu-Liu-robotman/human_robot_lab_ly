# Copyright (c) 2022-2024, fudeRobot All rights reserved.
#
#

import os
import yaml

ENV_CFG_FILENAME = '01_env_cfg.yaml'
TRAIN_CFG_FILENAME = '01_train_cfg.yaml'
SIM2SIMCFG_FILENAME = 'sim2sim_cfg.yaml'


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type) or type(attr).__module__ != 'builtins':
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def save_class_to_yaml_file(obj, filename):
    dct = class_to_dict(obj)
    if len(dct) == 0:
        return
    dctyaml = yaml.dump(dct)

    # 提取文件夹路径
    folder = os.path.dirname(filename)
    # 如果文件夹不存在，就创建它
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(filename, 'w') as f:
        f.write(dctyaml)


def update_class_from_yaml_file(obj, filename):
    with open(filename, 'r') as yaml_file:
        data = yaml.load(yaml_file, Loader=yaml.FullLoader)
        update_class_from_dict(obj, data)
