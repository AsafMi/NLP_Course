import pickle
import json
import os
import shutil

from consts import *


def save_to_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)


def load_from_json(path):
    with open(path, 'r') as f:
        obj = json.load(f)
    return obj


def remove_checkpoint(output_dir):
    for d in os.listdir(output_dir):
        if 'checkpoint' in d and os.path.isdir(os.path.join(output_dir, d)):
            shutil.rmtree(os.path.join(output_dir, d))
