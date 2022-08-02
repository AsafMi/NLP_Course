from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments

from consts import *


@dataclass
class ProjectTrainingArguments(TrainingArguments):
    #
    # output_dir: str = field(
    #     default='test_trainer',
    #     metadata={"help": "output_dir"}
    # )
    #
    # evaluation_strategy: str = field(
    #     default='epoch',
    #     metadata={"help": "epoch"}
    # )
    pass
