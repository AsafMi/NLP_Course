from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments

from consts import *


@dataclass
class ProjectTrainingArguments(TrainingArguments):
    pass
