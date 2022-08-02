import random
import sys
from transformers import (
    HfArgumentParser,
    set_seed,
    AutoTokenizer
)


from consts import *
from args.data_args import DataTrainingArguments
from args.model_args import ModelArguments
from args.training_args import ProjectTrainingArguments
from args.da_args import DomainAdaptationArguments
from utils.utils import *
from utils.data_utils import *
from utils.train_utils import *


def train_model(data_args, model_args, training_args, src_datasets, unlabeled_datasets, target_datasets):
    # Load pretrained config tokenizer and model
    config = ModelArguments.config_name
    tokenizer = ModelArguments.tokenizer_name
    model = ModelArguments.model_name_or_path

    # Get compute metrics fn
    compute_metrics = 111

    # Initialize our Trainer
    trainer =111

    # Training
    if training_args.do_train:
        train_result = trainer.train()

        # Save results

    # Evaluation
    if training_args.do_eval:
        pass

    # Prediction
    if training_args.do_predict:
        pass
    return trainer


def main():
    parser = HfArgumentParser(
        (DataTrainingArguments, ModelArguments, ProjectTrainingArguments, DomainAdaptationArguments),
        description=DESCRIPTION,
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args, model_args, training_args, da_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        data_args, model_args, training_args, da_args = parser.parse_args_into_dataclasses()

    # Set extra arguments here

    # Setup logging

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load datasets
    src_datasets =111
    unlabeled_datasets =111
    target_datasets =111

    # Preprocess & Tokenize datasets


    # run training
    trainer = train_model(data_args, model_args, training_args, src_datasets, unlabeled_datasets, target_datasets)


if __name__ == "__main__":
    main()
