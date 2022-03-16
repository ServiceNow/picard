# Set up logging
import sys
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)

from typing import Optional
from dataclasses import dataclass, field
import os
import json
from contextlib import nullcontext
from alive_progress import alive_bar
from transformers.hf_argparser import HfArgumentParser
from transformers.models.auto import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from seq2seq.utils.pipeline import ConversationalText2SQLGenerationPipeline, Text2SQLGenerationPipeline, Text2SQLInput, ConversationalText2SQLInput
from seq2seq.utils.picard_model_wrapper import PicardArguments, PicardLauncher, with_picard
from seq2seq.utils.dataset import DataTrainingArguments


@dataclass
class PredictionOutputArguments:
    """
    Arguments pertaining to execution.
    """

    model_path: str = field(
        default="tscholak/cxmefzzi",
        metadata={"help": "Path to pretrained model"},
    )
    cache_dir: Optional[str] = field(
        default="/tmp",
        metadata={"help": "Where to cache pretrained models and data"},
    )
    db_path: str = field(
        default="database",
        metadata={"help": "Where to to find the sqlite files"},
    )
    inputs_path: str = field(default="data/dev.json", metadata={"help": "Where to find the inputs"})
    output_path: str = field(
        default="predicted_sql.txt", metadata={"help": "Where to write the output queries"}
    )
    device: int = field(
        default=0,
        metadata={
            "help": "Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU. A non-negative value will run the model on the corresponding CUDA device id."
        },
    )
    conversational: bool = field(default=False, metadata={"help": "Whether or not the inputs are conversations"})


def main():
    # See all possible arguments by passing the --help flag to this program.
    parser = HfArgumentParser((PicardArguments, PredictionOutputArguments, DataTrainingArguments))
    picard_args: PicardArguments
    prediction_output_args: PredictionOutputArguments
    data_training_args: DataTrainingArguments
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        picard_args, prediction_output_args, data_training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        picard_args, prediction_output_args, data_training_args = parser.parse_args_into_dataclasses()

    if os.path.isfile(prediction_output_args.output_path):
        raise RuntimeError("file `{}` already exists".format(prediction_output_args.output_path))

    # Initialize config
    config = AutoConfig.from_pretrained(
        prediction_output_args.model_path,
        cache_dir=prediction_output_args.cache_dir,
        max_length=data_training_args.max_target_length,
        num_beams=data_training_args.num_beams,
        num_beam_groups=data_training_args.num_beam_groups,
        diversity_penalty=data_training_args.diversity_penalty,
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        prediction_output_args.model_path,
        cache_dir=prediction_output_args.cache_dir,
        use_fast=True,
    )

    # Initialize Picard if necessary
    with PicardLauncher() if picard_args.launch_picard else nullcontext(None):
        # Get Picard model class wrapper
        if picard_args.use_picard:
            model_cls_wrapper = lambda model_cls: with_picard(
                model_cls=model_cls, picard_args=picard_args, tokenizer=tokenizer
            )
        else:
            model_cls_wrapper = lambda model_cls: model_cls

        # Initialize model
        model = model_cls_wrapper(AutoModelForSeq2SeqLM).from_pretrained(
            prediction_output_args.model_path,
            config=config,
            cache_dir=prediction_output_args.cache_dir,
        )

        if prediction_output_args.conversational:
            conversational_text2sql(model, tokenizer, prediction_output_args, data_training_args)
        else:
            text2sql(model, tokenizer, prediction_output_args, data_training_args)


def get_pipeline_kwargs(
    model, tokenizer: AutoTokenizer, prediction_output_args: PredictionOutputArguments, data_training_args: DataTrainingArguments
) -> dict:
    return {
        "model": model,
        "tokenizer": tokenizer,
        "db_path": prediction_output_args.db_path,
        "prefix": data_training_args.source_prefix,
        "normalize_query": data_training_args.normalize_query,
        "schema_serialization_type": data_training_args.schema_serialization_type,
        "schema_serialization_with_db_id": data_training_args.schema_serialization_with_db_id,
        "schema_serialization_with_db_content": data_training_args.schema_serialization_with_db_content,
        "device": prediction_output_args.device,
    }


def text2sql(model, tokenizer, prediction_output_args, data_training_args):
    # Initalize generation pipeline
    pipe = Text2SQLGenerationPipeline(**get_pipeline_kwargs(model, tokenizer, prediction_output_args, data_training_args))

    with open(prediction_output_args.inputs_path) as fp:
        questions = json.load(fp)

    with alive_bar(len(questions)) as bar:
        for question in questions:
            try:
                outputs = pipe(inputs=Text2SQLInput(question["question"],question["db_id"]))
                output = outputs[0]
                query = output["generated_text"]
            except Exception as e:
                logger.error(e)
                query = ""
            logger.info("writing `{}` to `{}`".format(query, prediction_output_args.output_path))
            bar.text(query)
            bar()
            with open(prediction_output_args.output_path, "a") as fp:
                fp.write(query + "\n")


def conversational_text2sql(model, tokenizer, prediction_output_args, data_training_args):
    # Initalize generation pipeline
    pipe = ConversationalText2SQLGenerationPipeline(
        **get_pipeline_kwargs(model, tokenizer, prediction_output_args, data_training_args)
    )

    with open(prediction_output_args.inputs_path) as fp:
        conversations = json.load(fp)

    length = sum(len(conversation["interaction"]) for conversation in conversations)
    with alive_bar(length) as bar:
        for conversation in conversations:
            utterances = []
            for turn in conversation["interaction"]:
                utterances.extend((utterance.strip() for utterance in turn["utterance"].split(sep="|")))
                try:
                    outputs = pipe(
                        inputs=ConversationalText2SQLInput(list(utterances),
                        db_id=conversation["database_id"])
                    )
                    output = outputs[0]
                    query = output["generated_text"]
                except Exception as e:
                    logger.error(e)
                    query = ""
                logger.info("writing `{}` to `{}`".format(query, prediction_output_args.output_path))
                bar.text(query)
                bar()
                with open(prediction_output_args.output_path, "a") as fp:
                    fp.write(query + "\n")

            with open(prediction_output_args.output_path, "a") as fp:
                fp.write("\n")


if __name__ == "__main__":
    main()
