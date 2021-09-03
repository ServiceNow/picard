import pytest
from dataclasses import replace
from transformers.models.auto import AutoTokenizer
from transformers.training_args import TrainingArguments
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from seq2seq.utils.args import ModelArguments
from seq2seq.utils.dataset import DataArguments, DataTrainingArguments
from seq2seq.utils.dataset_loader import load_dataset


@pytest.fixture
def training_args(tmpdir, split) -> TrainingArguments:
    if split == "train":
        do_train = True
        do_eval = False
    elif split == "eval":
        do_train = False
        do_eval = True
    else:
        raise NotImplementedError()
    return TrainingArguments(
        output_dir=str(tmpdir),
        do_train=do_train,
        do_eval=do_eval,
        do_predict=False,
    )


@pytest.fixture(params=[True, False])
def schema_serialization_with_db_content(request) -> bool:
    return request.param


@pytest.fixture
def data_training_args(schema_serialization_with_db_content: bool) -> DataTrainingArguments:
    return DataTrainingArguments(
        max_source_length=4096,
        max_target_length=4096,
        schema_serialization_type="peteshaw",
        schema_serialization_with_db_id=True,
        schema_serialization_with_db_content=schema_serialization_with_db_content,
        normalize_query=True,
        target_with_db_id=True,
    )


@pytest.fixture(params=["cosql", "spider", "cosql+spider"])
def data_args(request) -> DataArguments:
    return DataArguments(dataset=request.param)


@pytest.fixture(params=["train", "eval"])
def split(request) -> str:
    return request.param


@pytest.fixture
def expected_max_input_ids_len(data_args: DataArguments, split: str, schema_serialization_with_db_content: bool) -> int:
    def _expected_max_input_ids_len(_data_args: DataArguments) -> int:
        if _data_args.dataset == "spider" and split == "train" and schema_serialization_with_db_content:
            return 1927
        elif _data_args.dataset == "spider" and split == "train" and not schema_serialization_with_db_content:
            return 1892
        elif _data_args.dataset == "spider" and split == "eval" and schema_serialization_with_db_content:
            return 468
        elif _data_args.dataset == "spider" and split == "eval" and not schema_serialization_with_db_content:
            return 468
        elif _data_args.dataset == "cosql" and split == "train" and schema_serialization_with_db_content:
            return 2229
        elif _data_args.dataset == "cosql" and split == "train" and not schema_serialization_with_db_content:
            return 1984
        elif _data_args.dataset == "cosql" and split == "eval" and schema_serialization_with_db_content:
            return 545
        elif _data_args.dataset == "cosql" and split == "eval" and not schema_serialization_with_db_content:
            return 545
        elif _data_args.dataset == "cosql+spider":
            return max(
                _expected_max_input_ids_len(_data_args=replace(_data_args, dataset="spider")),
                _expected_max_input_ids_len(_data_args=replace(_data_args, dataset="cosql")),
            )
        else:
            raise NotImplementedError()

    return _expected_max_input_ids_len(_data_args=data_args)


@pytest.fixture
def expected_max_labels_len(data_args: DataArguments, split: str, schema_serialization_with_db_content: bool) -> int:
    def _expected_max_labels_len(_data_args: DataArguments) -> int:
        if _data_args.dataset == "spider" and split == "train" and schema_serialization_with_db_content:
            return 250
        elif _data_args.dataset == "spider" and split == "train" and not schema_serialization_with_db_content:
            return 250
        elif _data_args.dataset == "spider" and split == "eval" and schema_serialization_with_db_content:
            return 166
        elif _data_args.dataset == "spider" and split == "eval" and not schema_serialization_with_db_content:
            return 166
        elif _data_args.dataset == "cosql" and split == "train" and schema_serialization_with_db_content:
            return 250
        elif _data_args.dataset == "cosql" and split == "train" and not schema_serialization_with_db_content:
            return 250
        elif _data_args.dataset == "cosql" and split == "eval" and schema_serialization_with_db_content:
            return 210
        elif _data_args.dataset == "cosql" and split == "eval" and not schema_serialization_with_db_content:
            return 210
        elif _data_args.dataset == "cosql+spider":
            return max(
                _expected_max_labels_len(_data_args=replace(_data_args, dataset="spider")),
                _expected_max_labels_len(_data_args=replace(_data_args, dataset="cosql")),
            )
        else:
            raise NotImplementedError()

    return _expected_max_labels_len(_data_args=data_args)


@pytest.fixture
def model_name_or_path() -> str:
    return "t5-small"


@pytest.fixture
def model_args(model_name_or_path: str) -> ModelArguments:
    return ModelArguments(model_name_or_path=model_name_or_path)


@pytest.fixture
def tokenizer(model_args: ModelArguments) -> PreTrainedTokenizerFast:
    return AutoTokenizer.from_pretrained(model_args.model_name_or_path)


def test_dataset_loader(
    data_args: DataArguments,
    split: str,
    expected_max_input_ids_len: int,
    expected_max_labels_len: int,
    training_args: TrainingArguments,
    data_training_args: DataTrainingArguments,
    model_args: ModelArguments,
    tokenizer: PreTrainedTokenizerFast,
) -> None:

    _metric, dataset_splits = load_dataset(
        data_args=data_args,
        model_args=model_args,
        data_training_args=data_training_args,
        training_args=training_args,
        tokenizer=tokenizer,
    )

    if split == "train":
        dataset = dataset_splits.train_split.dataset
    elif split == "eval":
        dataset = dataset_splits.eval_split.dataset
    elif split == "test":
        dataset = dataset_splits.test_split.dataset
    else:
        raise NotImplementedError()

    max_input_ids_len = max(len(item["input_ids"]) for item in dataset)
    assert max_input_ids_len == expected_max_input_ids_len

    max_labels_len = max(len(item["labels"]) for item in dataset)
    assert max_labels_len == expected_max_labels_len
