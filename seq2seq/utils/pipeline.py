from typing import Union, List, Dict, Optional
from transformers.pipelines.text2text_generation import Text2TextGenerationPipeline
from transformers.tokenization_utils import TruncationStrategy
from transformers.tokenization_utils_base import BatchEncoding
from third_party.spider.preprocess.get_tables import dump_db_json_schema
from seq2seq.utils.dataset import serialize_schema
from seq2seq.utils.spider import spider_get_input
from seq2seq.utils.cosql import cosql_get_input


class Text2SQLGenerationPipeline(Text2TextGenerationPipeline):
    """
    Pipeline for text-to-SQL generation using seq2seq models.

    The models that this pipeline can use are models that have been fine-tuned on the Spider text-to-SQL task.

    Usage::

        model = AutoModelForSeq2SeqLM.from_pretrained(...)
        tokenizer = AutoTokenizer.from_pretrained(...)
        db_path = ... path to "concert_singer" parent folder
        text2sql_generator = Text2SQLGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            db_path=db_path,
        )
        text2sql_generator(inputs="How many singers do we have?", db_id="concert_singer")
    """

    def __init__(self, *args, **kwargs):
        self.db_path: str = kwargs.pop("db_path")
        self.prefix: Optional[str] = kwargs.pop("prefix", None)
        self.normalize_query: bool = kwargs.pop("normalize_query", True)
        self.schema_serialization_type: str = kwargs.pop("schema_serialization_type", "peteshaw")
        self.schema_serialization_randomized: bool = kwargs.pop("schema_serialization_randomized", False)
        self.schema_serialization_with_db_id: bool = kwargs.pop("schema_serialization_with_db_id", True)
        self.schema_serialization_with_db_content: bool = kwargs.pop("schema_serialization_with_db_content", True)
        self.schema_cache: Dict[str, dict] = dict()
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        inputs: Union[str, List[str]],
        db_id: str,
        return_tensors: bool = False,
        return_text: bool = True,
        clean_up_tokenization_spaces: bool = False,
        truncation: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        **generate_kwargs,
    ):
        r"""
        Generate the output SQL expression(s) using text(s) given as inputs.

        Args:
            inputs (:obj:`str` or :obj:`List[str]`):
                Input text(s) for the encoder.
            db_id (:obj:`str`):
                The id of the targeted SQL database.
            return_tensors (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to include the tensors of predictions (as token indices) in the outputs.
            return_text (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to include the decoded texts in the outputs.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to clean up the potential extra spaces in the text output.
            truncation (:obj:`TruncationStrategy`, `optional`, defaults to :obj:`TruncationStrategy.DO_NOT_TRUNCATE`):
                The truncation strategy for the tokenization within the pipeline.
                :obj:`TruncationStrategy.DO_NOT_TRUNCATE` (default) will never truncate, but it is sometimes desirable
                to truncate the input to fit the model's max_length instead of throwing an error down the line.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework `here <./model.html#generative-models>`__).

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a dictionary with the following keys:

            - **generated_sql** (:obj:`str`, present when ``return_text=True``) -- The generated SQL.
            - **generated_token_ids** (:obj:`torch.Tensor` or :obj:`tf.Tensor`, present when ``return_tensors=True``)
              -- The token ids of the generated SQL.
        """
        assert return_tensors or return_text, "You must specify return_tensors=True or return_text=True"

        with self.device_placement():
            inputs = self._parse_and_tokenize(inputs, db_id=db_id, truncation=truncation)
            outputs = self._generate(inputs, return_tensors, return_text, clean_up_tokenization_spaces, generate_kwargs)
            for output in outputs:
                if "generated_text" in output:
                    output["generated_text"] = output["generated_text"].split("|", 1)[-1].strip()
            return outputs

    def _pre_process(self, input: str, db_id: str) -> str:
        prefix = self.prefix if self.prefix is not None else ""
        if db_id not in self.schema_cache:
            self.schema_cache[db_id] = get_schema(db_path=self.db_path, db_id=db_id)
        schema = self.schema_cache[db_id]
        if hasattr(self.model, "add_schema"):
            self.model.add_schema(db_id=db_id, db_info=schema)
        serialized_schema = serialize_schema(
            question=input,
            db_path=self.db_path,
            db_id=db_id,
            db_column_names=schema["db_column_names"],
            db_table_names=schema["db_table_names"],
            schema_serialization_type=self.schema_serialization_type,
            schema_serialization_randomized=self.schema_serialization_randomized,
            schema_serialization_with_db_id=self.schema_serialization_with_db_id,
            schema_serialization_with_db_content=self.schema_serialization_with_db_content,
            normalize_query=self.normalize_query,
        )
        return spider_get_input(question=input, serialized_schema=serialized_schema, prefix=prefix)

    def _parse_and_tokenize(
        self, inputs: Union[str, List[str]], db_id: str, truncation: TruncationStrategy
    ) -> BatchEncoding:
        if isinstance(inputs, list):
            assert (
                self.tokenizer.pad_token_id is not None
            ), "Please make sure that the tokenizer has a pad_token_id when using a batch input"
            inputs = [self._pre_process(input=input, db_id=db_id) + input for input in inputs]
            padding = True
        elif isinstance(inputs, str):
            inputs = self._pre_process(input=inputs, db_id=db_id)
            padding = False
        else:
            raise ValueError(
                f" `inputs`: {inputs} have the wrong format. The should be either of type `str` or type `list`"
            )
        inputs = self.tokenizer(
            inputs,
            return_tensors=self.framework,
            padding=padding,
            truncation=truncation,
        )
        # This is produced by tokenizers but is an invalid generate kwargs
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        return inputs


class ConversationalText2SQLGenerationPipeline(Text2TextGenerationPipeline):
    """
    Pipeline for conversational text-to-SQL generation using seq2seq models.

    The models that this pipeline can use are models that have been fine-tuned on the CoSQL SQL-grounded dialogue state tracking task.

    Usage::

        model = AutoModelForSeq2SeqLM.from_pretrained(...)
        tokenizer = AutoTokenizer.from_pretrained(...)
        db_path = ... path to "concert_singer" parent folder
        convText2sql_generator = ConversationalText2SQLGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            db_path=db_path,
        )
        convText2sql_generator(inputs=["How many singers do we have?"], db_id="concert_singer")
    """

    def __init__(self, *args, **kwargs):
        self.db_path: str = kwargs.pop("db_path")
        self.prefix: Optional[str] = kwargs.pop("prefix", None)
        self.normalize_query: bool = kwargs.pop("normalize_query", True)
        self.schema_serialization_type: str = kwargs.pop("schema_serialization_type", "peteshaw")
        self.schema_serialization_randomized: bool = kwargs.pop("schema_serialization_randomized", False)
        self.schema_serialization_with_db_id: bool = kwargs.pop("schema_serialization_with_db_id", True)
        self.schema_serialization_with_db_content: bool = kwargs.pop("schema_serialization_with_db_content", True)
        self.schema_cache: Dict[str, dict] = dict()
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        inputs: Union[List[str], List[List[str]]],
        db_id: str,
        return_tensors: bool = False,
        return_text: bool = True,
        clean_up_tokenization_spaces: bool = False,
        truncation: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        **generate_kwargs,
    ):
        r"""
        Generate the output SQL expression(s) using conversation(s) given as inputs.

        Args:
            inputs (:obj:`List[str]` or :obj:`List[List[str]]`):
                Input conversation(s) for the encoder.
            db_id (:obj:`str`):
                The id of the targeted SQL database.
            return_tensors (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to include the tensors of predictions (as token indices) in the outputs.
            return_text (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to include the decoded texts in the outputs.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to clean up the potential extra spaces in the text output.
            truncation (:obj:`TruncationStrategy`, `optional`, defaults to :obj:`TruncationStrategy.DO_NOT_TRUNCATE`):
                The truncation strategy for the tokenization within the pipeline.
                :obj:`TruncationStrategy.DO_NOT_TRUNCATE` (default) will never truncate, but it is sometimes desirable
                to truncate the input to fit the model's max_length instead of throwing an error down the line.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework `here <./model.html#generative-models>`__).

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a dictionary with the following keys:

            - **generated_sql** (:obj:`str`, present when ``return_text=True``) -- The generated SQL.
            - **generated_token_ids** (:obj:`torch.Tensor` or :obj:`tf.Tensor`, present when ``return_tensors=True``)
              -- The token ids of the generated SQL.
        """
        assert return_tensors or return_text, "You must specify return_tensors=True or return_text=True"

        with self.device_placement():
            inputs = self._parse_and_tokenize(inputs, db_id=db_id, truncation=truncation)
            outputs = self._generate(inputs, return_tensors, return_text, clean_up_tokenization_spaces, generate_kwargs)
            for output in outputs:
                if "generated_text" in output:
                    output["generated_text"] = output["generated_text"].split("|", 1)[-1].strip()
            return outputs

    def _pre_process(self, input: List[str], db_id: str) -> str:
        prefix = self.prefix if self.prefix is not None else ""
        if db_id not in self.schema_cache:
            self.schema_cache[db_id] = get_schema(db_path=self.db_path, db_id=db_id)
        schema = self.schema_cache[db_id]
        if hasattr(self.model, "add_schema"):
            self.model.add_schema(db_id=db_id, db_info=schema)
        serialized_schema = serialize_schema(
            question=" | ".join(input),
            db_path=self.db_path,
            db_id=db_id,
            db_column_names=schema["db_column_names"],
            db_table_names=schema["db_table_names"],
            schema_serialization_type=self.schema_serialization_type,
            schema_serialization_randomized=self.schema_serialization_randomized,
            schema_serialization_with_db_id=self.schema_serialization_with_db_id,
            schema_serialization_with_db_content=self.schema_serialization_with_db_content,
            normalize_query=self.normalize_query,
        )
        return cosql_get_input(utterances=input, serialized_schema=serialized_schema, prefix=prefix)

    def _parse_and_tokenize(
        self, inputs: Union[List[str], List[List[str]]], db_id: str, truncation: TruncationStrategy
    ) -> BatchEncoding:
        if all(isinstance(input, list) for input in inputs):
            assert (
                self.tokenizer.pad_token_id is not None
            ), "Please make sure that the tokenizer has a pad_token_id when using a batch input"
            inputs = [self._pre_process(input=input, db_id=db_id) + input for input in inputs]
            padding = True
        elif all(isinstance(input, str) for input in inputs):
            inputs = self._pre_process(input=inputs, db_id=db_id)
            padding = False
        else:
            raise ValueError(
                f" `inputs`: {inputs} have the wrong format. The should be lists with elements either of type `str` or type `list`"
            )
        inputs = self.tokenizer(
            inputs,
            return_tensors=self.framework,
            padding=padding,
            truncation=truncation,
        )
        # This is produced by tokenizers but is an invalid generate kwargs
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        return inputs


def get_schema(db_path: str, db_id: str) -> dict:
    schema = dump_db_json_schema(db_path + "/" + db_id + "/" + db_id + ".sqlite", db_id)
    return {
        "db_table_names": schema["table_names_original"],
        "db_column_names": {
            "table_id": [table_id for table_id, _ in schema["column_names_original"]],
            "column_name": [column_name for _, column_name in schema["column_names_original"]],
        },
        "db_column_types": schema["column_types"],
        "db_primary_keys": {"column_id": [column_id for column_id in schema["primary_keys"]]},
        "db_foreign_keys": {
            "column_id": [column_id for column_id, _ in schema["foreign_keys"]],
            "other_column_id": [other_column_id for _, other_column_id in schema["foreign_keys"]],
        },
    }