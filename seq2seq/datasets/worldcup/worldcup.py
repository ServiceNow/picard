"""WorldCup: design space paper dataset"""

import json
import os
from typing import List, Generator, Any, Dict, Tuple
# import sys
# print(sys.path)
import datasets
from datasets.info import DatasetInfo
from datasets.utils.download_manager import DownloadManager
from seq2seq.utils.sql_database import SQLDatabase
from dotenv import dotenv_values
from itertools import product
db_config = dotenv_values('.env')

logger = datasets.logging.get_logger(__name__)

DB_SCHEMAS = ['exp_v1', 'exp_v2', 'exp_v3']

_VERSION = "1.1.0"
_CITATION = ""
_DESCRIPTION = "A real world world cup database in different data models"
_HOMEPAGE = ""
_LICENSE = ""
_CONFIGS = ['v1', 'v2', 'v3']
_TRAIN_SPLITS = ['100', '200', '300']

# _URL = "https://drive.google.com/uc?export=download&id=1kbMHZXHdX9s1Jq7tmhKiYNPH1quMpcbA"
_URL = "https://drive.google.com/uc?export=download&id=1kbMHZXHdX9s1Jq7tmhKiYNPH1quMpcbA"

def load_db_config(_schema=DB_SCHEMAS):
    host = db_config["WORLDCUP_CUP_DB_HOST"]
    port = db_config["WORLDCUP_CUP_DB_PORT"]
    database = db_config["WORLDCUP_CUP_DB_DATABASE"]
    username = db_config["WORLDCUP_CUP_DB_USERNAME"]
    password = db_config["WORLDCUP_CUP_DB_PASS"]
    database_uri = f'postgresql://{username}:{password}@{host}:{str(port)}/{database}'
    train_splits = _TRAIN_SPLITS
    res = locals()
    
    res['schema'] =  dict(((c, s) for c, s in zip(_CONFIGS, _schema)))
    return res

class WorldCupConfig(datasets.BuilderConfig):
    """BuilderConfig for World Cup datasets."""
    
    def __init__(self, data_dir, description, url, **kwargs):
        """
         Args:
            data_dir: `string`, the path to the folder containing the files in the
            downloaded .zip
            citation: `string`, citation for the data set
            url: `string`, url for information about the data set
            **kwargs: keyword arguments forwarded to super.
        """
        super(WorldCupConfig, self).__init__(
            version=datasets.Version(_VERSION, ""), **kwargs
        )
        self.data_dir = data_dir
        self.description = description
        self.url = url
        db_config_dict = load_db_config()
        self.db_uri = db_config_dict['database_uri']
        self.db_schema = db_config_dict['schema'][data_dir.split('/')[0]]
        
    

class WorldCup(datasets.GeneratorBasedBuilder):
    
    
    BUILDER_CONFIGS = [
        WorldCupConfig(
            name=f"worldcup_{config}_{train_split}",
            data_dir = f"{config}/{train_split}",
            description = f"World Cup Database {config} with {train_split} sampled trainning data",
            url = _URL
        ) for config, train_split in list(product(_CONFIGS, _TRAIN_SPLITS)) + [('v3', 859)]
    ]
    
    def __init__(self, *args, writer_batch_size = None, **kwargs) -> None:
        super().__init__(*args, writer_batch_size = writer_batch_size, **kwargs)
        db = SQLDatabase.from_uri(self.config.db_uri, schema=self.config.db_schema)
        self.schema_cache = {}
        self.schema_cache[self.config.db_schema] = db.transform_to_spider_schema_format(db.get_table_info_dict(with_col_details=False, do_sampling=False))
        # print(self.schema_cache)
        # self.include_train_others: bool = kwargs.pop("include_train_others", False)

        
    def _info(self) -> DatasetInfo:
        features = datasets.Features(
            {
                "query": datasets.Value("string"),
                "question": datasets.Value("string"),
                "db_id": datasets.Value("string"),
                "db_uri": datasets.Value("string"),
                "db_schema": datasets.Value("string"),
                "db_table_names": datasets.features.Sequence(datasets.Value("string")),
                "db_column_names": datasets.features.Sequence(
                    {
                        "table_id": datasets.Value("int32"),
                        "column_name": datasets.Value("string"),
                    }
                ),
                "db_column_types": datasets.features.Sequence(datasets.Value("string")),
                "db_primary_keys": datasets.features.Sequence({"column_id": datasets.Value("int32")}),
                "db_foreign_keys": datasets.features.Sequence(
                    {
                        "column_id": datasets.Value("int32"),
                        "other_column_id": datasets.Value("int32"),
                    }
                ),
            }
        )
        return datasets.DatasetInfo(
            description=self.config.description,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION
        )
        
    def _split_generators(self, dl_manager: DownloadManager) -> List[datasets.SplitGenerator]:
        downloaded_filepath = dl_manager.download_and_extract(url_or_urls=_URL)
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs = {
                    "data_filepaths": [
                        os.path.join(downloaded_filepath, f"worldcup/{self.config.data_dir}/train.json")
                    ],
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs = {
                    "data_filepaths": [
                        os.path.join(downloaded_filepath, f"worldcup/{self.config.data_dir}/dev.json")
                    ],
                }
            )
        ]
        
    def _generate_examples(self, data_filepaths: List[str]) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
        """This function return the examples in the raw (text) form."""
        for data_filepath in data_filepaths:
            logger.info("generating examples form = %s", data_filepath)
            with open(data_filepath, encoding='utf-8') as f:
                worldcup_data = json.load(f)
                for idx, sample in enumerate(worldcup_data):
                    assert any(self.schema_cache) and sample['db_id'] in list(self.schema_cache.keys())
                    schema = self.schema_cache[sample['db_id']]
                    yield idx, {
                        "query": sample['query'],
                        "question": sample['question'],
                        "db_id": sample['db_id'],
                        "db_uri": self.config.db_uri,
                        "db_table_names": schema["table_names_original"],
                        "db_schema": self.config.db_schema,
                        "db_column_names": [
                            {"table_id": table_id, "column_name": column_name}
                            for table_id, column_name in schema["column_names_original"]
                        ],
                        "db_column_types": schema["column_types"],
                        "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
                        "db_foreign_keys": [
                            {"column_id": column_id, "other_column_id": other_column_id}
                            for column_id, other_column_id in schema["foreign_keys"]
                        ],
                    }
