"""Worldcup metrics."""

from typing import Optional, Union
from seq2seq.metrics.worldcup.worldcup_exec_res_match import compute_exec_res_match
import datasets


_DESCRIPTION = """
Worldcup metrics.
"""

_KWARGS_DESCRIPTION = """
"""

_CITATION = """
"""

_URL = ""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Worldcup(datasets.Metric):
    def __init__(
        self,
        config_name: Optional[str] = None,
        keep_in_memory: bool = False,
        cache_dir: Optional[str] = None,
        num_process: int = 1,
        process_id: int = 0,
        seed: Optional[int] = None,
        experiment_id: Optional[str] = None,
        max_concurrent_cache_files: int = 10000,
        timeout: Union[int, float] = 100,
        **kwargs
    ):
        super().__init__(
            config_name=config_name,
            keep_in_memory=keep_in_memory,
            cache_dir=cache_dir,
            num_process=num_process,
            process_id=process_id,
            seed=seed,
            experiment_id=experiment_id,
            max_concurrent_cache_files=max_concurrent_cache_files,
            timeout=timeout,
            **kwargs
        )
        self.test_suite_db_dir: Optional[str] = kwargs.pop("test_suite_db_dir", None)

    def _info(self):
        if self.config_name not in [
            "exec_match"
        ]:
            raise KeyError(
                "You should supply a configuration name selected in " '["exec_match"]'
            )
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": {
                        "query": datasets.Value("string"),
                        "question": datasets.Value("string"),
                        "context": datasets.Value("string"),
                        "label": datasets.Value("string"),
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
                        "db_foreign_keys": datasets.features.Sequence(
                            {
                                "column_id": datasets.Value("int32"),
                                "other_column_id": datasets.Value("int32"),
                            }
                        ),
                    },
                }
            ),
            reference_urls=[_URL],
        )

    def _compute(self, predictions, references):
        if self.config_name == "exec_match":
            exec_match = compute_exec_res_match(predictions, references)
        else:
            exec_match = dict()

        return {**exec_match}
