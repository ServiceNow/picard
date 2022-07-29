"""Spider metrics."""

from typing import Optional, Union
from sacrebleu.metrics import BLEU
from seq2seq.metrics.lc_quad.lc_quad_accuracy import compute_accuracy_metric
from seq2seq.metrics.lc_quad.lc_quad_sacrebleu import compute_sacrebleu_metric
from seq2seq.metrics.lc_quad.lc_quad_query_match import compute_query_match_metric
from seq2seq.metrics.lc_quad.lc_quad_f1 import compute_f1_metric
import datasets


_DESCRIPTION = """
lc-quad metrics.
"""

_KWARGS_DESCRIPTION = """
"""

_CITATION = """\
"""

_URL = "https://drive.google.com/uc?export=download&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0"


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class LC_QuAD(datasets.Metric):
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
        self.bleu = BLEU()

    def _info(self):

        print("metric.config_name: ", self.config_name)

        if self.config_name not in ["accuracy","bleu","both"]:
            
            raise KeyError(
                "You should supply a configuration name selected in bleu"
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
                    },
                }
            ),
            reference_urls=[_URL],
        )

    def _compute(self, predictions, references):
        if self.config_name == "both":
            query_match = compute_query_match_metric(predictions, references)
            f1 = compute_f1_metric(predictions, references)
            bleu = compute_sacrebleu_metric(predictions, references)
            acc = compute_accuracy_metric(predictions, references)
        else:
            query_match = dict()
            bleu = dict()
            acc = dict()
            f1 = dict()

        return {**query_match, **bleu, **acc, **f1}

