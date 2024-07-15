# refer to https://huggingface.co/docs/datasets/dataset_script#create-a-dataset-loading-script

import os
import datasets
import jsonlines

_LABEL2ID = {
    'O': 0,
    'PER': 1,
    'ORG': 2,
    'LOC': 3,
    'FAC': 4,
    'WEA': 5,
    'VEH': 6,
    'GPE': 7,
}

logger = datasets.logging.get_logger(__name__)
_CITATION = """
@misc{https://doi.org/10.35111/mwxc-vh88,
  doi = {10.35111/MWXC-VH88},
  url = {https://catalog.ldc.upenn.edu/LDC2006T06},
  author = {{Walker,  Christopher} and {Strassel,  Stephanie} and {Medero,  Julie} and {Maeda,  Kazuaki}},
  title = {ACE 2005 Multilingual Training Corpus},
  publisher = {Linguistic Data Consortium},
  year = {2006}
}
"""

_DESCRIPTION = """ACE 2005 dataset for NER task"""

_URL = f'./raw/'
_URLS = {
    "train": f'{_URL}/train.jsonl',
    "valid": f'{_URL}/valid.jsonl',
    "test": f'{_URL}/test.jsonl',
}

class Ace2005Config(datasets.BuilderConfig):
    """BuilderConfig for Ace2005"""

    def __init__(self, **kwargs):
        """BuilderConfig for Ace2005.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Ace2005Config, self).__init__(**kwargs)


class Ace2005(datasets.GeneratorBasedBuilder):
    """Ace2005 dataset."""

    BUILDER_CONFIG_CLASS = Ace2005Config
    BUILDER_CONFIGS = [
        Ace2005Config(name="ner", version=datasets.Version("1.0.0"), description="Ace2005 dataset for NER task"),
    ]
    DEFAULT_CONFIG_NAME = "ner"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                'O',
                                'PER',
                                'ORG',
                                'LOC',
                                'FAC',
                                'WEA',
                                'VEH',
                                'GPE',
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://catalog.ldc.upenn.edu/LDC2006T06",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> list[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["valid"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        # todo, debug
        logger.info("⏳ Generating examples from = %s", filepath)
        guid = -1
        with jsonlines.open(filepath) as reader:
            for line in reader:  # an article in a line
                for tags, tokens in zip(line["ners"], line["sentences"]):  # 'tags' shape like [[0, 0, "PER"], [6, 6, "PER"]]
                    guid += 1
                    ner_tags = [str(start), str(end+1), tag for e in tags]
                    for e in tags:
                        start, end, tag = e[0], e[1], e[2]
                        if start == end:
                            ner_tags[start] = _LABEL2ID[f'B-{tag}']
                        else:
                            ner_tags[start] = _LABEL2ID[f'B-{tag}']
                            for i in range(start + 1, end + 1):
                                if i < len(ner_tags):
                                    ner_tags[i] = _LABEL2ID[f'I-{tag}']
                    yield guid, {
                        "id": str(guid),
                        "tokens": tokens,
                        "ner_tags": ner_tags,
                    }
