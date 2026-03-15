# refer to https://huggingface.co/docs/datasets/dataset_script#create-a-dataset-loading-script
# https://huggingface.co/datasets/Rosenberg/CMeEE-V2
import os
import datasets
import json

# todo
_LABEL2ID = {
    'O': 0,
    'dis': 1,
    'sym': 2,
    'dru': 3,
    'equ': 4,
    'pro': 5,
    'bod': 6,
    'ite': 7,
    'mic': 8,
    'dep': 9,
}

logger = datasets.logging.get_logger(__name__)
_CITATION = """
@inproceedings{zhang-etal-2022-cblue,
    title = "{CBLUE}: A {C}hinese Biomedical Language Understanding Evaluation Benchmark",
    author = "Zhang, Ningyu  and
      Chen, Mosha  and
      Bi, Zhen  and
      Liang, Xiaozhuan  and
      Li, Lei  and
      Shang, Xin  and
      Yin, Kangping  and
      Tan, Chuanqi  and
      Xu, Jian  and
      Huang, Fei  and
      Si, Luo  and
      Ni, Yuan  and
      Xie, Guotong  and
      Sui, Zhifang  and
      Chang, Baobao  and
      Zong, Hui  and
      Yuan, Zheng  and
      Li, Linfeng  and
      Yan, Jun  and
      Zan, Hongying  and
      Zhang, Kunli  and
      Tang, Buzhou  and
      Chen, Qingcai",
    editor = "Muresan, Smaranda  and
      Nakov, Preslav  and
      Villavicencio, Aline",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.544/",
    doi = "10.18653/v1/2022.acl-long.544",
    pages = "7888--7915",
}
"""

_DESCRIPTION = """CMeEE-V2 dataset for NER task"""

_URL = f'./raw/'
_URLS = {
    "train": f'{_URL}/CMeEE-V2_train.json',
    "valid": f'{_URL}/CMeEE-V2_dev.json',
}

class CMeEEV2Config(datasets.BuilderConfig):
    """BuilderConfig for CMeEE-V2"""

    def __init__(self, **kwargs):
        """BuilderConfig for CMeEE-V2.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CMeEEV2Config, self).__init__(**kwargs)


class CMeEEV2(datasets.GeneratorBasedBuilder):
    """CMeEE-V2 dataset."""

    BUILDER_CONFIG_CLASS = CMeEEV2Config
    BUILDER_CONFIGS = [
        CMeEEV2Config(name="ner", version=datasets.Version("1.0.0"), description="CMeEE-V2 dataset for NER task"),
    ]
    DEFAULT_CONFIG_NAME = "ner"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "starts": datasets.Sequence(datasets.Value("int16")),  # start position for each entity
                    "ends": datasets.Sequence(datasets.Value("int16")),  # end position (excluded) for each entity
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                'O',
                                'dis',
                                'sym',
                                'dru',
                                'equ',
                                'pro',
                                'bod',
                                'ite',
                                'mic',
                                'dep',
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/cbluebenchmark/cblue",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> list[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["valid"]}),
        ]

    def _generate_examples(self, filepath):
        # todo, debug
        logger.info("⏳ Generating examples from = %s", filepath)
        guid = -1
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for line in data:  # an instance in a line
                guid += 1
                starts, ends, spans, ner_tags = [], [], [], []
                text, entities = line["text"], line["entities"]
                tokens = [char for char in text]
                for entity in entities:
                    # 'entities' shape like [{"end_idx": 7, "entity": "房室结消融", "start_idx": 3, "type": "pro" },..]
                    start, end, span, label = entity['start_idx'], entity['end_idx'], entity['entity'], entity['type']
                    starts.append(start)  # start position
                    ends.append(end + 1)  # end position (included -> excluded)
                    spans.append(span)
                    ner_tags.append(label)  # tag
                yield guid, {
                    "id": guid,
                    "tokens": tokens,
                    "starts": starts,
                    "ends": ends,
                    "spans": spans,
                    "ner_tags": ner_tags,
                }
