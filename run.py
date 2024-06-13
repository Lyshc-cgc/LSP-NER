from module.annotation import Annotation
from module.processor import Processor
from module.func_util import get_config

def annotate(anno_cfg, api_cfg, labels_cfg, formated_dataset, **kwargs):
    anno = Annotation(anno_cfg, api_cfg, labels_cfg)
    anno.annotate_by_all(formated_dataset, quality=False, **kwargs)

def main():
    config = get_config('config.yml')
    # 1. pre-process the data
    dataset_name = 'ontonotes'  # 'conll', 'ontonotes'
    assert dataset_name in config['data_cfgs'].keys() # ('conll', 'ontonotes')

    data_cfg = get_config(config['data_cfgs'][dataset_name])
    labels_cfg = get_config(config['labels_cfg'])[dataset_name]
    proc = Processor(data_cfg, labels_cfg)
    formated_dataset = proc.process()

    # 2. annotate the data by LLMs
    # 2.1 api config
    use_api = False
    api_model = 'gpt'
    assert api_model in ('qwen', 'deepseek', 'glm', 'gpt')
    api_cfg = get_config(config['api_cfg'])[api_model] if use_api else None

    # 2.2 annotation prompt settings
    prompt_type = 'mt_few_shot'
    assert prompt_type in ('raw', 'single_type', 'mt_few_shot', 'few_shot', 'st_few_shot')
    anno_cfgs = config['anno_cfgs'][prompt_type]
    for anno_cfg in anno_cfgs:
        anno_cfg = get_config(anno_cfg)
        annotate(anno_cfg, api_cfg, labels_cfg, formated_dataset, dataset_name=dataset_name)

if __name__ == '__main__':
    main()