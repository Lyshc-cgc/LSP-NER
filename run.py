from module.annotation import Annotation, annotate_augmented
from module.processor import Processor
from module.func_util import get_config

def main():
    config = get_config('config.yml')
    # 1. pre-process the data
    dataset_name = 'ontonotes'  # 'conll', 'ontonotes'
    assert dataset_name in config['data_cfgs'].keys() # ('conll', 'ontonotes')

    data_cfg = get_config(config['data_cfgs'][dataset_name])
    labels_cfg = get_config(config['labels_cfg'])[dataset_name]
    proc = Processor(data_cfg, labels_cfg)
    dataset = proc.process()

    # 2. annotate the data by LLMs
    # 2.1 api config
    use_api = True
    api_model = 'gpt'
    assert api_model in ('qwen', 'deepseek', 'glm', 'gpt')
    api_cfg = get_config(config['api_cfg'])[api_model] if use_api else None

    # 2.2 annotation prompt settings
    prompt_type = 'mt_few_shot'
    augmented = False
    if not augmented:
        # dataset = dataset.shuffle().select(range(200))
        dataset = proc.test_subset_sampling(dataset, 100)
        assert prompt_type in ('raw', 'single_type', 'mt_few_shot', 'few_shot', 'st_few_shot', 'cand_mention_fs')
        anno_cfg_paths = config['anno_cfgs'][prompt_type]

        for anno_cfg_path in anno_cfg_paths:
            anno_cfg = get_config(anno_cfg_path)
            anno = Annotation(anno_cfg, api_cfg, labels_cfg)
            anno.annotate_by_all(dataset, quality=False, dataset_name=dataset_name, eval=True, cache=True,
                                 prompt_type=prompt_type, augmented=augmented)

    else:
        # 2.3 (optional) augmented annotation
        type_fs_cfg_paths = config['anno_cfgs'][prompt_type]
        raw_cfg_paths = config['anno_cfgs']['raw']
        for type_fs_cfg_path in type_fs_cfg_paths:
            for raw_cfg_path in raw_cfg_paths:
                type_fs_cfg = get_config(type_fs_cfg_path)
                raw_cfg = get_config(raw_cfg_path)
                annotate_augmented(type_fs_cfg=type_fs_cfg,
                                   raw_cfg=raw_cfg,
                                   api_cfg=api_cfg,
                                   labels_cfg=labels_cfg,
                                   dataset=dataset,
                                   dataset_name=dataset_name)

if __name__ == '__main__':
    main()