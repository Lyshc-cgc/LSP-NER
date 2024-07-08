from module.annotation import Annotation, annotate_augmented
from module.processor import Processor
from module.func_util import get_config

def main():
    config = get_config('config.yml')
    # 1. pre-process the data
    dataset_name = 'conll'  # 'conll', 'ontonotes'
    assert dataset_name in config['data_cfgs'].keys() # ('conll', 'ontonotes')

    data_cfg = get_config(config['data_cfgs'][dataset_name])
    labels_cfg = get_config(config['labels_cfg'])[dataset_name]
    proc = Processor(data_cfg, labels_cfg)
    dataset = proc.process()

    # 2. annotate the data by LLMs
    # 2.1 api config
    use_api = False
    api_model = 'gpt'
    assert api_model in ('qwen', 'deepseek', 'glm', 'gpt')
    api_cfg = get_config(config['api_cfg'])[api_model] if use_api else None

    # 2.2 annotation prompt settings
    prompt_type = 'sc_fs'
    assert prompt_type in ('raw', 'single_type', 'mt_few_shot', 'raw_few_shot', 'st_few_shot',
                           'cand_mention_fs', 'sb_fs', 'sc_fs')
    augmented = False

    # 2.3 test subset sampling settings
    # 'random' for random sampling. Each instance has the same probability of being selected.
    # 'lab_uniform' for uniform sampling at label-level. Choice probability is uniform for each label.
    # 'proportion' for proportion sampling. Choice probability is proportional to the number of entities for each label.
    # 'shot_sample' for sampling test set like k-shot sampling. Each label has at least k instances.
    test_subset_size = 200
    sampling_strategy = 'random'
    assert sampling_strategy in ('random', 'lab_uniform', 'proportion', 'shot_sample')

    # 2.4 dialogue style settings
    # 'multi-qa' for multi-turn QA, we concatenate the output of the previous turn with the input of the current turn.
    # 'batch-qa' for batch QA, we use new context for each query.
    dialogue_style = 'batch_qa'
    assert dialogue_style in ('multi_qa', 'batch_qa')
    if not augmented:

        anno_cfg_paths = config['anno_cfgs'][prompt_type]
        print(f'use prompt type: {prompt_type}')
        print(f'test subset size: {test_subset_size}')
        print(f'subset sampling strategy: {sampling_strategy}')
        print(f'dialogue style: {dialogue_style}')

        if test_subset_size > 0:
            dataset = proc.subset_sampling(dataset, test_subset_size, sampling_strategy)

        for anno_cfg_path in anno_cfg_paths:
            anno_cfg = get_config(anno_cfg_path)
            print(f"anno cfg: {anno_cfg['name']}")
            anno = Annotation(anno_cfg, api_cfg, labels_cfg)
            anno.annotate_by_all(dataset, quality=False, dataset_name=dataset_name, eval=True, cache=True,
                                 prompt_type=prompt_type, sampling_strategy=sampling_strategy, dialogue_style=dialogue_style,
                                 augmented=augmented)

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