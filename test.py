
from module import Annotation, Processor
from module import func_util as fu

logger = fu.get_logger('test_scipt')

def main():
    config = fu.get_config('config.yml')
    # 1. pre-process the data
    dataset_name = 'mit_restaurant'  # 'conll2003', 'ontonotes5', 'mit_movies', 'mit_restaurant',
    assert dataset_name in config['data_cfgs'].keys()

    # label form
    natural_form = False  # natural_form is used to indicate whether the labels are in natural language form.

    data_cfg = fu.get_config(config['data_cfgs'][dataset_name])  # data config
    labels_cfg = fu.get_config(config['label_cfgs'][dataset_name])  # label config
    proc = Processor(data_cfg, labels_cfg, natural_form)
    dataset = proc.process()

    # 2. annotate the data by LLMs
    # 2.1 api annotator config (optional) and local annotator config
    # api annotator
    use_api = False
    api_model = 'gpt'
    assert api_model in {'qwen', 'deepseek', 'glm', 'gpt'}
    api_cfg = fu.get_config(config['api_cfg'])[api_model] if use_api else None

    # local annotator
    local_model = 'Qwen1.5'  # Mistral, Qwen1.5
    assert local_model in {'Qwen1.5', 'Mistral'}  # add more
    annotator_cfg = fu.get_config(config['annotators_cfg'])[local_model]

    # 2.2 annotation prompt settings
    prompt_type = 'mt_fs'
    assert prompt_type in {'mt_fs', 'sc_fs', 'st_fs'}

    # 2.3 test subset sampling settings
    # 'random' for random sampling. Each instance has the same probability of being selected.
    # 'lab_uniform' for uniform sampling at label-level. Choice probability is uniform for each label.
    # 'proportion' for proportion sampling. Choice probability is proportional to the number of entities for each label.
    # 'shot_sample' for sampling test set like k-shot sampling. Each label has at least k instances.
    test_subset_size = 200
    sampling_strategy = 'random'
    assert sampling_strategy in {'random', 'lab_uniform', 'proportion', 'shot_sample'}

    # 2.4 dialogue style settings
    # 'multi-qa' for multi-turn QA, we concatenate the output of the previous turn with the input of the current turn.
    # 'batch-qa' for batch QA, we use new context for each query.
    dialogue_style = 'batch_qa'
    assert dialogue_style in {'multi_qa', 'batch_qa'}

    # 2.5 other testing settings
    if prompt_type == 'sc_fs':
        subset_sizes = [0.1, 0.2, 0.3, 0.4, 0.5] # label subset sizes for sc_fs
    else:
        subset_sizes = [0.5]
    ignore_sent = False  # whether to ignore the sentence. If True, the sentence in the examples will be shown as '***'.
    label_mention_map_portions = [1]  # [1, 0.75, 0.5, 0.25] the portion of the corrected label-mention pair. Default is 1, which means all the label-mention pairs are correct.
    just_test=True  # for test, so skip to load the LLMs
    repeat_num = 1
    seeds = [22, 32, 42]  # [22, 32, 42], ['00', '01', '02']
    start_row = 2

    logger.info(f'use api: {use_api}')
    logger.info(f'api model: {api_model}')
    logger.info(f'local model: {local_model}')
    logger.info(f'use prompt type: {prompt_type}')
    logger.info(f'test subset size: {test_subset_size}')
    logger.info(f'subset sampling strategy: {sampling_strategy}')
    logger.info(f'dialogue style: {dialogue_style}')
    logger.info(f'ignore sentence: {ignore_sent}')

    anno_cfg_paths = config['anno_cfgs'][prompt_type]
    anno_cfgs = [fu.get_config(anno_cfg_path) for anno_cfg_path in anno_cfg_paths]
    # anno_cfgs =  [fu.get_config(anno_cfg_path) for anno_cfg_path in anno_cfg_paths if '1' in anno_cfg_path or '5' in anno_cfg_path ]

    anno = Annotation(annotator_cfg,
                      api_cfg,
                      labels_cfg,
                      just_test=just_test)
    for lmm_portion in label_mention_map_portions:
        for rep_num in range(repeat_num):
            for anno_cfg in anno_cfgs:
                for subset_size in subset_sizes:
                    for seed in seeds:
                        anno_cfg['demo_times'] = rep_num + 1  # for mt_fs
                        anno_cfg['repeat_num'] = rep_num + 1  # for sc_fs
                        anno_cfg['subset_size'] = subset_size  # for sc_fs
                        if test_subset_size > 0:
                            dataset_subset = proc.subset_sampling(dataset, test_subset_size, sampling_strategy, seed)
                        logger.info(f"anno cfg: {anno_cfg['name']}")

                        # write metric to excel
                        start_row = anno.annotate_by_one(dataset_subset,
                                                         anno_cfg=anno_cfg,
                                                         dataset_name=dataset_name,
                                                         eval=True,
                                                         cache=True,
                                                         prompt_type=prompt_type,
                                                         sampling_strategy=sampling_strategy,
                                                         dialogue_style=dialogue_style,
                                                         ignore_sent=ignore_sent,
                                                         label_mention_map_portion=lmm_portion,
                                                         seed=seed,
                                                         start_row=start_row)
    anno.workbook.close()

if __name__ == '__main__':
    main()