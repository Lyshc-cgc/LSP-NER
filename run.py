from module.annotation import Annotation
from module.processor import Processor
from module.func_util import get_config

def main():
    config = get_config('config.yml')
    # 1. pre-process the data
    dataset_name = 'conll2003'  # 'conll2003', 'ontonotes5', 'ace2005'
    assert dataset_name in config['data_cfgs'].keys()

    # label form
    natural_form = False  # natural_form is used to indicate whether the labels are in natural language form.

    data_cfg = get_config(config['data_cfgs'][dataset_name])  # data config
    labels_cfg = get_config(config['label_cfgs'][dataset_name])  # label config
    proc = Processor(data_cfg, labels_cfg, natural_form)
    dataset = proc.process()

    # 2. annotate the data by LLMs
    # 2.1 api annotator config (optional) and local annotator config
    # api annotator
    use_api = False
    api_model = 'gpt'
    assert api_model in ('qwen', 'deepseek', 'glm', 'gpt')
    api_cfg = get_config(config['api_cfg'])[api_model] if use_api else None

    # local annotator
    local_model = 'Qwen1.5'
    assert local_model in ('Qwen1.5',)  # add more
    annotator_cfg = get_config(config['annotators_cfg'])[local_model]

    # 2.2 annotation prompt settings
    prompt_type = 'sc_fs'
    assert prompt_type in ('raw', 'mt_fs', 'raw_few_shot', 'st_few_shot', 'cand_mention_fs', 'sb_fs', 'sc_fs')

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

    # 2.5 other testing settings
    ignore_sent = True  # whether to ignore the sentence. If True, the sentence in the examples will be shown as '***'.
    label_mention_map_portions = [1, 0.75, 0.5, 0.25]  # [1, 0.75, 0.5, 0.25], the portion of the corrected label-mention pair. Default is 1, which means all the label-mention pairs are correct.
    repeat_num = 6
    seeds = [22, 32, 42]
    start_row = -1  # we set -1, because we don't want write metrics to excel files when annotating


    anno_cfg_paths = config['anno_cfgs'][prompt_type]
    anno_cfgs =  [get_config(anno_cfg_path) for anno_cfg_path in anno_cfg_paths]
    anno = Annotation(annotator_cfg, api_cfg, labels_cfg)

    for label_mention_map_portion in label_mention_map_portions:
        for rep_num in range(repeat_num):
            for anno_cfg in anno_cfgs:
                for seed in seeds:
                    anno_cfg['demo_times'] = rep_num + 1  # for mt_fs
                    anno_cfg['repeat_num'] = rep_num + 1  # for sc_fs

                    print(f'use api: {use_api}')
                    print(f'api model: {api_model}')
                    print(f'local model: {local_model}')
                    print(f'use prompt type: {prompt_type}')
                    print(f'test subset size: {test_subset_size}')
                    print(f'subset sampling strategy: {sampling_strategy}')
                    print(f'dialogue style: {dialogue_style}')
                    print(f'ignore sentence: {ignore_sent}')
                    print(f'label-mention map portion: {label_mention_map_portion}')
                    if prompt_type == 'mt_fs':
                        print(f'demo_times: {repeat_num}')
                    elif prompt_type == 'sc_fs':
                        print(f'repeat num: {repeat_num}')

                    if test_subset_size > 0:
                        dataset_subset = proc.subset_sampling(dataset, test_subset_size, sampling_strategy, seed)

                    print(f"anno cfg: {anno_cfg['name']}")
                    anno.annotate_by_one(dataset_subset,
                                         anno_cfg=anno_cfg,
                                         dataset_name=dataset_name,
                                         eval=True,
                                         cache=True,
                                         prompt_type=prompt_type,
                                         sampling_strategy=sampling_strategy,
                                         dialogue_style=dialogue_style,
                                         ignore_sent=ignore_sent,
                                         label_mention_map_portion=label_mention_map_portion,
                                         seed=seed,
                                         start_row=start_row)

if __name__ == '__main__':
    main()