import os
import asyncio
import xlsxwriter
import multiprocess
import multiprocessing

import module.func_util as fu
from module import Annotation, Processor, Annotator

# 'ontonotes5_zh'
# 'conll2003',
# 'ontonotes5_en',
# 'mit_restaurant',
# 'mit_movies'
# 'CMeEE_V2'
dataset_names = ['ontonotes5_en', 'mit_movies', 'CMeEE_V2', 'ontonotes5_zh']  # 'ontonotes5_en', 'mit_movies',
use_api = True
api_model = 'deepseek'  # 'qwen', 'deepseek', 'glm', 'gpt'
local_model = 'Mistral'  # 'Qwen1.5', 'Mistral', 'Qwen2.5'
seeds = [22, 32, 42]
test_subset_size = 100
concurrency_level = 10  # number of concurrent requests

async def main():
    logger = fu.get_async_logger()

    config = fu.get_config('config.yml')
    # 1. load annotator
    # api annotator
    assert api_model in ('qwen', 'deepseek', 'glm', 'gpt')
    api_cfg = fu.get_config(config['api_cfg'])[api_model] if use_api else None

    # local annotator
    assert local_model in ('Qwen1.5', 'Mistral', 'Qwen2.5')  # add more
    annotator_cfg = fu.get_config(config['annotators_cfg'])[local_model]

    # init annotator
    annotator = Annotator(annotator_cfg, api_cfg)

    for dataset_name in dataset_names:
        # 2. load and pre-process the data
        assert dataset_name in config['data_cfgs'].keys()

        # label form
        natural_form = False  # natural_form is used to indicate whether the labels are in natural language form.

        data_cfg = fu.get_config(config['data_cfgs'][dataset_name])  # data config
        labels_cfg = fu.get_config(config['label_cfgs'][dataset_name])  # label config
        proc = Processor(data_cfg, labels_cfg, natural_form)
        dataset = proc.process()
        language = data_cfg['language']  # language of the dataset

        # 3. annotate the data by LLMs
        # 3.1 test subset sampling settings
        # 'random' for random sampling. Each instance has the same probability of being selected.
        # 'lab_uniform' for uniform sampling at label-level. Choice probability is uniform for each label.
        # 'proportion' for proportion sampling. Choice probability is proportional to the number of entities for each label.
        # 'shot_sample' for sampling test set like k-shot sampling. Each label has at least k instances.
        sampling_strategy = 'random'
        assert sampling_strategy in ('random', 'lab_uniform', 'proportion', 'shot_sample')

        # 3.2 dialogue style settings
        # 'multi-qa' for multi-turn QA, we concatenate the output of the previous turn with the input of the current turn.
        # 'batch-qa' for batch QA, we use new context for each query.
        dialogue_style = 'batch_qa'
        assert dialogue_style in ('multi_qa', 'batch_qa')

        # 3.3 annotation prompt settings
        anno = Annotation(annotator, labels_cfg)
        for prompt_type in ['sc_fs', 'st_fs', 'mt_fs']: # 'mt_fs', 'st_fs', 'sc_fs',
            assert prompt_type in ('mt_fs', 'st_fs', 'sc_fs')

            if dialogue_style == 'multi_qa' and prompt_type != 'mt_fs':
                await logger.error('multi_qa style only support mt_fs')
                dialogue_style = 'batch_qa'
            if dialogue_style == 'multi_qa' and use_api and annotator.batch_infer:
                await logger.error('batch_qa style cannot support batch inference using API')
                annotator.batch_infer = False  # set batch_infer to False for batch_qa

            # 3.4 other testing settings
            if prompt_type == 'sc_fs':
                subset_sizes = [0.1, 0.2, 0.3, 0.4, 0.5] # label subset sizes for sc_fs
            else:
                subset_sizes = [0.5]

            subset_sizes= [0.5]
            ignore_sent_set = [False] # [False, True]  # whether to ignore the sentence. If True, the sentence in the examples will be shown as '***'.
            label_mention_map_portions_set = [[1]]# [1, 0.75, 0.5, 0.25], the portion of the corrected label-mention pair. Default is 1, which means all the label-mention pairs are correct.
            repeat_num = 1
            demo_times = 1
            # subset_size = 0.5

            anno_cfg_paths = config['anno_cfgs'][prompt_type]
            anno_cfgs = [fu.get_config(anno_cfg_path) for anno_cfg_path in anno_cfg_paths]

            # 5. start annotation
            results = []  # for storing the results with different cfg and seeds
            for ignore_sent, label_mention_map_portions in zip(ignore_sent_set, label_mention_map_portions_set):
                for label_mention_map_portion in label_mention_map_portions:
                    for subset_size in subset_sizes:
                        for rep_num in range(repeat_num):
                            for anno_cfg in anno_cfgs:
                                await logger.info(f'dataset: {dataset_name}')
                                await logger.info(f'language: {language}')
                                await logger.info(f'use api: {use_api}')
                                await logger.info(f'api model: {api_model}')
                                await logger.info(f'local model: {local_model}')
                                await logger.info(f'use prompt type: {prompt_type}')
                                await logger.info(f'test subset size: {test_subset_size}')
                                await logger.info(f'subset sampling strategy: {sampling_strategy}')
                                await logger.info(f'dialogue style: {dialogue_style}')
                                await logger.info(f'ignore sentence: {ignore_sent}')
                                await logger.info(f'label-mention map portion: {label_mention_map_portion}')

                                if prompt_type == 'mt_fs':
                                    await logger.info(f'demo_times: {rep_num + 1}')
                                elif prompt_type == 'sc_fs':
                                    await logger.info(f'repeat num: {rep_num + 1}')

                                anno_cfg['demo_times'] = rep_num + 1  # for mt_fs
                                anno_cfg['language'] = language
                                anno_cfg['repeat_num'] = rep_num + 1  # for sc_fs
                                anno_cfg['subset_size'] = subset_size
                                anno_cfg['prompt_template'] = fu.get_config(anno_cfg['prompt_template_dir'])
                                anno_cfg['label_mention_map_portion'] = label_mention_map_portion
                                anno_cfg['ignore_sent'] = ignore_sent
                                anno_cfg['dialogue_style'] = dialogue_style
                                anno_cfg['sampling_strategy'] = sampling_strategy

                                # 3. run the annotation with the given seed
                                tasks = []
                                for seed in seeds:
                                    if test_subset_size > 0:
                                        loop = asyncio.get_running_loop()
                                        dataset_subset = await loop.run_in_executor(
                                            None,
                                            proc.subset_sampling,
                                            dataset,
                                            test_subset_size,
                                            sampling_strategy,
                                            seed
                                        )

                                    await logger.info(f"anno cfg: {anno_cfg['name']}")
                                    tasks.append(
                                        anno.annotate_by_one(dataset_subset,
                                                             anno_cfg=anno_cfg,
                                                             dataset_name=dataset_name,
                                                             eval=True,
                                                             cache=True,
                                                             prompt_type=prompt_type,
                                                             seed=seed,
                                                             concurrency_level=concurrency_level,
                                                             )
                                    )

                                # 4. save the results to a excel file
                                results += await asyncio.gather(*tasks)

            # 6. save all the metrics to excel files
            start_row = 2  # the starting row of the excel file
            excel_file = f'{dataset_name}_metrics.xlsx'
            workbook = xlsxwriter.Workbook(excel_file)  # write metric to excel
            worksheet = workbook.add_worksheet()  # default 'Sheet 1'
            for res_file, anno_cfg in results:
                if res_file is None:
                    eval_dir = anno_cfg['eval_dir'].format(dataset_name=dataset_name)
                    res_cache_dir = os.path.join(eval_dir, anno_cfg['task_dir'])
                    if not os.path.exists(res_cache_dir):
                        os.makedirs(res_cache_dir)
                    res_file = os.path.join(res_cache_dir, '{}_res.txt'.format(anno_cfg['annotator_name']))

                logger.info(f'write metrics ({res_file}) to excel file {excel_file}')
                start_row = fu.write_metrics_to_excel(
                    worksheet=worksheet,
                    start_row=start_row,
                    res_file=res_file,
                    anno_cfg=anno_cfg,
                )
            workbook.close()
    await logger.shutdown()  # close the logger


if __name__ == '__main__':
    # set 'spawn' start method in the main process to parallelize computation across several GPUs when using multi-processes in the map function
    # refer to https://huggingface.co/docs/datasets/process#map
    multiprocess.set_start_method('spawn')
    multiprocessing.set_start_method('spawn')
    asyncio.run(main())