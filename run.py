import json
import os
import asyncio
import argparse
import xlsxwriter
import multiprocess
import multiprocessing

import module.func_util as fu
from module import Annotation, Processor, Annotator

async def main_cli(args):
    """
    CLI version of main function that accepts command line arguments.
    
    Args:
        args: Command line arguments parsed by argparse
    """
    logger = fu.get_async_logger()
    
    config = fu.get_config('config.yml')
    
    # 1. load annotator - API-only mode
    assert args.api_model in ('qwen', 'deepseek', 'glm', 'gpt')
    use_api = True
    api_cfg = fu.get_config(config['api_cfg'])[args.api_model]

    # Create minimal annotator_cfg for backward compatibility
    # self_consistency needs to modify generation parameters
    annotator_cfg = {
        'anno_temperature': 0.1,
        'anno_top_p': 0.5,
        'anno_max_tokens': 100,
        'repetition_penalty': 1,
    }

    # init annotator
    annotator = Annotator(annotator_cfg, api_cfg)
    
    for dataset_name in args.datasets:
        # 2. load and pre-process the data
        assert dataset_name in config['data_cfgs'].keys()
        
        # label form
        natural_form = False  # natural_form is used to indicate whether the labels are in natural language form.
        
        data_cfg = fu.get_config(config['data_cfgs'][dataset_name])  # data config
        labels_cfg = fu.get_config(config['label_cfgs'][dataset_name])  # label config
        proc = Processor(data_cfg, labels_cfg, natural_form)
        dataset, support_set_info = proc.process(method=args.method, retrieval_base_size=args.retrieval_base_size)
        label_statistics = proc.statistics(dataset)
        logger.info(f'dataset {dataset_name} label statistics:\n{label_statistics["label_nums"]}')
        label_nums, label_dist = label_statistics["label_nums"], label_statistics["label_dist"]
        with open(f'{dataset_name}_label_stats.json', 'w') as f:
            json.dump(label_nums, f)
            json.dump(label_dist, f)
        language = data_cfg['language']  # language of the dataset
        
        # 3. annotate the data by LLMs
        # 3.1 test subset sampling settings
        # 'random' for random sampling. Each instance has the same probability of being selected.
        # 'lab_uniform' for uniform sampling at label-level. Choice probability is uniform for each label.
        # 'proportion' for proportion sampling. Choice probability is proportional to the number of entities for each label.
        # 'shot_sample' for sampling test set like k-shot sampling. Each label has at least k instances.
        # 'mix' for mixed sampling. Combine 'lab_uniform' and 'proportion' sampling strategies.
        sampling_strategy = args.sampling_strategy
        if args.test_subset_size > 0:
            assert sampling_strategy in ('random', 'lab_uniform', 'proportion', 'shot_sample', 'mix')
            dataset_subset = proc.subset_sampling(dataset, args.test_subset_size, sampling_strategy, 42)
        else:
            dataset_subset = dataset
        # dataset_subset = dataset.shuffle(seed=42).flatten_indices().select(
        #     range(test_subset_size))  # fixed test subset for all experiments
        
        # 3.2 dialogue style settings
        # 'multi-qa' for multi-turn QA, we concatenate the output of the previous turn with the input of the current turn.
        # 'batch-qa' for batch QA, we use new context for each query.
        dialogue_style = args.dialogue_style
        assert dialogue_style in ('multi_qa', 'batch_qa')
        
        # 3.3 annotation prompt settings
        anno = Annotation(annotator, labels_cfg)
        for prompt_type in args.prompt_types:
            assert prompt_type in ('mt_fs', 'st_fs', 'sc_fs', 'self_cons', 'retrieval_fs', 'retrieval_lsp')
            
            if dialogue_style == 'multi_qa' and prompt_type != 'mt_fs':
                await logger.error('multi_qa style only support mt_fs')
                dialogue_style = 'batch_qa'
            if dialogue_style == 'multi_qa' and use_api and annotator.batch_infer:
                await logger.error('multi_qa style cannot support batch inference using API')
                annotator.batch_infer = False  # set batch_infer to False for multi_qa
            if args.method == 'retrieval' and prompt_type not in ('retrieval_fs', 'retrieval_lsp'):
                await logger.error('retrieval-based processingg only support retrieval_fs ')
                return
            # 3.4 other testing settings
            
            if prompt_type == 'sc_fs':
                subset_sizes = args.sc_fs_subset_sizes  # label subset sizes for sc_fs
            else:
                subset_sizes = [0.5]
            
            subset_sizes = args.override_subset_sizes if args.override_subset_sizes else subset_sizes
            ignore_sent_set = args.ignore_sent_set  # whether to ignore the sentence. If True, the sentence in the examples will be shown as '***'.
            label_mention_map_portions_set = args.label_mention_map_portions_set  # the portion of the corrected label-mention pair. Default is 1, which means all the label-mention pairs are correct.
            label_mention_map_choice = args.label_mention_map_choice  # 'accuracy', 'redundancy'
            repeat_num = args.repeat_num
            
            anno_cfg_paths = config['anno_cfgs'][prompt_type]
            anno_cfgs = [fu.get_config(anno_cfg_path) for anno_cfg_path in anno_cfg_paths]
            
            # 5. start annotation
            results = []  # for storing the results with different cfg and seeds (seeds are used to sample k-shot examples)
            for ignore_sent, label_mention_map_portions in zip(ignore_sent_set, label_mention_map_portions_set):
                for label_mention_map_portion in label_mention_map_portions:
                    for subset_size in subset_sizes:
                        for rep_num in range(repeat_num):
                            for anno_cfg in anno_cfgs:
                                await logger.info(f'dataset: {dataset_name}')
                                await logger.info(f'language: {language}')
                                await logger.info(f'use api: {use_api}')
                                await logger.info(f'api model: {args.api_model}')
                                await logger.info(f'use prompt type: {prompt_type}')
                                await logger.info(f'test subset size: {args.test_subset_size}')
                                await logger.info(f'subset sampling strategy: {sampling_strategy}')
                                await logger.info(f'dialogue style: {dialogue_style}')
                                await logger.info(f'ignore sentence: {ignore_sent}')
                                await logger.info(f'label-mention map portion: {label_mention_map_portion}')
                                await logger.info(f'label_mention_map_choice: {label_mention_map_choice}')
                                
                                match prompt_type:
                                    case 'mt_fs':
                                        await logger.info(f'demo_times: {rep_num + 1}')
                                    case 'sc_fs':
                                        await logger.info(f'repeat num: {rep_num + 1}')
                                    case 'self_cons':
                                        # if we use self-consistency,
                                        # we need to set the temperature, top_p, num_return_sequences manually before init annotator
                                        anno.annotator.annotator_cfg['anno_temperature'] = anno_cfg['temperature']
                                        anno.annotator.annotator_cfg['anno_top_p'] = anno_cfg['top_p']
                                    case 'retrieval_fs' | 'retrieval_lsp':
                                        anno_cfg['retrieval_base_size'] = args.retrieval_base_size
                                anno_cfg['support_set_info'] = support_set_info
                                anno_cfg['demo_times'] = rep_num + 1  # for mt_fs
                                anno_cfg['language'] = language
                                anno_cfg['repeat_num'] = rep_num + 1  # for sc_fs
                                anno_cfg['subset_size'] = subset_size
                                anno_cfg['prompt_template'] = fu.get_config(anno_cfg['prompt_template_dir'])
                                anno_cfg['label_mention_map_portion'] = label_mention_map_portion
                                anno_cfg['label_mention_map_choice'] = label_mention_map_choice
                                anno_cfg['ignore_sent'] = ignore_sent
                                anno_cfg['dialogue_style'] = dialogue_style
                                anno_cfg['sampling_strategy'] = sampling_strategy
                                
                                # 3. run the annotation with the given seed (seed for sampling k-shot examples)
                                tasks = []
                                for seed in args.seeds:
                                    await logger.info(f"anno cfg: {anno_cfg['name']}")
                                    task = asyncio.create_task(
                                        anno.annotate_by_one(dataset_subset,
                                                             anno_cfg=anno_cfg,
                                                             dataset_name=dataset_name,
                                                             eval=True,
                                                             cache=True,
                                                             prompt_type=prompt_type,
                                                             seed=seed,
                                                             concurrency_level=args.concurrency_level,
                                        )
                                    )
                                    tasks.append(task)
                                
                                # 4. save the results to a excel file
                                results += await asyncio.gather(*tasks)
            
            # 6. save all the metrics to excel files
            start_row = 2  # the starting row of the excel file
            excel_file = f'{dataset_name}_metrics.xlsx'
            workbook = xlsxwriter.Workbook(excel_file)  # write metric to excel
            metrics_worksheet = workbook.add_worksheet('metrics')  # default 'Sheet 1'
            metrics_by_class_worksheet = workbook.add_worksheet('metrics_by_class')  # default 'Sheet 2'
            for res_file, res_by_class_file, anno_cfg in results:
                if res_file is None or res_by_class_file is None:
                    eval_dir = anno_cfg['eval_dir'].format(dataset_name=dataset_name)
                    res_cache_dir = os.path.join(eval_dir, anno_cfg['task_dir'])
                    if not os.path.exists(res_cache_dir):
                        os.makedirs(res_cache_dir)
                    res_file = os.path.join(res_cache_dir, '{}_res.txt'.format(anno_cfg['annotator_name']))
                    res_by_class_file = os.path.join(res_cache_dir, '{}_res_by_class.csv'.format(anno_cfg['annotator_name']))
                    logger.info(f'res_file not found, set to {res_file}')
                    logger.info(f'res_by_class_file not found, set to {res_by_class_file}')
                
                logger.info(f'write metrics ({res_file}) to excel file {excel_file}')
                start_row = fu.write_metrics_to_excel(
                    metrics_worksheet=metrics_worksheet,
                    metrics_by_class_worksheet=metrics_by_class_worksheet,
                    start_row=start_row,
                    res_file=res_file,
                    res_by_class_file=res_by_class_file,
                    anno_cfg=anno_cfg,
                )
            workbook.close()
    await logger.shutdown()  # close the logger


async def main():
    # 'ontonotes5_zh'
    # 'ontonotes5_en',
    # 'mit_restaurant',
    # 'mit_movies'
    # 'CMeEE_V2'
    dataset_names = ['ontonotes5_en']  # 'ontonotes5_en', 'mit_movies', 'CMeEE_V2', 'ontonotes5_zh'
    use_api = True
    api_model = 'gpt'  # 'qwen', 'deepseek', 'glm', 'gpt'
    method = 'lsp'  # 'lsp', 'retrieval'

    seeds = [22]  # retrieval-based时 只需要一个seed
    test_subset_size = -1
    retrieval_base_size = 20  # number of training data used for k-shot sampling or retrieval setting
    concurrency_level = 10  # number of concurrent requests

    logger = fu.get_async_logger()

    config = fu.get_config('config.yml')
    # 1. load annotator - API-only mode
    assert api_model in ('qwen', 'deepseek', 'glm', 'gpt')
    api_cfg = fu.get_config(config['api_cfg'])[api_model]

    # Create minimal annotator_cfg for backward compatibility
    # self_consistency needs to modify generation parameters
    annotator_cfg = {
        'anno_temperature': 0.1,
        'anno_top_p': 0.5,
        'anno_max_tokens': 100,
        'repetition_penalty': 1,
    }

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
        dataset, support_set_info = proc.process(method=method, retrieval_base_size=retrieval_base_size)
        label_statistics = proc.statistics(dataset)
        logger.info(f'dataset {dataset_name} label statistics:\n{label_statistics["label_nums"]}')
        label_nums, label_dist = label_statistics["label_nums"], label_statistics["label_dist"]
        with open(f'{dataset_name}_label_stats.json', 'w') as f:
            json.dump(label_nums, f)
            json.dump(label_dist, f)
        language = data_cfg['language']  # language of the dataset

        # 3. annotate the data by LLMs
        # 3.1 test subset sampling settings
        # 'random' for random sampling. Each instance has the same probability of being selected.
        # 'lab_uniform' for uniform sampling at label-level. Choice probability is uniform for each label.
        # 'proportion' for proportion sampling. Choice probability is proportional to the number of entities for each label.
        # 'shot_sample' for sampling test set like k-shot sampling. Each label has at least k instances.
        # 'mix' for mixed sampling. Combine 'lab_uniform' and 'proportion' sampling strategies.
        sampling_strategy = None
        if test_subset_size > 0:
            assert sampling_strategy in ('random', 'lab_uniform', 'proportion', 'shot_sample', 'mix')
            dataset_subset = proc.subset_sampling(dataset, test_subset_size, sampling_strategy, 42)
        else:
            dataset_subset = dataset
        # dataset_subset = dataset.shuffle(seed=42).flatten_indices().select(
        #     range(test_subset_size))  # fixed test subset for all experiments

        # 3.2 dialogue style settings
        # 'multi-qa' for multi-turn QA, we concatenate the output of the previous turn with the input of the current turn.
        # 'batch-qa' for batch QA, we use new context for each query.
        dialogue_style = 'batch_qa'
        assert dialogue_style in ('multi_qa', 'batch_qa')

        # 3.3 annotation prompt settings
        anno = Annotation(annotator, labels_cfg)
        for prompt_type in ['sc_fs']: # 'mt_fs', 'st_fs', 'sc_fs', 'self_cons', 'retrieval_fs', 'retrieval_lsp'
            assert prompt_type in ('mt_fs', 'st_fs', 'sc_fs', 'self_cons', 'retrieval_fs', 'retrieval_lsp')

            if dialogue_style == 'multi_qa' and prompt_type != 'mt_fs':
                await logger.error('multi_qa style only support mt_fs')
                dialogue_style = 'batch_qa'
            if dialogue_style == 'multi_qa' and use_api and annotator.batch_infer:
                await logger.error('multi_qa style cannot support batch inference using API')
                annotator.batch_infer = False  # set batch_infer to False for multi_qa
            if method == 'retrieval' and prompt_type not in ('retrieval_fs', 'retrieval_lsp'):
                await logger.error('retrieval-based processingg only support retrieval_fs ')
                return
            # 3.4 other testing settings

            if prompt_type == 'sc_fs':
                subset_sizes = [0.1, 0.2, 0.3, 0.4, 0.5] # label subset sizes for sc_fs
            else:
                subset_sizes = [0.5]

            subset_sizes = [0.5]
            ignore_sent_set = [False] # [False, True]  # whether to ignore the sentence. If True, the sentence in the examples will be shown as '***'.
            label_mention_map_portions_set = [[1]] # [[1]]  [[1], [1, 0.75, 0.5, 0.25, 0]], the portion of the corrected label-mention pair. Default is 1, which means all the label-mention pairs are correct.
            label_mention_map_choice = 'redundancy'  # 'accuracy', 'redundancy'
            repeat_num = 2

            anno_cfg_paths = config['anno_cfgs'][prompt_type]
            anno_cfgs = [fu.get_config(anno_cfg_path) for anno_cfg_path in anno_cfg_paths]

            # 5. start annotation
            results = []  # for storing the results with different cfg and seeds (seeds are used to sample k-shot examples)
            for ignore_sent, label_mention_map_portions in zip(ignore_sent_set, label_mention_map_portions_set):
                for label_mention_map_portion in label_mention_map_portions:
                    for subset_size in subset_sizes:
                        for rep_num in range(repeat_num):
                            for anno_cfg in anno_cfgs:
                                await logger.info(f'dataset: {dataset_name}')
                                await logger.info(f'language: {language}')
                                await logger.info(f'use api: {use_api}')
                                await logger.info(f'api model: {api_model}')
                                await logger.info(f'use prompt type: {prompt_type}')
                                await logger.info(f'test subset size: {test_subset_size}')
                                await logger.info(f'subset sampling strategy: {sampling_strategy}')
                                await logger.info(f'dialogue style: {dialogue_style}')
                                await logger.info(f'ignore sentence: {ignore_sent}')
                                await logger.info(f'label-mention map portion: {label_mention_map_portion}')
                                await logger.info(f'label_mention_map_choice: {label_mention_map_choice}')

                                match prompt_type:
                                    case 'mt_fs':
                                        await logger.info(f'demo_times: {rep_num + 1}')
                                    case 'sc_fs':
                                        await logger.info(f'repeat num: {rep_num + 1}')
                                    case 'self_cons':
                                        # if we use self-consistency,
                                        # we need to set the temperature, top_p, num_return_sequences manually before init annotator
                                        anno.annotator.annotator_cfg['anno_temperature'] = anno_cfg['temperature']
                                        anno.annotator.annotator_cfg['anno_top_p'] = anno_cfg['top_p']
                                    case 'retrieval_fs' | 'retrieval_lsp':
                                        anno_cfg['retrieval_base_size'] = retrieval_base_size
                                anno_cfg['support_set_info'] = support_set_info
                                anno_cfg['demo_times'] = rep_num + 1  # for mt_fs
                                anno_cfg['language'] = language
                                anno_cfg['repeat_num'] = rep_num + 1  # for sc_fs
                                anno_cfg['subset_size'] = subset_size
                                anno_cfg['prompt_template'] = fu.get_config(anno_cfg['prompt_template_dir'])
                                anno_cfg['label_mention_map_portion'] = label_mention_map_portion
                                anno_cfg['label_mention_map_choice'] = label_mention_map_choice
                                anno_cfg['ignore_sent'] = ignore_sent
                                anno_cfg['dialogue_style'] = dialogue_style
                                anno_cfg['sampling_strategy'] = sampling_strategy

                                # 3. run the annotation with the given seed (seed for sampling k-shot examples)
                                tasks = []
                                for seed in seeds:
                                    await logger.info(f"anno cfg: {anno_cfg['name']}")
                                    task = asyncio.create_task(
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
                                    tasks.append(task)

                                # 4. save the results to a excel file
                                results += await asyncio.gather(*tasks)

            # 6. save all the metrics to excel files
            start_row = 2  # the starting row of the excel file
            excel_file = f'{dataset_name}_metrics.xlsx'
            workbook = xlsxwriter.Workbook(excel_file)  # write metric to excel
            metrics_worksheet = workbook.add_worksheet('metrics')  # default 'Sheet 1'
            metrics_by_class_worksheet = workbook.add_worksheet('metrics_by_class')  # default 'Sheet 2'
            for res_file, res_by_class_file, anno_cfg in results:
                if res_file is None or res_by_class_file is None:
                    eval_dir = anno_cfg['eval_dir'].format(dataset_name=dataset_name)
                    res_cache_dir = os.path.join(eval_dir, anno_cfg['task_dir'])
                    if not os.path.exists(res_cache_dir):
                        os.makedirs(res_cache_dir)
                    res_file = os.path.join(res_cache_dir, '{}_res.txt'.format(anno_cfg['annotator_name']))
                    res_by_class_file = os.path.join(res_cache_dir, '{}_res_by_class.csv'.format(anno_cfg['annotator_name']))
                    logger.info(f'res_file not found, set to {res_file}')
                    logger.info(f'res_by_class_file not found, set to {res_by_class_file}')

                logger.info(f'write metrics ({res_file}) to excel file {excel_file}')
                start_row = fu.write_metrics_to_excel(
                    metrics_worksheet=metrics_worksheet,
                    metrics_by_class_worksheet=metrics_by_class_worksheet,
                    start_row=start_row,
                    res_file=res_file,
                    res_by_class_file=res_by_class_file,
                    anno_cfg=anno_cfg,
                )
            workbook.close()
    await logger.shutdown()  # close the logger


if __name__ == '__main__':
    # set 'spawn' start method in the main process to parallelize computation across several GPUs when using multi-processes in the map function
    # refer to https://huggingface.co/docs/datasets/process#map
    multiprocess.set_start_method('spawn')
    multiprocessing.set_start_method('spawn')
    
    # Check if running in CLI mode
    import sys
    if len(sys.argv) > 1:
        # CLI mode
        parser = argparse.ArgumentParser(description='LSP-NER annotation pipeline CLI')
        
        # Required arguments
        parser.add_argument('--datasets', type=str, nargs='+', required=True,
                          help='Dataset names to process (e.g., ontonotes5_en mit_movies CMeEE_V2)')
        parser.add_argument('--method', type=str, required=True,
                          choices=['lsp', 'retrieval'],
                          help='Processing method')
        
        # Model selection - API-only mode
        parser.add_argument('--api-model', type=str, default='gpt',
                          choices=['qwen', 'deepseek', 'glm', 'gpt'],
                          help='API model to use')
        
        # Sampling and data settings
        parser.add_argument('--test-subset-size', type=int, default=-1,
                          help='Test subset size (-1 for full test set)')
        parser.add_argument('--sampling-strategy', type=str, default=None,
                          choices=['random', 'lab_uniform', 'proportion', 'shot_sample', 'mix'],
                          help='Sampling strategy for test subset')
        parser.add_argument('--retrieval-base-size', type=int, default=20,
                          help='Number of training data used for k-shot sampling or retrieval setting')
        parser.add_argument('--seeds', type=int, nargs='+', default=[22, 32, 42],
                          help='Random seeds for reproducibility')
        parser.add_argument('--concurrency-level', type=int, default=10,
                          help='Number of concurrent requests')
        
        # Prompt and dialogue settings
        parser.add_argument('--prompt-types', type=str, nargs='+', 
                          default=['sc_fs'],
                          choices=['mt_fs', 'st_fs', 'sc_fs', 'self_cons', 'retrieval_fs', 'retrieval_lsp'],
                          help='Prompt types to use')
        parser.add_argument('--dialogue-style', type=str, default='batch_qa',
                          choices=['multi_qa', 'batch_qa'],
                          help='Dialogue style for annotation')
        
        # Advanced settings
        parser.add_argument('--sc-fs-subset-sizes', type=float, nargs='+', 
                          default=[0.1, 0.2, 0.3, 0.4, 0.5],
                          help='Label subset sizes for sc_fs')
        parser.add_argument('--override-subset-sizes', type=float, nargs='+',
                          help='Override subset sizes for all prompt types')
        parser.add_argument('--ignore-sent-set', type=lambda x: (x.lower() == 'true'), nargs='+',
                          default=[False],
                          help='Whether to ignore the sentence')
        parser.add_argument('--label-mention-map-portions-set', type=lambda x: [float(y) for y in x.split(',')],
                          nargs='+', default=[[1]],
                          help='Portion of the corrected label-mention pair (comma-separated values)')
        parser.add_argument('--label-mention-map-choice', type=str, default='redundancy',
                          choices=['accuracy', 'redundancy'],
                          help='Label-mention map choice')
        parser.add_argument('--repeat-num', type=int, default=2,
                          help='Number of repetitions')
        
        args = parser.parse_args()
        
        # Run CLI version
        asyncio.run(main_cli(args))
    else:
        # Original main() mode
        asyncio.run(main())