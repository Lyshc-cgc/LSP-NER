import copy
import json
import math
import random
import re
import os

import jsonlines
import wandb
import torch
from collections import Counter
from vllm import LLM, SamplingParams
from tenacity import retry, retry_if_exception_type, wait_random
from openai import OpenAI, AsyncOpenAI
from datasets import load_from_disk, Dataset, load_dataset
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, matthews_corrcoef, cohen_kappa_score
from module import func_util as fu
from module.label import Label

class Annotation(Label):
    """
    The Annotation class is used to annotate the data.
    """
    def __init__(self, annotator_cfg, api_cfg, labels_cfg, natural_form=False):
        """
        Initialize the annotation model.
        :param annotator_cfg: the configuration of the local annotator model.
        :param api_cfg: the configuration of the LLM API.
        :param labels_cfg: the configuration of the label_cfgs.
        :param natural_form: whether the labels are in natural language form.
        """
        # 0. cfg initialization
        super().__init__(labels_cfg, natural_form)
        self.annotator_cfg = annotator_cfg
        self.api_cfg = api_cfg if api_cfg else None
        self.use_api = True if api_cfg else False
        self.natural_flag = 'natural' if natural_form else 'bio'  # use natural labels or bio labels

        # 1. GPU setting
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # avoid parallelism in tokenizers
        cuda_devices = [str(i) for i in range(annotator_cfg['tensor_parallel_size'])]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_devices)

        # 2. Init the annotating model
        if self.use_api:
            self.client = OpenAI(
                api_key=os.getenv(self.api_cfg['api_key']),
                base_url=self.api_cfg['base_url'],  # base_url
                max_retries=3,
            )
        else:
            # if not use api, we employ local annotator using vllm
            # https://docs.vllm.ai/en/latest/getting_started/quickstart.html
            self.anno_model = LLM(model=annotator_cfg['checkpoint'],
                             tensor_parallel_size=annotator_cfg['tensor_parallel_size'],
                             dtype=annotator_cfg['dtype'],
                             gpu_memory_utilization=annotator_cfg['gpu_memory_utilization'],
                             trust_remote_code=True)
            self.sampling_params = SamplingParams(temperature=annotator_cfg['anno_temperature'],
                                             top_p=annotator_cfg['anno_top_p'],
                                             max_tokens=annotator_cfg['anno_max_tokens'],
                                             repetition_penalty=annotator_cfg['repetition_penalty'])

            # get anno_model's tokenizer to apply the chat template
            # https://github.com/vllm-project/vllm/issues/3119
            # anno_tokenizer = anno_model.llm_engine.tokenizer.tokenizer
            self.anno_tokenizer = self.anno_model.get_tokenizer()

    def _init_chat_msg_template(self, examples, anno_cfg, **kwargs) -> list[None | dict[str, str]]:
        """
        Init the chat messages template for the annotation models according to template settings.
        :param examples: the examples to be sho wn to annotators.
        :param anno_cfg: the configuration of the annotation settings
        :param kwargs: other arguments, including
            1) use_api, whether to use LLM API as annotator
            2) labels, the labels used in this annotating stage.
            3) dialogue_style, the style of the dialogue. 'batch_qa' or 'multi_qa'
        :return:
        """
        # 0. examples
        if examples:
            examples_prompt = f"Here are some examples to help you understand the task better:\n ### Examples \n {examples}\n"
            prompt_kwargs = {'examples_prompt': examples_prompt}
        else:
            prompt_kwargs = {'examples_prompt': ''}

        # 1. system role
        system_role = ''
        if 'system_role' in anno_cfg.keys() and anno_cfg['system_role']:
            system_role = anno_cfg['system_role'] + '\n'
        prompt_kwargs.update({'system_role': system_role})

        # 2. task prompt
        task_prompt = ''
        if 'task_prompt' in anno_cfg.keys() and anno_cfg['task_prompt']:
            if isinstance(anno_cfg['task_prompt'], dict):
                task_prompt = anno_cfg['task_prompt'][kwargs['dialogue_style']]
            else:
                task_prompt = anno_cfg['task_prompt']
            task_prompt = f"Here is your task: \n ### Task \n {task_prompt}\n"
        prompt_kwargs.update({'task_prompt': task_prompt})

        # 3. types prompt
        types_prompt = ''
        if 'types_prompt' in anno_cfg.keys() and anno_cfg['types_prompt']:
            types_string = ''
            if 'labels' in kwargs.keys():
                labels = kwargs['labels']
            else:
                labels = self.labels.keys()
            for type_id, type in enumerate(labels):
                if self.natural_flag == 'natural':  # use natual format
                    type_string = self.labels[type]['natural']
                else:
                    type_string = type

                if 'des_format' in anno_cfg.keys() and anno_cfg['des_format'] == 'empty':  # don't show the description of the types
                    description = ''
                else:
                    if 'des_format' in anno_cfg.keys() and anno_cfg['des_format'] == 'simple':  # use simple description
                        description = self.labels[type]['description'].split('.')[:2]  # only show the first two sentences
                        description = ' '.join(description)
                    else:  # use full description
                        description = self.labels[type]['description']

                    description = type_string + ' ' + description
                types_string += '{idx}) {type}\n {description}\n'.format(idx=type_id+1, type=type_string, description=description)
            types_prompt = f"Given types : ### Types \n {types_string}\n"
        prompt_kwargs.update({'types_prompt': types_prompt})

        # 4. guidelines prompt
        guidelines_prompt = ''
        if 'guidelines' in anno_cfg.keys() and anno_cfg['guidelines']:
            guidelines = anno_cfg['guidelines']
            guidelines_prompt = f"In your annotation process, please follow these guidelines: \n ### Guidelines \n {guidelines}\n"
        prompt_kwargs.update({'guidelines': guidelines_prompt})

        sys_prompt = anno_cfg['prompt_template'].format(**prompt_kwargs)

        chat_message = [{"role": "system", "content": sys_prompt}]
        if kwargs['use_api'] and self.api_cfg['model'] == 'qwen-long':
            # see https://help.aliyun.com/document_detail/2788814.html?spm=a2c4g.2788811.0.0.1440240aUbuyYI#b7f81199e2laz
            # when use qwen-long, we should add an extra system message for role-play to the chat_message
            chat_message = [{'role': 'system', 'content': anno_cfg['system_role']}] + chat_message
        return chat_message

    def _pipeline_msg(self, annotator_cfg, anno_cfg, dataset_name, use_api=False) -> list[None | dict[str, str]]:
        """
        Get examples and init the chat messages template for the annotation models using 2-stage pipeline.

        :param annotator_cfg: The config of the annotation model.
        :param anno_cfg: the configuration of the annotation settings
        :param dataset_name: the name of the dataset.
        :param use_api: whether to use LLM API as annotator
        :return:
        """
        if annotator_cfg['chat']:
            pass
        else:
            if 'examples' in anno_cfg.keys() and anno_cfg['examples'][dataset_name]:
                examples = ''
                for idx, example in enumerate(anno_cfg['examples'][dataset_name]):
                    instance = anno_cfg['instance_template'].format(sentence=example['sentence'], output=example['output'])
                    examples += f'{idx + 1})\n{instance}\n'
            else:  # 0-shot
                examples = None
        return self._init_chat_msg_template(examples, anno_cfg=anno_cfg, use_api=use_api)

    def _st_msg(self, annotator_cfg, anno_cfg, dataset_name, use_api=False) -> list[None | dict[str, str]]:
        """
        Get examples and init the chat messages for the annotation models to extract entity directly by annotators according the single_type_prompt.

        :param annotator_cfg: The parameters of the annotation model.
        :param anno_cfg: the configuration of the annotation settings
        :param dataset_name: the name of the dataset.
        :param use_api: whether to use LLM API as annotator
        :return:
        """
        if annotator_cfg['chat']:
            pass
        else:
            if 'examples' in anno_cfg.keys() and anno_cfg['examples'][dataset_name]:
                examples = ''
                for idx, example in enumerate(anno_cfg['examples'][dataset_name]):
                    instance = anno_cfg['instance_template'].format(label=example['label'],
                                                                       sentence=example['sentence'],
                                                                       output=example['output'])
                    examples += f'{idx + 1})\n{instance}\n'
            else:   # 0-shot
                examples = None
        return self._init_chat_msg_template(examples, anno_cfg=anno_cfg, use_api=use_api)

    def _pipeline_fs_msg(self, annotator_cfg, anno_cfg, dataset_name, use_api=False) -> list[None | dict[str, str]]:
        """
        Get examples and init the chat messages for the annotation models using 2-stage pipeline with few-shot settings.
        Init examples from the support set sampled from the dataset.

        :param annotator_cfg: The parameters of the annotation model.
        :param anno_cfg: the configuration of the annotation settings
        :param dataset_name: the name of the dataset.
        :param use_api: whether to use LLM API as annotator.
        :return:
        """
        if annotator_cfg['chat']:
            pass
        else:
            examples = None
            spans_o_class = []  # store the spans of 'O' class
            span_nums = []  # count the number of gold spans for each example

            index = 0  # index of the examples
            # 1. for Non-O class
            # k-shot examples to show to the annotator
            k_shot = anno_cfg['k_shot']
            if k_shot != 0:
                examples = ''
                anno_cfg['support_set_dir'] = anno_cfg['support_set_dir'].format(dataset_name=dataset_name)
                if anno_cfg['gold_span']:
                    k_shot_file = os.path.join(anno_cfg['support_set_dir'], f'gold_span_{self.natural_flag}', f'train_support_set_{k_shot}_shot.jsonl')
                else:
                    k_shot_file = os.path.join(anno_cfg['support_set_dir'], f'span_{self.natural_flag}', f'train_support_set_{k_shot}_shot.jsonl')
                with jsonlines.open(k_shot_file) as reader:
                    for line in reader:
                        span_nums.append(len(line['spans_labels']))
                        spans = set()  # store the spans parsed by parsers
                        gold_spans = set()  # store the gold spans
                        for start, end, entity_mention, label_id in line['spans_labels']:
                            gold_spans.add((start, end, entity_mention))
                            # gold_span is a tuple like (start, end, gold_mention_span, gold_label)
                            start, end = int(start), int(end)
                            sentence = ' '.join(line['tokens'][:start] + ['[ ', entity_mention, ' ]'] + line['tokens'][end:])

                            label = self.id2label[int(label_id)]
                            output = f'{{"answer": "{label}"}}'
                            instance = anno_cfg['instance_template'].format(sentence=sentence, output=output)
                            examples += f'{index + 1})\n{instance}\n'
                            index += 1

                        for start, end, entity_mention in line['spans']:
                            spans.add((start, end, entity_mention))

                        # add examples of 'O' class
                        spans_o_class.append(spans - gold_spans)

                # 2. for 'O' class
                # sample the examples of 'O' class according to the number of gold spans
                # the more gold spans, the less probability to be sampled
                softmin = torch.nn.Softmin(dim=0)
                probability = softmin(torch.tensor(span_nums, dtype=torch.float32))
                sampled_idx = torch.topk(probability, k = anno_cfg['k_shot']).indices.tolist()  # the instance indices to be sampled
                # get the sampled spans of 'O' class and filter the empty spans
                sampled_idx = list(filter(lambda x: len(spans_o_class[x]) > 0, sampled_idx))
                sampled_idx = sampled_idx[:anno_cfg['k_shot']]  # Get the first k_shot elements
                with jsonlines.open(k_shot_file) as reader:
                    for idx, line in enumerate(reader):
                        if idx in sampled_idx:
                            start, end, entity_mention = random.choice(list(spans_o_class[idx]))  # randomly select a span in this instance
                            start, end = int(start), int(end)
                            sentence = ' '.join(line['tokens'][:start] + ['[ ', entity_mention, ' ]'] + line['tokens'][end:])
                            output = '{"answer": "O"}'
                            instance = anno_cfg['instance_template'].format(sentence=sentence, output=output)
                            examples += f'{index + 1})\n{instance}\n'
                            index += 1

        return self._init_chat_msg_template(examples, anno_cfg=anno_cfg, use_api=use_api)

    def _st_fs_msg(self, annotator_cfg, anno_cfg, dataset_name, use_api=False) -> list[None | dict[str, str]]:
        """
        Init the chat messages for the annotation models to extract entity directly by annotators according the single_type_prompt.
        Init examples from the support set sampled from the dataset.

        :param annotator_cfg: The parameters of the annotation model.
        :param anno_cfg: the configuration of the annotation settings
        :param dataset_name: the name of the dataset.
        :param use_api: whether to use LLM API as annotator
        :return:
        """
        if annotator_cfg['chat']:
            pass
        else:
            examples = None
            index = 0  # index of the examples
            k_shot = anno_cfg['k_shot']
            if k_shot != 0:
                examples = ''
                anno_cfg['support_set_dir'] = anno_cfg['support_set_dir'].format(dataset_name=dataset_name)
                if anno_cfg['gold_span']:
                    k_shot_file = os.path.join(anno_cfg['support_set_dir'], f'gold_span_{self.natural_flag}', f'train_support_set_{k_shot}_shot.jsonl')
                else:
                    k_shot_file = os.path.join(anno_cfg['support_set_dir'], f'span_{self.natural_flag}', f'train_support_set_{k_shot}_shot.jsonl')
                with jsonlines.open(k_shot_file) as reader:
                    for line in reader:
                        gold_mentions = dict()  # store the gold mentions according to the label
                        # 1. find the gold spans according to the label
                        for start, end, entity_mention, label_id in line['spans_labels']:
                            label = self.id2label[int(label_id)]
                            if label not in gold_mentions.keys():
                                gold_mentions[label] = {(int(start), int(end), entity_mention)}  # init the set
                            else:
                                gold_mentions[label].add((int(start), int(end), entity_mention))

                            # 2. replace the gold mentions with the formatted mentions
                            for label, mentions in gold_mentions.items():
                                # mentions is a set of gold mentions like {(start, end, gold_mention_span), ...}
                                # sort the mentions by the end position by ascending order
                                # e.g., {(7, 8, 'Mass.'), (5, 6, 'Westborough')} -> {(5, 6, 'Westborough'), (7, 8, 'Mass.'),}
                                mentions = sorted(mentions, key=lambda x: x[1], reverse=False)

                                output = []  # init the output for this label
                                pre_end = 0  # the end position of the previous mention
                                for start, end, mention in mentions:
                                    formatted_mention = f'@@ {mention} ##'
                                    output = output + line['tokens'][pre_end:start] + [formatted_mention]
                                    pre_end = end
                                output += line['tokens'][pre_end:]  # add the rest tokens, or line['tokens'][end:]

                                sent = ' '.join(line['tokens'])
                                output = '"' + ' '.join(output) + '"'
                                instance = anno_cfg['instance_template'].format(label=label, sentence=sent, output=output)
                                examples += f'{index + 1})\n{instance}\n'
                                index += 1
        return self._init_chat_msg_template(examples, anno_cfg=anno_cfg, use_api=use_api)

    def _mt_fs_msg(self, annotator_cfg, anno_cfg, dataset_name, use_api=False) -> list[None | dict[str, str]]:
        """
        Init the chat messages for the annotation models to extract entity directly by annotators according the multi_type_prompt.
        Init examples from the support set sampled from the dataset.

        :param annotator_cfg: The parameters of the annotation model.
        :param anno_cfg: the configuration of the annotation settings
        :param dataset_name: the name of the dataset.
        :param use_api: whether to use LLM API as annotator
        :return:
        """
        if annotator_cfg['chat']:
            pass
        else:
            examples = None
            k_shot = anno_cfg['k_shot']
            if k_shot != 0:
                # read all examples from the support set
                example_list = []
                anno_cfg['support_set_dir'] = anno_cfg['support_set_dir'].format(dataset_name=dataset_name)
                if anno_cfg['gold_span']:
                    k_shot_file = os.path.join(anno_cfg['support_set_dir'], f'gold_span_{self.natural_flag}', f'train_support_set_{k_shot}_shot.jsonl')
                else:
                    k_shot_file = os.path.join(anno_cfg['support_set_dir'], f'span_{self.natural_flag}', f'train_support_set_{k_shot}_shot.jsonl')
                with jsonlines.open(k_shot_file) as reader:
                    for line in reader:
                        sentence = ' '.join((line['tokens']))
                        output = '['
                        for start, end, entity_mention, label_id in line['spans_labels']:
                            label = self.id2label[int(label_id)]
                            output += f'("{label}", "{entity_mention}"),'
                        output += ']'
                        instance = anno_cfg['instance_template'].format(sentence=sentence, output=output)
                        example_list.append(instance)

                index = 0
                assert 'demo_times' in anno_cfg.keys(), "The demo_times is required for 'multi_type_prompt'. Defualt 1"
                examples = ''  # store the examples input to context
                for _ in range(anno_cfg['demo_times']):
                    for instance in example_list:
                        examples += f'{index + 1})\n{instance}\n'
                        index += 1
        return self._init_chat_msg_template(examples, anno_cfg=anno_cfg, use_api=use_api)

    def _cm_fs_msg(self, annotator_cfg, anno_cfg, dataset_name, use_api=False) -> list[None | dict[str, str]]:
        """
        Init the chat messages for the annotation models using candidate mention parsed by parsers and few-shot settings.
        Init examples from the support set sampled from the dataset.

        :param annotator_cfg: The parameters of the annotation model.
        :param anno_cfg: the configuration of the annotation settings
        :param dataset_name: the name of the dataset.
        :param use_api: whether to use LLM API as annotator
        :return:
        """
        if annotator_cfg['chat']:
            pass
        else:
            examples = None
            index = 0
            k_shot = anno_cfg['k_shot']
            if k_shot != 0:
                examples = ''
                anno_cfg['support_set_dir'] = anno_cfg['support_set_dir'].format(dataset_name=dataset_name)
                if anno_cfg['gold_span']:
                    k_shot_file = os.path.join(anno_cfg['support_set_dir'], f'gold_span_{self.natural_flag}', f'train_support_set_{k_shot}_shot.jsonl')
                else:
                    k_shot_file = os.path.join(anno_cfg['support_set_dir'], f'span_{self.natural_flag}', f'train_support_set_{k_shot}_shot.jsonl')
                with jsonlines.open(k_shot_file) as reader:
                    for line in reader:
                        sentence = ' '.join((line['tokens']))
                        candidate_mentions = ''
                        for start, end, entity_mention in line['spans']:  # shape: (start, end, entity_mention)
                            candidate_mentions += f'"{entity_mention}",'
                        output = '['
                        for start, end, entity_mention, label_id in line['spans_labels']:
                            label = self.id2label[int(label_id)]
                            output += f'("{label}", "{entity_mention}"),'
                        output += ']'
                        instance = anno_cfg['instance_template'].format(sentence=sentence, candidate_mentions=candidate_mentions, output=output)
                        examples += f'{index + 1})\n{instance}\n'
                        index += 1
        return self._init_chat_msg_template(examples, anno_cfg=anno_cfg, use_api=use_api)

    def _subset_type_fs_msg(self, annotator_cfg, anno_cfg, dataset_name, use_api=False, dialogue_style='batch_qa'):
        """
        Init the chat messages for the annotation models using subset types with few-shot settings.
        Init examples from the support set sampled from the dataset.

        :param annotator_cfg: The parameters of the annotation model.
        :param anno_cfg: the configuration of the annotation settings
        :param dataset_name: the name of the dataset.
        :param use_api: whether to use LLM API as annotator
        :param dialogue_style: the style of the dialogue. 'batch_qa' or 'multi_qa'
        :return: a list of chat message template for each label subset
        """
        # todo,为每个示例qurey添加一个Instruction
        if annotator_cfg['chat']:
            pass
        else:
            all_labels = list(self.label2id.keys())
            if 'O' in all_labels:
                all_labels.remove('O')
            if 0 < anno_cfg['subset_size'] < 1:
                subset_size = math.floor(len(self.label2id.keys()) * anno_cfg['subset_size'])
            else:
                subset_size = anno_cfg['subset_size']
            label_subsets = fu.get_label_subsets(all_labels, subset_size, anno_cfg['repeat_num'])
            examples = None
            k_shot = anno_cfg['k_shot']
            chat_msg_template_list = []  # store the chat message template for each label subset
            if k_shot != 0:
                anno_cfg['support_set_dir'] = anno_cfg['support_set_dir'].format(dataset_name=dataset_name)
                if anno_cfg['gold_span']:
                    k_shot_file = os.path.join(anno_cfg['support_set_dir'], f'gold_span_{self.natural_flag}',f'train_support_set_{k_shot}_shot.jsonl')
                else:
                    k_shot_file = os.path.join(anno_cfg['support_set_dir'], f'span_{self.natural_flag}',f'train_support_set_{k_shot}_shot.jsonl')

                instance_template = anno_cfg['instance_template'][dialogue_style]
                for label_subset in label_subsets:
                    examples = ''
                    index = 0  # example index
                    sents, empty_sents = [], []  # store the sentence for non-empty and empty outputs
                    with jsonlines.open(k_shot_file) as reader:
                        for line in reader:
                            sentence = ' '.join((line['tokens']))
                            output = '['
                            for start, end, entity_mention, label_id in line['spans_labels']:
                                label = self.id2label[int(label_id)]
                                if label in label_subset:
                                    output += f'("{label}", "{entity_mention}"),'
                            output += ']'
                            if output == '[]':
                                empty_sents.append(sentence)
                                continue
                            else:
                                sents.append(sentence)
                                instance = instance_template.format(sentence=sentence, output=output)
                                examples += f'{index + 1})\n{instance}\n'
                                index += 1
                    if len(empty_sents) > 0:  # random select empty outputs
                        select_num = len(label_subsets) if len(empty_sents) > len(label_subsets) else len(empty_sents)
                        for sentence in random.sample(empty_sents, select_num):
                            output = '[]'
                            instance = instance_template.format(sentence=sentence, output=output)
                            examples += f'{index + 1})\n{instance}\n'
                            index += 1
                    chat_msg_template_list.append(
                        self._init_chat_msg_template(examples, anno_cfg=anno_cfg, use_api=use_api, labels=label_subset, dialogue_style=dialogue_style)
                    )
            else:
                for label_subset in label_subsets:
                    chat_msg_template_list.append(
                        self._init_chat_msg_template(examples, anno_cfg=anno_cfg, use_api=use_api, labels=label_subset, dialogue_style=dialogue_style)
                    )

        return chat_msg_template_list

    def _subset_cand_fs_msg(self, annotator_cfg, anno_cfg, dataset_name, use_api=False, dialogue_style='batch_qa'):
        """
        Init the chat messages for the annotation models using subset candidate prompt with few-shot settings.
        Init examples from the support set sampled from the dataset.

        :param annotator_cfg: The parameters of the annotation model.
        :param anno_cfg: the configuration of the annotation settings
        :param dataset_name: the name of the dataset.
        :param use_api: whether to use LLM API as annotator
        :param dialogue_style: the style of the dialogue. 'batch_qa' or 'multi_qa'
        :return: a list of chat message template for each label subset
        """
        if annotator_cfg['chat']:
            pass
        else:
            all_labels = list(self.label2id.keys())
            if 'O' in all_labels:
                all_labels.remove('O')
            if 0 < anno_cfg['subset_size'] < 1:
                subset_size = math.floor(len(self.label2id.keys()) * anno_cfg['subset_size'])
            else:
                subset_size = anno_cfg['subset_size']
            label_subsets = fu.get_label_subsets(all_labels, subset_size, anno_cfg['repeat_num'])
            examples = None
            k_shot = anno_cfg['k_shot']
            if k_shot != 0:
                anno_cfg['support_set_dir'] = anno_cfg['support_set_dir'].format(dataset_name=dataset_name)
                if anno_cfg['gold_span']:
                    k_shot_file = os.path.join(anno_cfg['support_set_dir'], f'gold_span_{self.natural_flag}',f'train_support_set_{k_shot}_shot.jsonl')
                else:
                    k_shot_file = os.path.join(anno_cfg['support_set_dir'], f'span_{self.natural_flag}',f'train_support_set_{k_shot}_shot.jsonl')

                instance_template = anno_cfg['instance_template'][dialogue_style]
                examples = ''
                index = 0
                for label_subset in label_subsets:
                    sents, empty_sents = [], []  # store the sentence for non-empty and empty outputs
                    with jsonlines.open(k_shot_file) as reader:
                        for line in reader:
                            sentence = ' '.join((line['tokens']))
                            output = '['
                            for start, end, entity_mention, label_id in line['spans_labels']:
                                label = self.id2label[int(label_id)]
                                if label in label_subset:
                                    output += f'("{label}", "{entity_mention}"),'
                            output += ']'
                            if output == '[]':
                                empty_sents.append(sentence)
                                continue
                            else:
                                sents.append(sentence)
                                instance = instance_template.format(sentence=sentence, output=output)
                                examples += f'{index + 1})\n{instance}\n'
                                index += 1
                    if len(empty_sents) > 0:  # random select empty outputs
                        select_num = len(label_subsets) if len(empty_sents) > len(label_subsets) else len(empty_sents)
                        for sentence in random.sample(empty_sents, select_num):
                            output = '[]'
                            instance = instance_template.format(sentence=sentence, output=output)
                            examples += f'{index + 1})\n{instance}\n'
                            index += 1
        return self._init_chat_msg_template(examples, anno_cfg=anno_cfg, use_api=use_api, dialogue_style=dialogue_style)

    def _generate_chat_msg(self, instances, anno_cfg, chat_msg_template, anno_style, dialogue_style):
        """
        For batch chat.
        Generate chat messages for each instance. Meanwhile, init the labels for each instance.

        :param instances: The instances to be annotated.
        :param anno_cfg: the configuration of the annotation settings
        :param chat_msg_template: The chat message template for the annotating model.
        :param anno_style: The annotation style including 'single_type', 'multi_type', 'cand_mention_type', 'subset_type' or 'raw'
        :param dialogue_style, the style of the dialogue 'batch_qa' or 'multi_qa'
        :return:
        """
        for instance_id, tokens, spans, spans_labels in zip(instances['id'], instances['tokens'], instances['spans'], instances['spans_labels']):
            if anno_style == 'single_type':
                # generate chat message using the single_type_prompt to extract entity directly by annotators
                sentence = ' '.join(tokens)
                for label, label_id in self.label2id.items():
                    if label == 'O':
                        continue
                    chat_message = copy.deepcopy(chat_msg_template)
                    query = anno_cfg['instance_template'].format(label=label, sentence=sentence, output='')
                    if dialogue_style == 'batch_qa':
                        user_prompt = '\n### Query\n' + query
                    else:
                        user_prompt = query
                    chat_message.append({"role": "user", "content": user_prompt})
                    yield instance_id, label_id, chat_message, sentence
            elif anno_style == 'multi_type':
                # generate chat message using the multi_type_prompt to extract entity directly by annotators
                sentence = ' '.join(tokens)
                chat_message = copy.deepcopy(chat_msg_template)
                query = anno_cfg['instance_template'].format(sentence=sentence, output='')
                if dialogue_style == 'batch_qa':
                    user_prompt = '\n### Query\n' + query
                else:
                    user_prompt = query
                chat_message.append({"role": "user", "content": user_prompt})
                yield instance_id, chat_message, sentence
            elif anno_style == 'cand_mention_type':
                # generate chat message using the candidate mention parsed by parsers
                chat_message = copy.deepcopy(chat_msg_template)
                sentence = ' '.join(tokens)
                candidate_mentions = ''
                for start, end, entity_mention in spans:  # shape: (start, end, entity_mention)
                    candidate_mentions += f'"{entity_mention}",'
                query = anno_cfg['instance_template'].format(sentence=sentence,candidate_mentions=candidate_mentions, output='')
                if dialogue_style == 'batch_qa':
                    user_prompt = '\n### Query\n' + query
                else:
                    user_prompt = query
                chat_message.append({"role": "user", "content": user_prompt})
                yield instance_id, chat_message, sentence
            elif anno_style == 'subset_type':
                # generate chat message using the subset types
                # In this case, the chat msg template is a list of chat message template for each label subset
                instance_template = anno_cfg['instance_template'][dialogue_style]
                for chat_msg_temp in chat_msg_template:
                    chat_message = copy.deepcopy(chat_msg_temp)
                    sentence = ' '.join(tokens)
                    query = instance_template.format(sentence=sentence, output='')
                    if dialogue_style == 'batch_qa':
                        user_prompt = '\n### Query\n' + query
                    else:
                        user_prompt = query
                    chat_message.append({"role": "user", "content": user_prompt})
                    yield instance_id, chat_message, sentence
            elif anno_style == 'subset_cand':
                # generate chat message using the subset candidate prompt
                instance_template = anno_cfg['instance_template'][dialogue_style]
                chat_message = copy.deepcopy(chat_msg_template)
                sentence = ' '.join(tokens)
                query = instance_template.format(sentence=sentence, output='')
                if dialogue_style == 'batch_qa':
                    user_prompt = '\n### Query\n' + query
                else:
                    user_prompt = query
                chat_message.append({"role": "user", "content": user_prompt})
                yield instance_id, chat_message, sentence
            else:  # the 'raw' style
                # do not generate chat message given entity
                for idx, (start, end, entity_mention) in enumerate(spans):
                    start, end = int(start), int(end)
                    sentence = ' '.join(tokens[:start] + ['[ ', entity_mention, ' ]'] + tokens[end:])

                    chat_message = copy.deepcopy(chat_msg_template)
                    query = anno_cfg['instance_template'].format(sentence=sentence, output='')
                    if dialogue_style == 'batch_qa':
                        user_prompt = '\n### Query\n' + query
                    else:
                        user_prompt = query
                    chat_message.append({"role": "user", "content": user_prompt})
                    # if anno_cfg['gold_span'] is True, label is the ground truth label id
                    # yield the ID of the sentence, the mention span, the label id of the entity mention and the chat message
                    # else, label is the tuple of the ground truth span and its label, like (start, end, gold_mention_span, gold_label_id)
                    # yield the ID of the sentence, the mention span, gold span and the chat message
                    span = (str(start), str(end), entity_mention)
                    if anno_cfg['gold_span']:  # use the gold span from the annotation
                        # In this case, entity mentions and gold labels are one-to-one
                        label = spans_labels[idx]
                        yield instance_id, span, label, chat_message, sentence

                    else: # get the span from scratch by spaCy parsers
                        # In this case, entity mention and gold spans with labels are not one-to-one
                        yield instance_id, span, chat_message, sentence

    @staticmethod
    def _process_output(output_text, sentence, **kwargs):
        """
        process the output text to get the label of the entity mention.
        :param output_text: the text output by the annotator model
        :param sentence: the query sentence
        :param kwargs: other parameters including,
            1) anno_style, indicate the annotation style including 'single_type', 'multi_type', 'cand_mention_type', 'subset_type' or 'raw'
            2) analysis, whether to extract the analysis process in the output text
        :return: if the single_type_prompt is True, return the predicted spans and their labels.
        """
        import ast
        if not output_text:
            output_text = ''
        output_text = output_text.strip().replace('\_', '_')

        # if kwargs['anno_style'] == 'single_type, we recognize all entity mention in the output_text given the type
        if kwargs['anno_style'] == 'single_type':

            if 'analysis' in kwargs.keys() and kwargs['analysis']:  # add analysis process in the output
                # output text is formatted like '{{"analysis": "your analysis process in a concise manner", "answer": "your answer"}}'.
                # we need to extract JSON string from output.outputs[0].text first, then extract the answer from the JSON string
                json_pattern = [ r'\{(.*?)\}', r'\{\{(.*?)\}\}']  # the pattern to extract JSON string

                for pattern in json_pattern:
                    result = re.search(pattern, output_text, re.DOTALL)  # only extract the first JSON string
                    if result:
                        try:
                            json_string = '{' + result.group(1) + '}'
                            output_text = json.loads(json_string)['answer'].strip()
                            break
                        except (json.decoder.JSONDecodeError, KeyError):
                            continue

            # at last, we extract mention spans from the output text
            pattern = r'@@(.*?)##'
            matches = re.finditer(pattern, output_text, re.DOTALL)
            out_spans = []
            for match in matches:
                start_ch_idx, end_ch_idx = match.span(1)  # get the capture group 1, ('span')
                span = output_text[start_ch_idx:end_ch_idx].strip()  # clear the white space around the span

                start_ch_idx, end_ch_idx = match.span(0)  # get the capture group 0, ('@@ span ##')
                # To get the start position of the first word of the matched span in the original sentence,
                # we just need to count the number of spaces before the start character ('@') in the output text
                start = output_text[:start_ch_idx].count(' ')

                # To get the end position of the last word of the matched span in the original sentence,
                # we just need to count the number of spaces in the span
                end = start + span.count(' ') + 1 # end position of the span, excluded
                out_spans.append((str(start), str(end), span))

            return out_spans

        # kwargs['multi_type_prompt'] is True, we recognize all entity mention in the output_text given multiple types
        # we extract a list of tuples
        elif (kwargs['anno_style'] == 'multi_type' or kwargs['anno_style'] == 'cand_mention_type'
              or kwargs['anno_style'] == 'subset_type' or kwargs['anno_style'] == 'subset_cand'):
            out_spans = []
            pattern = r'\[(.*?)\]'  # the pattern to extract a list string
            result = re.search(pattern, output_text, re.DOTALL)  # only find the first list string
            try:
                tmp_spans = ast.literal_eval(result.group(0).strip())  # tmp_spans shapes like [(type 0, mention0),...]
                tmp_spans = filter(lambda e: e and len(e) == 2, tmp_spans)  # filter the invalid spans
            except Exception:  # the output_text is not valid
                pattern = r'\((.*?)\)'  # the pattern to extract a tuple
                result = re.findall(pattern, output_text, re.DOTALL)  # find all the tuples
                try:  # try to extract tuple directly
                    tmp_spans = []
                    for e in result:
                        e = e.split(',')
                        if len(e) == 2:
                            tmp_spans.append((e[0].strip(), e[1].strip()))
                    tmp_spans = filter(lambda e: e and len(e) == 2, tmp_spans)  # filter the invalid spans
                except Exception:
                    tmp_spans = []

            for label, mention in tmp_spans:
                founded_spans = fu.find_span(sentence, mention)
                out_spans += [(str(start), str(end), span, label) for start, end, span in set(founded_spans)]
            return out_spans
        # kwargs['anno_style'] == 'raw', we extract the labelof the entity mention
        else:
            json_pattern = [r'\{(.*?)\}', r'\{\{(.*?)\}\}']  # the pattern to extract JSON string
            # kwargs['single_type_prompt'] is False, we classify the given entity mention in the output_text
            # the out_label is just one label for the given entity mention
            out_label = 'O'  # we assign 'O' to label to this span if we cannot extract the JSON string

            # extract JSON string from output.outputs[0].text
            for pattern in json_pattern:
                result = re.search(pattern, output_text, re.DOTALL)  # only extract the first JSON string
                if result:
                    try:
                        json_string = '{' + result.group(1) + '}'
                        out_label = json.loads(json_string)['answer'].strip()
                        break
                    except Exception:
                        continue
            return out_label


    @retry(wait=wait_random(1,3), retry=retry_if_exception_type(Exception))
    def get_response(self, client, chat_message, **kwargs):
        """
        Get the response of the annotator using api.
        :param client: the client of the LLM API
        :param chat_message: the chat message to be sent to the api.
        :return:
        """
        # https://help.aliyun.com/zh/dashscope/developer-reference/compatibility-of-openai-with-dashscope/?spm=a2c4g.11186623.0.0.5fef6e432o2fr6
        import openai

        try:
            print('---------------')
            completion = client.chat.completions.create(
                model=self.api_cfg['model'],
                messages=chat_message,
                stream=kwargs['stream'],
                top_p=kwargs['top_p'],
                temperature=kwargs['temperature'],
                max_tokens=kwargs['max_tokens'],
            )
            output = completion.choices[0].message.content
            print(output)
        except openai.APIConnectionError as e:
            print("openai.APIConnectionError")
            print(e)
        except openai.RateLimitError as e:
            print("openai.RateLimitError")
            print(e)
        except openai.APIStatusError as e:
            print("openai.APIStatusError")
            print(e)
            print(e.status_code)
        except Exception as e:
            print("other exception")
            print(e)
        return output

    def annotate_by_one(self,dataset, anno_cfg, **kwargs):
        """
        Annotate the dataset by one specific annotator.
        :param dataset: the dataset to be annotated
        :param anno_cfg: the configuration of the annotation settings
        :param kwargs: other arguments, including
            2) dataset_name, the name of the dataset
            3) eval, whether to evaluate the annotation quality for each annotator
            4) cache, whether to cache the annotation results
            5) prompt_type, the type of the prompt, including single_type, mt_few_shot, raw, and so on
            6) sampling_strategy, the strategy to sample the test subset
            7) dialogue_style, the style of the dialogue 'batch_qa' or 'multi_qa'
        :return:
        """

        dataset_name = kwargs['dataset_name']
        # 0. Some settings
        # 0.1 init wandb
        if 'gold_span' in anno_cfg.keys() and anno_cfg['gold_span']:
            wandb.init(
                project=f'{dataset_name}_annotation_by_llm',
                config=self.annotator_cfg
            )

        # 0.3 get cache dir for result
        annotate_flag = True  # whether to annotate the dataset from scratch
        cache_dir = anno_cfg['cache_dir'].format(dataset_name=dataset_name)

        # 0.3.1 label format dir
        if 'gold_span' in anno_cfg.keys() and anno_cfg['gold_span']:
            label_format_dir = f'gold_span_{self.natural_flag}'
        else:
            label_format_dir = f'span_{self.natural_flag}'

        # 0.3.2 label description dir
        label_des_dir = '{}_description'.format(anno_cfg['des_format'])

        if kwargs['prompt_type'] == 'sb_fs' or kwargs['prompt_type'] == 'sc_fs':
            subset_size = '-size_{}'.format(anno_cfg['subset_size'])
            repeat_num = '-rep_{}'.format(anno_cfg['repeat_num'])
            prompt_type_dir = os.path.join(kwargs['prompt_type'], '{}{}{}'.format(kwargs['prompt_type'], subset_size, repeat_num))
        else:
            prompt_type_dir = kwargs['prompt_type']

        # 0.3.3 test subset sampling strategy dir
        sub_samp_dir = '{}_sampling'.format(kwargs['sampling_strategy'])

        # 0.3.4 dialogue style dir
        dialogue_style_dir = '{}'.format(kwargs['dialogue_style'])

        # 0.3.5 annotator name
        if self.use_api:
            model_name = self.api_cfg['model']
        else:
            model_name = self.annotator_cfg['name']
        annotator_name =  model_name + '-' + anno_cfg['name']
        task_dir = os.path.join(label_format_dir, prompt_type_dir, label_des_dir, sub_samp_dir, dialogue_style_dir, model_name)
        res_cache_dir = os.path.join(cache_dir, task_dir, annotator_name)
        print(f'result cache dir: {res_cache_dir}')

        try:
            print(f'Trying to load cache file from {res_cache_dir}')
            cache_result = load_from_disk(res_cache_dir)
            annotate_flag = False
        except FileNotFoundError:
            print(f'No cache file found in {res_cache_dir}')
            if not os.path.exists(res_cache_dir):
                os.makedirs(res_cache_dir)

        # 0.5 annotation style setting
        anno_style = kwargs['prompt_type'].split('_')[0]
        if anno_style == 'single' or anno_style == 'st':
            anno_style = 'single_type'
        elif anno_style == 'mt':
            anno_style = 'multi_type'
        elif anno_style == 'cand' or anno_style == 'cm':
            anno_style = 'cand_mention_type'
        elif anno_style == 'sb':
            anno_style = 'subset_type'
        elif anno_style == 'sc':
            anno_style = 'subset_cand'
        else:
            anno_style = 'raw'

        # annotation process
        if annotate_flag:
            # 1. Init the chat messages
            if anno_style == 'single_type':
                if 'k_shot' in anno_cfg.keys() and anno_cfg['k_shot'] >= 0:  # single_type_prompt with few-shot setting
                    chat_msg_template = self._st_fs_msg(annotator_cfg=self.annotator_cfg, anno_cfg=anno_cfg, dataset_name=dataset_name, use_api=self.use_api)
                else:  # single_type_prompt without few-shot setting
                    chat_msg_template = self._st_msg(annotator_cfg=self.annotator_cfg, anno_cfg=anno_cfg, dataset_name=dataset_name, use_api=self.use_api)
            elif anno_style == 'multi_type':
                if 'k_shot' in anno_cfg.keys() and anno_cfg['k_shot'] >= 0:
                    chat_msg_template = self._mt_fs_msg(annotator_cfg=self.annotator_cfg, anno_cfg=anno_cfg, dataset_name=dataset_name, use_api=self.use_api)
            elif anno_style == 'cand_mention_type':
                if 'k_shot' in anno_cfg.keys() and anno_cfg['k_shot'] >= 0:
                    chat_msg_template = self._cm_fs_msg(annotator_cfg=self.annotator_cfg, anno_cfg=anno_cfg, dataset_name=dataset_name, use_api=self.use_api)
            elif anno_style == 'subset_type':
                if 'k_shot' in anno_cfg.keys() and anno_cfg['k_shot'] >= 0:
                    chat_msg_template = self._subset_type_fs_msg(annotator_cfg=self.annotator_cfg, anno_cfg=anno_cfg, dataset_name=dataset_name, use_api=self.use_api)
            elif anno_style == 'subset_cand':
                if 'k_shot' in anno_cfg.keys() and anno_cfg['k_shot'] >= 0:
                    chat_msg_template = self._subset_cand_fs_msg(annotator_cfg=self.annotator_cfg, anno_cfg=anno_cfg, dataset_name=dataset_name, use_api=self.use_api)
            else:
                if 'k_shot' in anno_cfg.keys() and anno_cfg['k_shot'] >= 0:  # 2-stage pipeline with few-shot setting
                    chat_msg_template = self._pipeline_fs_msg(annotator_cfg=self.annotator_cfg, anno_cfg=anno_cfg, dataset_name=dataset_name, use_api=self.use_api)
                else:  # 2-stage pipeline without few-shot setting
                    chat_msg_template = self._pipeline_msg(annotator_cfg=self.annotator_cfg, anno_cfg=anno_cfg, dataset_name=dataset_name, use_api=self.use_api)

            # 2. batch process
            # 2.1 yield batch data
            if len(chat_msg_template) > 1:
                # when we use subset type prompt, chat_msg_template is a list of chat message template
                # we set the batch size to the number of label subsets
                # As a result, we can process one instance for each label subset in one batch
                batch_size = len(chat_msg_template)
            else:
                batch_size = self.annotator_cfg['anno_bs']
            pbar = tqdm(fu.batched(self._generate_chat_msg(instances=dataset,
                                                           anno_cfg=anno_cfg,
                                                           chat_msg_template=chat_msg_template,
                                                           anno_style=anno_style,
                                                           dialogue_style=kwargs['dialogue_style']),
                                   batch_size),
                        desc=f'annotating by {annotator_name}, dataset {dataset_name}')

            res_labels, res_label_ids = [], []  # store the output labels and label ids

            # if anno_cfg['gold_span'] is True, we use gold span from annotation
            # y_true stores the ground truth label ids, shaped like [label_id_0, label_id_1, ...]
            # else, we get the span from scratch by spaCy and stanza parsers
            # y_true stores the gold span and its label in a tuple, shaped like [(start, end, gold_mention_span, gold_label_id), ...]
            y_true = []
            pred_spans = [] # store the predicted spans and its label
            output_texts = []  # store the output text for each instance
            instance_results = []  # store the sentence, pred spans, gold results for each instance. Only used for multi/sigle type prompt
            for batch_id, batch in enumerate(pbar):
                batch_spans = []  # store the span for each batch
                batch_instance_ids = []  # store the instance ids for each batch
                batch_labels = []  # store the gold labels for each batch
                batch_chats = []  # store the chats for each batch
                batch_sents = []  # store the sentences for each batch
                batch_res_label_ids = []  # store the output label ids for each batch to evaluate

                # 3.1 store different information according to the different annotation settings
                if anno_style == 'single_type':
                    # if anno_cfg['single_type_prompt'] is True, batch is a tuple like ((instance_id_0, label_0, chat_0, sent_0), (instance_id_1, label_1, chat_1, sent_1),...)
                    for instance_id, label_id, chat, sent in batch:
                        batch_instance_ids.append(instance_id)
                        batch_labels.append(label_id)
                        batch_chats.append(chat)
                        batch_sents.append(sent)
                elif anno_style == 'multi_type' or anno_style == 'cand_mention_type' or anno_style == 'subset_type' or anno_style =='subset_cand':
                    # batch is a tuple like ((instance_id_0, chat_0, sent_0), (instance_id_1, chat_1, sent_1),...)
                    for instance_id, chat, sent in batch:
                        batch_instance_ids.append(instance_id)
                        batch_chats.append(chat)
                        batch_sents.append(sent)
                else:
                    if anno_cfg['gold_span']:
                        # if anno_cfg['gold_span'] is true,
                        # batch is a tuple like ((instance_id_0, span_0, label_0, chat_0, sent_0),(instance_id_1,span_1, label_1, chat_1, sent_1)...)
                        for instance_id, span, label, chat, sent in batch:
                            batch_instance_ids.append(instance_id)
                            batch_spans.append(span)
                            batch_labels.append(label)
                            batch_chats.append(chat)
                            batch_sents.append(sent)
                            y_true.append(label)
                    else:
                        # else, batch is a tuple like ((instance_id_0, span_0, chat_0),(instance_id_1,span_1, chat_1)...)
                        # we  get all gold span labels after annotation
                        for instance_id, span, chat, sent in batch:
                            batch_instance_ids.append(instance_id)
                            batch_spans.append(span)
                            batch_chats.append(chat)
                            batch_sents.append(sent)

                # 2.2 get the response of the annotator
                if kwargs['dialogue_style'] == 'batch_qa':
                    if self.use_api:  # use LLM API and batch_qa
                        outputs = []
                        for chat in batch_chats:
                            output = self.get_response(client=self.client,
                                                       chat_message=chat,
                                                       stream=self.annotator_cfg['stream'],
                                                       temperature=self.annotator_cfg['anno_temperature'],
                                                       top_p=self.annotator_cfg['anno_top_p'],
                                                       max_tokens=self.annotator_cfg['anno_max_tokens'])
                            outputs.append(output)
                    else:  # use local annotator and batch_qa
                        # we should use tokenizer.apply_chat_template to add generation template to the chats explicitly
                        templated_batch_chats = self.anno_tokenizer.apply_chat_template(batch_chats, add_generation_prompt=True,tokenize=False)
                        outputs = self.anno_model.generate(templated_batch_chats, self.sampling_params)  # annotate
                        # for test
                        # test_answer = []
                        # for output in outputs:
                        #     test_answer.append({'prompt': output.prompt, 'output': output.outputs[0].text})
                        outputs = [e.outputs[0].text for e in outputs]
                elif kwargs['dialogue_style'] == 'multi_qa':
                    # todo,subset type 和 subset candidate还没适配multi_qa.
                    outputs = []

                    # for all chats in a batch,
                    # chat[0] is the system role message, they are all the same
                    # chat[1] or chat[-1] is the user role message which is the query, they are all different
                    multi_qa_chat = [batch_chats[0][0],]  # use the first chat's system role message as the initial chat

                    for chat in batch_chats:
                        query = chat[-1]['content']
                        multi_qa_chat.append({"role": "user", "content": query})
                        if self.use_api:  # use LLM API and multi_qa
                            output_text = self.get_response(client=self.client,
                                                       chat_message=multi_qa_chat,
                                                       stream=self.annotator_cfg['stream'],
                                                       temperature=self.annotator_cfg['anno_temperature'],
                                                       top_p=self.annotator_cfg['anno_top_p'],
                                                       max_tokens=self.annotator_cfg['anno_max_tokens'])
                        else:  # use local annotator and multi_qa
                            templated_multi_qa_chat = self.anno_tokenizer.apply_chat_template(multi_qa_chat, add_generation_prompt=True,tokenize=False)
                            tmp_outputs = self.anno_model.generate(templated_multi_qa_chat, self.sampling_params)  # len(tmp_outputs) == 1
                            output_texts = [e.outputs[0].text for e in tmp_outputs]  # len(output_texts) == 1
                            output_text = output_texts[0]
                        outputs.append(output_text)

                        # process the output
                        out_spans = self._process_output(output_text, batch_sents[0], anno_style=anno_style)
                        context_output = '['  # process output and append to chat message as context
                        tmp_pred_spans = []  # store prediction spans for each instance
                        for start, end, entity_mention, label in set(out_spans):
                            if label not in self.label2id.keys():
                                out_label_id = self.label2id['O']  # the label output is invalid, we assign 'O' label to this entity
                                # do not append the invalid label to the context
                            else:
                                out_label_id = self.label2id[label]  # valid label
                                context_output += f'("{label}", "{entity_mention}"),'  # append the entity mention to the context
                            pred_spans.append((str(start), str(end), entity_mention, str(out_label_id)))
                            tmp_pred_spans.append((str(start), str(end), entity_mention, str(out_label_id)))
                        context_output += ']'
                        instance_results.append({'id': instance_id, 'sent': sent, 'pred_spans': tmp_pred_spans})
                        # append processed output to chat message as context
                        multi_qa_chat.append({"role": "assistant", "content": context_output})

                # 2.3 process the output
                for out_idx, (instance_id, output, sent) in enumerate(zip(batch_instance_ids, outputs, batch_sents)):

                    output_texts.append(output)
                    if anno_style == 'single_type':
                        if 'analysis' in anno_cfg.keys() and anno_cfg['analysis']:
                            out_spans = self._process_output(output, sent, anno_style=anno_style, analysis=anno_cfg['analysis'])
                        else:
                            out_spans = self._process_output(output, sent, anno_style=anno_style)
                        if len(out_spans) == 0:
                            continue
                        out_label_id = batch_labels[out_idx]
                        tmp_pred_spans = [(*out_span, str(out_label_id)) for out_span in set(out_spans)]
                        pred_spans += tmp_pred_spans
                        instance_results.append({'id':instance_id, 'sent': sent, 'pred_spans': tmp_pred_spans})
                    elif anno_style == 'multi_type' or anno_style == 'cand_mention_type' or anno_style == 'subset_type' or anno_style == 'subset_cand':
                        # if multi_qa, we have processed the output immediately after getting the response in 3.2
                        # so we just skip the processing here
                        # only 'batch_qa' need to process the output here
                        if kwargs['dialogue_style'] == 'multi_qa':
                            continue

                        # the method of processing output of cand_mention_prompt and 'subset_type_prompt' are same to that of 'multi_type_prompt'
                        out_spans = self._process_output(output, sent, anno_style=anno_style)
                        if len(out_spans) == 0:
                            continue
                        tmp_pred_spans = []
                        for start, end, span, label in set(out_spans):
                            if label not in self.label2id.keys():
                                label = 'O'
                            out_label_id = self.label2id[label]
                            pred_spans.append((str(start), str(end), span, str(out_label_id)))
                            tmp_pred_spans.append((str(start), str(end), span, str(out_label_id)))
                        instance_results.append({'id':instance_id, 'sent': sent, 'pred_spans': tmp_pred_spans})
                    else:
                        out_label = self._process_output(output, sent, anno_style=anno_style)
                        if out_label not in self.label2id.keys():
                            # check if the output label is redundant
                            tmp_out_label_0 = out_label.split(' ')[0].strip()
                            tmp_out_label_1 = out_label.split(',')[0].strip()
                            if tmp_out_label_0 in self.label2id.keys():
                                out_label = tmp_out_label_0
                            elif tmp_out_label_1 in self.label2id.keys():
                                out_label = tmp_out_label_1
                            else:
                                out_label = 'O'

                        out_label_id = self.label2id[out_label]
                        res_labels.append(out_label)
                        res_label_ids.append(out_label_id)
                        batch_res_label_ids.append(out_label_id)
                        out_span = batch_spans[out_idx]
                        pred_spans.append((*out_span, str(out_label_id)))

                    # 2.4 evaluate batch results
                    if 'gold_span' in anno_cfg.keys() and anno_cfg['gold_span']:
                        wandb.log({'f1-macro': f1_score(y_true=batch_labels, y_pred=batch_res_label_ids, average='macro', zero_division=0),
                                   'precision-weighted': precision_score(y_true=batch_labels, y_pred=batch_res_label_ids, average='weighted', zero_division=0),
                                   'recall-weighted': recall_score(y_true=batch_labels, y_pred=batch_res_label_ids, average='weighted', zero_division=0),
                                   'accuracy & f1-micro': accuracy_score(y_true=batch_labels, y_pred=batch_res_label_ids),
                                   'matthews_corrcoef': matthews_corrcoef(y_true=batch_labels, y_pred=batch_res_label_ids),
                                   'cohen_kappa_score': cohen_kappa_score(y1=batch_labels, y2=batch_res_label_ids)
                                     })

            # 3. cache and evaluate the annotation result
            if 'gold_span' in anno_cfg.keys() and anno_cfg['gold_span']:
                cache_result = {
                    "y_true": y_true,  # the ground truth label ids, shaped like [label_id_0, label_id_1, ...]
                    "labels": res_labels,  # the output labels, shaped like [label_0, label_1, ...]
                    "label_ids": res_label_ids,  # the output label ids, shaped like [label_id_0, label_id_1, ...]
                    "output_text": output_texts, # shape like [out_text_0, out_text1, ...]
                }

            else:
                # y_true is shape of [(start, end, gold_mention_span, gold_label_id), ...]
                # pred_spans is shape of [(start, end, pred_mention_span, pred_label_id), ...],
                # they are not one-to-one, so we convert them into 2-d list
                y_true = [span_label for spans_labels in dataset['spans_labels'] for span_label in spans_labels]  # flatten the spans_labels
                res_y_true = [y_true]
                res_pred_spans = [pred_spans]
                res_output_texts = [output_texts]

                if len(instance_results) > 0:  # for multi/single type prompt, we have instance_results
                    ins_res = [instance_results]
                    cache_result = {
                        "y_true": res_y_true, # the gold span and its label, shaped like [(start, end, gold_mention_span, gold_label_id), ...]
                        "pred_spans": res_pred_spans, # the predicted spans and its label, shaped like [(start, end, pred_mention_span, pred_label_id), ...]
                        "output_text": res_output_texts, # the output text for each instance, shaped like [out_text_0, out_text1, ...]
                        "instance_results": ins_res  # the sentence, pred spans, gold results for each instance, shaped like [{'sent': sent_0, 'pred_spans': [(start, end, span, label), ...]}, ...]
                    }
                else:  # for other settings, we do not have instance_results
                    cache_result = {
                        "y_true": res_y_true,  # the gold span and its label, shaped like [(start, end, gold_mention_span, gold_label_id), ...]
                        "pred_spans": res_pred_spans,  # the predicted spans and its label, shaped like [(start, end, pred_mention_span, pred_label_id), ...]
                        "output_text": res_output_texts  # # the output text for each instance, shaped like [out_text_0, out_text1, ...]
                    }

            cache_result = Dataset.from_dict(cache_result)
            if kwargs['cache']:
                cache_result.save_to_disk(res_cache_dir)

        # 4. evaluation
        if kwargs['eval']:
            if anno_cfg['gold_span']:
                self.evaluate(cache_result['y_true'], cache_result['label_ids'],
                              anno_cfg=anno_cfg,
                              dataset_name=dataset_name,
                              annotator_name=annotator_name,
                              task_dir=task_dir,
                              anno_style=anno_style)
            else:
                # important! we must remove the '0'('O' label) span from the pred_spans before evaluation
                pred_spans = [span for span in cache_result['pred_spans'][0] if int(span[-1]) != self.label2id['O']]
                self.evaluate(cache_result['y_true'][0], pred_spans,
                              anno_cfg=anno_cfg,
                              dataset_name=dataset_name,
                              annotator_name=annotator_name,
                              task_dir=task_dir,
                              anno_style=anno_style)

        # os.kill(os.getpid(), signal.SIGKILL)  # kill itself to release the GPU memory
        return  cache_result

    def evaluate(self, y_true, y_pred, anno_cfg, **kwargs):
        """
        Evaluate the annotation results by an annotator.
        :param y_true: if anno_cfg['gold_span'] is True, we use gold span from annotation, y_true stores the ground truth label ids, shaped like
        [label_id_0, label_id_1, ...]. Else, we get the span from scratch by parser, y_true stores the gold span and their labels in a tuple,
        shaped like [(start, end, gold_mention_span, gold_label_id), ...].
        :param y_pred: if anno_cfg['gold_span'] is True, y_pred stores the predicted label ids. Else, y_pred stores the predicted spans and their labels.
        :param anno_cfg: the configuration of the annotation settings
        :param kwargs: other arguments, including
            1) dataset_name,  the name of the dataset.
            2) annotator_name,  the name of the annotator LLM.
            3) task_dir, the directory of the task
            4) anno_style, the style of the annotation, including single_type, multi_type, cand_mention_type, subset_type, and raw.
        :return:
        """
        eval_dir = anno_cfg['eval_dir'].format(dataset_name=kwargs['dataset_name'])
        res_cache_dir = os.path.join(eval_dir, kwargs['task_dir'])
        if not os.path.exists(res_cache_dir):
            os.makedirs(res_cache_dir)
        res_file = os.path.join(res_cache_dir, '{}_res.txt'.format(kwargs['annotator_name']))
        res_by_class_file = os.path.join(res_cache_dir, '{}_res_by_class.csv'.format(kwargs['annotator_name']))
        print(f'saved the evaluation results to {res_file}')
        print(f'saved the evaluation results by class to {res_by_class_file}')

        if anno_cfg['gold_span']:
            # compute all classification metrics
            eval_results = {'f1-macro': f1_score(y_true=y_true, y_pred=y_pred, average='macro'),
                            'precision-weighted': precision_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0),
                            'recall-weighted': recall_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0),
                            'accuracy & f1-micro': accuracy_score(y_true=y_true, y_pred=y_pred),
                            'matthews_corrcoef': matthews_corrcoef(y_true=y_true, y_pred=y_pred),
                            'cohen_kappa_score': cohen_kappa_score(y1=y_true, y2=y_pred)}
        else:
            # compute span-level metrics
            eval_results = fu.compute_span_f1(copy.deepcopy(y_true),  copy.deepcopy(y_pred))
            fu.compute_span_f1_by_labels(copy.deepcopy(y_true), copy.deepcopy(y_pred), id2label=self.id2label, res_file=res_by_class_file)

        with open(res_file, 'w') as f:
            for metric, res in eval_results.items():
                f.write(f'{metric}: {res}\n')

        return