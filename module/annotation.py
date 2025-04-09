import copy
import json
import math
import random
import re
import os
import xlsxwriter
import jsonlines
import time
import openai
import uuid

from pathlib import Path
from openai import OpenAI
from datasets import load_from_disk, Dataset
from tqdm import tqdm
from module import func_util as fu
from module.label import Label

logger = fu.get_logger('Annotation')

class Annotator:
    def __init__(self, annotator_cfg, api_cfg, just_test=False):
        """
        Initialize the annotator model.

        :param annotator_cfg: the configuration of the local annotator model.
        :param api_cfg: the configuration of the LLM API.
        :param annotator_cfg:
        :param just_test: If True, we just want to test something in the Annotation, so that we don't need to load the model.
        """
        self.use_api = True if api_cfg else False
        self.batch_infer = api_cfg['batch_infer']
        self.annotator_cfg = annotator_cfg
        self.api_cfg = api_cfg if api_cfg else None

        # 1. GPU setting
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # avoid parallelism in tokenizers
        cuda_devices = [str(i) for i in range(annotator_cfg['tensor_parallel_size'])]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_devices)

        # 2. Init the annotating model
        if not just_test:
            logger.info('----- Init LLM -----')
            if self.use_api:
                self.client = OpenAI(
                    api_key=os.getenv(self.api_cfg['api_key']),
                    base_url=self.api_cfg['base_url'],  # base_url
                    max_retries=3,
                )
            else:
                from vllm import LLM, SamplingParams
                # if not use api, we employ local annotator using vllm
                # https://docs.vllm.ai/en/latest/getting_started/quickstart.html
                self.anno_model = LLM(model=annotator_cfg['checkpoint'],
                                      tensor_parallel_size=annotator_cfg['tensor_parallel_size'],
                                      dtype=annotator_cfg['dtype'],
                                      gpu_memory_utilization=annotator_cfg['gpu_memory_utilization'],
                                      trust_remote_code=True,
                                      # https://github.com/vllm-project/vllm/issues/6723
                                      # set explicitly enable_chunked_prefill to False For Volta GPU
                                      enable_chunked_prefill=False)
                self.sampling_params = SamplingParams(temperature=annotator_cfg['anno_temperature'],
                                                      top_p=annotator_cfg['anno_top_p'],
                                                      max_tokens=annotator_cfg['anno_max_tokens'],
                                                      repetition_penalty=annotator_cfg['repetition_penalty'])

                # get anno_model's tokenizer to apply the chat template
                # https://github.com/vllm-project/vllm/issues/3119
                # anno_tokenizer = anno_model.llm_engine.tokenizer.tokenizer
                self.anno_tokenizer = self.anno_model.get_tokenizer()

class Annotation(Label):
    """
    The Annotation class is used to annotate the data.
    """
    def __init__(self, annotator, labels_cfg, natural_form=False):
        """
        Initialize the annotation model.

        :param annotator: the annotator model.
        :param labels_cfg: the configuration of the label_cfgs.
        :param natural_form: whether the labels are in natural language form.

        """
        # 0. cfg initialization
        super().__init__(labels_cfg, natural_form)

        self.annotator = annotator
        self.natural_flag = 'natural' if natural_form else 'bio'  # use natural labels or bio labels
        self.workbook = xlsxwriter.Workbook('metrics.xlsx')  # write metric to excel
        self.worksheet = self.workbook.add_worksheet()  # default 'Sheet 1'

    def _init_chat_msg_template(self, examples, annotator_cfg, anno_cfg, **kwargs) -> list[None | dict[str, str]]:
        """
        Init the chat messages template for the annotation models according to template settings.
        :param examples: the examples to be sho wn to annotators.
        :param annotator_cfg: The parameters of the annotation model.
        :param anno_cfg: the configuration of the annotation settings
        :param kwargs: other arguments, including
            1) use_api, whether to use LLM API as annotator
            2) labels, the labels used in this annotating stage.
            3) dialogue_style, the style of the dialogue. 'batch_qa' or 'multi_qa'
            4) task_label, specify the label in the task prompt, this is used for single_type_prompt
        :return:
        """
        prompt_template = anno_cfg['prompt_template']
        # 0. examples
        if examples:
            examples_prompt = f"Here are some examples to help you understand the task better:\n ### Examples \n {examples}\n"
            prompt_kwargs = {'examples_prompt': examples_prompt}
        else:
            prompt_kwargs = {'examples_prompt': ''}

        # 1. system role
        system_role = ''
        if 'system_role' in prompt_template.keys() and prompt_template['system_role']:
            system_role = prompt_template['system_role'] + '\n'
        prompt_kwargs.update({'system_role': system_role})

        # 2. task prompt
        task_prompt = ''
        if 'task_prompt' in prompt_template.keys() and prompt_template['task_prompt']:
            if isinstance(prompt_template['task_prompt'], dict):
                task_prompt = prompt_template['task_prompt'][kwargs['dialogue_style']]
            else:
                task_prompt = prompt_template['task_prompt']

            if 'task_label' in kwargs.keys():
                task_label = kwargs['task_label']
                task_prompt = task_prompt.format(task_label=task_label)
            task_prompt = f"Here is your task: \n ### Task \n {task_prompt}\n"
        prompt_kwargs.update({'task_prompt': task_prompt})

        # 3. types prompt
        types_prompt = ''
        if 'types_prompt' in prompt_template.keys() and prompt_template['types_prompt']:
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
        if 'guidelines' in prompt_template.keys() and prompt_template['guidelines']:
            guidelines = prompt_template['guidelines']
            guidelines_prompt = f"In your annotation process, please follow these guidelines: \n ### Guidelines \n {guidelines}\n"
        prompt_kwargs.update({'guidelines': guidelines_prompt})

        sys_prompt = prompt_template['prompt_template'].format(**prompt_kwargs)

        if kwargs['use_api']:  # when use api, almost all models are chat models
            chat_message = [{"role": "system", "content": sys_prompt}]
        elif annotator_cfg['chat']:  # for local chat model like qwen, it has 'system' role message
            chat_message = [{"role": "system", "content": sys_prompt}]
        else:  # for non-chat models like mistral, it doesn't have 'system' role message
            chat_message = [{"role": "user", "content": sys_prompt}]
        return chat_message

    def _st_fs_msg(self, annotator_cfg, anno_cfg, use_api, **kwargs) -> list[None | dict[str, str]]:
        """
        Init the chat messages for the annotation models to extract entity directly by annotators according the single_type_prompt.
        Init examples from the support set sampled from the dataset.
        return a list of chat message template for each label.
        chat_msg_template_list[0] is for the first label, chat_msg_template_list[1] is for the second label, and so on.

        :param annotator_cfg: The parameters of the annotation model.
        :param anno_cfg: the configuration of the annotation settings
        :param use_api: whether to use LLM API as annotator
        :param kwargs: other parameters, including,
            1) dataset_name: the name of the dataset.
        :return:
        """
        if annotator_cfg['chat']:
            pass
        else:
            examples = None
            k_shot = anno_cfg['k_shot']
            chat_msg_template_list = []  # store the chat message template for each label
            all_labels = list(self.label2id.keys())
            if 'O' in all_labels:
                all_labels.remove('O')
            if k_shot != 0:
                anno_cfg['support_set_dir'] = anno_cfg['support_set_dir'].format(dataset_name=kwargs['dataset_name'])
                k_shot_file = os.path.join(anno_cfg['support_set_dir'], f'span_{self.natural_flag}', f'train_support_set_{k_shot}_shot.jsonl')
                prompt_template = anno_cfg['prompt_template']

                for target_label in all_labels:  # init examples and system messages for each label
                    examples = ''
                    index = 0  # example index
                    sents, empty_sents = [], []  # store the sentence for non-empty and empty outputs
                    with jsonlines.open(k_shot_file) as reader:
                        for line in reader:
                            gold_mentions = set()  # store the gold mentions according to the target label
                            sentence = ' '.join((line['tokens']))  # sentence

                            # 1. find the gold spans according to the target label
                            for start, end, entity_mention, label_id in line['spans_labels']:
                                label = self.id2label[int(label_id)]
                                if label == target_label:
                                    gold_mentions.add((int(start), int(end), entity_mention))

                            # 2. replace the gold mentions with the formatted mentions
                            # gold mentions like {(start, end, gold_mention_span), ...}
                            # sort the mentions by the end position by ascending order
                            # e.g., {(7, 8, 'Mass.'), (5, 6, 'Westborough')} -> {(5, 6, 'Westborough'), (7, 8, 'Mass.'),}
                            gold_mentions = sorted(gold_mentions, key=lambda x: x[1], reverse=False)

                            output = []  # init the output for this label
                            pre_end = 0  # the end position of the previous mention
                            if len(gold_mentions) >= 1:
                                sents.append(sentence)
                                for start, end, mention in gold_mentions:
                                    formatted_mention = f'@@ {mention} ##'
                                    output = output + line['tokens'][pre_end:start] + [formatted_mention]
                                    pre_end = end
                                output += line['tokens'][pre_end:]  # add the rest tokens, or line['tokens'][end:]
                                output = '"' + ' '.join(output) + '"'
                                instance = prompt_template['instance_template'].format(label=label, sentence=sentence, output=output)
                                examples += f'{index + 1})\n{instance}\n'
                                index += 1
                            else:  # no gold mentions
                                empty_sents.append(sentence)

                    if len(empty_sents) > 0:  # randomly select empty outputs
                        select_num = 3 if len(empty_sents) > 3 else len(empty_sents)
                        for sentence in random.sample(empty_sents, select_num):
                            output = '"' + sentence + '"'
                            instance = prompt_template['instance_template'].format(sentence=sentence, output=output)
                            examples += f'{index + 1})\n{instance}\n'
                            index += 1
                    chat_msg_template_list.append(
                        self._init_chat_msg_template(examples,
                                                     annotator_cfg=annotator_cfg,
                                                     anno_cfg=anno_cfg,
                                                     use_api=use_api,
                                                     task_label=target_label)
                    )
            else:
                for target_label in all_labels:
                    chat_msg_template_list.append(
                        self._init_chat_msg_template(examples,
                                                     annotator_cfg=annotator_cfg,
                                                     anno_cfg=anno_cfg,
                                                     use_api=use_api,
                                                     task_label=target_label)
                    )
        return chat_msg_template_list

    def _mt_fs_msg(self, annotator_cfg, anno_cfg, use_api, **kwargs) -> list[None | dict[str, str]]:
        """
        Init the chat messages for the annotation models to extract entity directly by annotators according the multi_type_prompt.
        Init examples from the support set sampled from the dataset.

        :param annotator_cfg: The parameters of the annotation model.
        :param anno_cfg: the configuration of the annotation settings.
        :param use_api: whether to use LLM API as annotator.
        :param kwargs: other parameters, including,
            1) dataset_name: the name of the dataset.
            2) label_mention_map_portion, the portion of the corrected label-mention pair. Default is 1, which means all the label-mention pairs are correct.
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
                anno_cfg['support_set_dir'] = anno_cfg['support_set_dir'].format(dataset_name=kwargs['dataset_name'])
                prompt_template = anno_cfg['prompt_template']

                k_shot_file = os.path.join(anno_cfg['support_set_dir'], f'span_{self.natural_flag}', f'train_support_set_{k_shot}_shot.jsonl')
                with jsonlines.open(k_shot_file) as reader:
                    for line in reader:
                        sentence = ' '.join((line['tokens']))
                        output = '['
                        label_mention_pairs = fu.get_label_mention_pairs(line['spans_labels'],
                                                                         kwargs['label_mention_map_portion'],
                                                                         self.id2label)
                        for start, end, entity_mention, label_id in label_mention_pairs:
                            label = self.id2label[int(label_id)]
                            output += f'("{label}", "{entity_mention}"),'
                        output += ']'
                        instance = prompt_template['instance_template'].format(sentence=sentence, output=output)
                        example_list.append(instance)

                index = 0
                assert 'demo_times' in anno_cfg.keys(), "The demo_times is required for 'multi_type_prompt'. Defualt 1"
                examples = ''  # store the examples input to context
                for _ in range(anno_cfg['demo_times']):
                    for instance in example_list:
                        examples += f'{index + 1})\n{instance}\n'
                        index += 1
        return self._init_chat_msg_template(examples, annotator_cfg=annotator_cfg, anno_cfg=anno_cfg, use_api=use_api)

    def _subset_type_fs_msg(self, annotator_cfg, anno_cfg, use_api, **kwargs):
        """
        Init the chat messages for the annotation models using subset types with few-shot settings.
        Init examples from the support set sampled from the dataset.
        Inn the st_fs setting, the LLM can only be provided the subset label space.

        :param annotator_cfg: The parameters of the annotation model.
        :param anno_cfg: the configuration of the annotation settings.
        :param use_api: whether to use LLM API as annotator.
        :param kwargs: other parameters, including,
            1) dataset_name: the name of the dataset.
            2) dialogue_style: the style of the dialogue. 'batch_qa' or 'multi_qa'
        :return: a list of chat message template for each label subset
        """
        if annotator_cfg['chat']:
            pass
        else:
            all_labels = list(self.label2id.keys())
            if 'O' in all_labels:
                all_labels.remove('O')

            label_subsets = fu.get_label_subsets(all_labels, anno_cfg['subset_size'], anno_cfg['repeat_num'])
            examples = None
            k_shot = anno_cfg['k_shot']
            chat_msg_template_list = []  # store the chat message template for each label subset
            if k_shot != 0:
                anno_cfg['support_set_dir'] = anno_cfg['support_set_dir'].format(dataset_name=kwargs['dataset_name'])
                k_shot_file = os.path.join(anno_cfg['support_set_dir'], f'span_{self.natural_flag}',f'train_support_set_{k_shot}_shot.jsonl')

                instance_template = anno_cfg['prompt_template']['instance_template'][kwargs['dialogue_style']]
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
                        self._init_chat_msg_template(examples,
                                                     annotator_cfg=annotator_cfg,
                                                     anno_cfg=anno_cfg,
                                                     use_api=use_api,
                                                     labels=label_subset,  # only provide the label subset, not all labels
                                                     dialogue_style=kwargs['dialogue_style'])
                    )
            else:
                for label_subset in label_subsets:
                    chat_msg_template_list.append(
                        self._init_chat_msg_template(examples,
                                                     annotator_cfg=annotator_cfg,
                                                     anno_cfg=anno_cfg,
                                                     use_api=use_api,
                                                     labels=label_subset,
                                                     dialogue_style=kwargs['dialogue_style'])
                    )

        return chat_msg_template_list

    def _subset_cand_fs_msg(self, annotator_cfg, anno_cfg, use_api, **kwargs):
        """
        Init the chat messages for the annotation models using subset candidate prompt with few-shot settings.
        Init examples from the support set sampled from the dataset.
        We provide the whole labels space in the types prompt in the sc_fs settings.

        :param annotator_cfg: The parameters of the annotation model.
        :param anno_cfg: the configuration of the annotation settings.
        :param use_api: whether to use LLM API as annotator.
        :param kwargs: other parameters, including,
            1) dataset_name: the name of the dataset.
            2) dialogue_style: the style of the dialogue. 'batch_qa' or 'multi_qa'
            3) ignore_sent: whether to ignore the sentence in the chat message. If True, the sentence will be shown as '***'.
            4) label_mention_map_portion, the portion of the correct label-mention pairs. Default is 1, which means all the label-mention pairs are correct.
        :return: a list of chat message template for each label subset
        """
        if annotator_cfg['chat']:
            pass
        else:
            all_labels = list(self.label2id.keys())
            if 'O' in all_labels:
                all_labels.remove('O')
            if 0 < anno_cfg['subset_size'] < 1:
                subset_size = math.floor(len(all_labels) * anno_cfg['subset_size'])
                if subset_size < 1:
                    subset_size = 1
            else:
                subset_size = anno_cfg['subset_size']

            label_subsets = fu.get_label_subsets(all_labels, subset_size, anno_cfg['repeat_num'])
            examples = None
            k_shot = anno_cfg['k_shot']
            if k_shot != 0:
                anno_cfg['support_set_dir'] = anno_cfg['support_set_dir'].format(dataset_name=kwargs['dataset_name'])
                k_shot_file = os.path.join(anno_cfg['support_set_dir'], f'span_{self.natural_flag}',f'train_support_set_{k_shot}_shot.jsonl')

                instance_template = anno_cfg['prompt_template']['instance_template'][kwargs['dialogue_style']]
                examples = ''
                index = 0
                for label_subset in label_subsets:
                    sents, empty_sents = [], []  # store the sentence for non-empty and empty outputs
                    with jsonlines.open(k_shot_file) as reader:
                        for line in reader:
                            sentence = '***' if kwargs['ignore_sent'] else ' '.join((line['tokens']))
                            output = '['
                            label_mention_pairs = fu.get_label_mention_pairs(line['spans_labels'],
                                                                             kwargs['label_mention_map_portion'],
                                                                             self.id2label)
                            for start, end, entity_mention, label_id in label_mention_pairs:
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
        return self._init_chat_msg_template(examples,
                                            annotator_cfg=annotator_cfg,
                                            anno_cfg=anno_cfg,
                                            use_api=use_api,
                                            dialogue_style=kwargs['dialogue_style'])

    def _generate_chat_msg(self, instances, annotator_cfg, anno_cfg, chat_msg_template, anno_style, dialogue_style):
        """
        For batch chat.
        Generate chat messages for each instance. Meanwhile, init the labels for each instance.

        :param instances: The instances to be annotated.
        :param annotator_cfg: The parameters of the annotation model.
        :param anno_cfg: the configuration of the annotation settings
        :param chat_msg_template: The chat message template for the annotating model.
        :param anno_style: The annotation style including 'single_type', 'multi_type', 'subset_type'
        :param dialogue_style, the style of the dialogue 'batch_qa' or 'multi_qa'
        :return:
        """
        def _generate_chat_info(query_template, chat_msg_template, tokens):
            sentence = ' '.join(tokens)
            chat_message = copy.deepcopy(chat_msg_template)
            query = query_template.format(sentence=sentence, output='')
            if dialogue_style == 'batch_qa':
                user_prompt = '\n### Query\n' + query
            else:
                user_prompt = query
            if self.annotator.use_api or annotator_cfg['chat']:
                chat_message.append({"role": "user", "content": user_prompt})
            else:
                user_prompt = chat_message[-1]["content"] + user_prompt  # concat the query to the system prompt
                chat_message[-1]["content"] = user_prompt  # replace the original user prompt
            return instance_id, chat_message, sentence, query

        prompt_template = anno_cfg['prompt_template']
        for instance_id, tokens, spans_labels in zip(instances['id'], instances['tokens'], instances['spans_labels']):
            if anno_style == 'single_type':
                # generate chat message using the single_type_prompt to extract entity directly by annotators
                # for single_type, chat_msg_template is a list of chat message template for each label
                # chat_msg_template[0] is for the first label, chat_msg_template[1] is for the second label, and so on.
                query_template = prompt_template['instance_template']
                for chat_msg_temp in chat_msg_template:
                    chat_message = copy.deepcopy(chat_msg_temp)
                    # yield instance_id, chat_message, sentence, query
                    yield _generate_chat_info(query_template, chat_message, tokens)
            elif anno_style == 'multi_type':
                # generate chat message using the multi_type_prompt to extract entity directly by annotators
                chat_message = copy.deepcopy(chat_msg_template)
                query_template = prompt_template['instance_template']
                # yield instance_id, chat_message, sentence, query
                yield _generate_chat_info(query_template, chat_message, tokens)
            elif anno_style == 'subset_type':
                # generate chat message using the subset types
                # In this case, the chat msg template is a list of chat message template for each label subset
                query_template = prompt_template['instance_template'][dialogue_style]
                for chat_msg_temp in chat_msg_template:
                    chat_message = copy.deepcopy(chat_msg_temp)
                    # yield instance_id, chat_message, sentence, query
                    yield _generate_chat_info(query_template, chat_message, tokens)
            elif anno_style == 'subset_cand':
                # generate chat message using the subset candidate prompt
                query_template = prompt_template['instance_template'][dialogue_style]
                chat_message = copy.deepcopy(chat_msg_template)
                # yield instance_id, chat_message, sentence, query
                yield _generate_chat_info(query_template, chat_message, tokens)

    @staticmethod
    def _process_output(output_text, sentence, **kwargs):
        """
        process the output text to get the label of the entity mention.
        :param output_text: the text output by the annotator model
        :param sentence: the query sentence
        :param kwargs: other parameters including,
            1) anno_style, indicate the annotation style including 'single_type', 'multi_type', 'subset_type'
            2) analysis, whether to extract the analysis process in the output text
        :return: if the single_type_prompt is True, return the predicted spans and their labels.
        """
        import ast
        if not output_text:
            output_text = ''
        # output_text = output_text.strip().replace('\_', '_')

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
        elif (kwargs['anno_style'] == 'multi_type' or kwargs['anno_style'] == 'subset_type' or kwargs['anno_style'] == 'subset_cand'):
            out_spans = []
            pattern = r'\[(.*?)\]'  # the pattern to extract a list string
            result = re.search(pattern, output_text, re.DOTALL)  # only find the first list string
            try:
                tmp_spans = ast.literal_eval(result.group(0).strip())  # tmp_spans shapes like [(type 0, mention0),...]
                # filter the invalid spans
                tmp_spans = filter(lambda e: isinstance(e, tuple), tmp_spans)
                tmp_spans = filter(lambda e: e and len(e) == 2, tmp_spans)
            except (TypeError, Exception):  # the output_text is not valid
                pattern = r'\((.*?)\)'  # the pattern to extract a tuple
                result = re.findall(pattern, output_text, re.DOTALL)  # find all the tuples
                try:  # try to extract tuple directly
                    tmp_spans = []
                    for e in result:
                        e = e.split(',')
                        if len(e) == 2:
                            tmp_spans.append((e[0].strip(), e[1].strip()))
                    # filter the invalid spans
                    tmp_spans = filter(lambda e: isinstance(e, tuple), tmp_spans)
                    tmp_spans = filter(lambda e: e and len(e) == 2, tmp_spans)
                except (TypeError, Exception):
                    tmp_spans = []

            for label, mention in tmp_spans:
                founded_spans = fu.find_span(sentence, str(mention))
                out_spans += [(str(start), str(end), span, label) for start, end, span in set(founded_spans)]
            return out_spans
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

    def get_response(self, client, chat_message, api_cfg, **kwargs):
        """
        Get the response of the annotator using api.
        :param client: the client of the LLM API
        :param chat_message: the chat message to be sent to the api.
        :param api_cfg: the configuration of the api settings
        :return:
        """
        # https://help.aliyun.com/zh/dashscope/developer-reference/compatibility-of-openai-with-dashscope

        output = '[]'
        try:
            logger.info('--------------- get response ---------------')
            completion = client.chat.completions.create(
                model=api_cfg['model'],
                messages=chat_message,
                stream=kwargs['stream'],
                top_p=kwargs['top_p'],
                temperature=kwargs['temperature'],
                max_tokens=kwargs['max_tokens'],
            )
            output = completion.choices[0].message.content
            logger.debug(output)
        except openai.APIConnectionError as e:
            logger.error(f"openai.APIConnectionError: {e}")
        except openai.RateLimitError as e:
            logger.error(f"openai.RateLimitError: {e}")
        except openai.APIStatusError as e:
            logger.error(f"openai.APIStatusError: {e}, status code: {e.status_code}")
        except Exception as e:
            logger.error(f"other exception: {e}")
        return output

    def get_batch_response(self, client, all_chat_message_info, api_cfg, anno_cfg, **kwargs):
        """
        Get the response of the annotator using batch inference.
        reffer to https://help.aliyun.com/zh/model-studio/batch-interfaces-compatible-with-openai
        :param client: the client of the LLM API
        :param all_chat_message_info: all the chat messages info to be sent to the api.
            an element of all_chat_message_info is a tuple like (instance_id, chat_message, sentence, query)
        :param api_cfg: the configuration of the api settings
        :param anno_cfg: the configuration of the annotation settings
        :param kwargs: other parameters, including,
            1) anno_style, the annotation style including 'single_type', 'multi_type', 'subset_type', 'subset_cand'
            2) dataset_name, the dataset name used in this annotation
            3) task_dir, the directory of this annotation task
            4) annotator_name: the name of the annotator
            5) stream, whether to use stream mode in batch inference api
            6) temperature, the temperature of the api model
            7) top_p, the top_p of the api model
            8) max_tokens, the max tokens of the api model
        :return:
        """
        def init_input_file(input_file, all_chat_message_info, anno_style):
            """
            init input file, a jsonline file
            # multi-line like

            # {"custom_id":"1","method":"POST","url":"/v1/chat/completions","body":{"model":"qwen-max","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"你好！有什么可以帮助你的吗？"}]}}

            # {"custom_id":"2","method":"POST","url":"/v1/chat/completions","body":{"model":"qwen-max","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"What is 2+2?"}]}}

            :param input_file: the path of the input file
            :param all_chat_message_info: all the chat messages info to be sent to the api.
                an element of all_chat_message_info is a tuple like (instance_id, chat_message, sentence, query)
            :param anno_style: the annotation style including 'single_type', 'multi_type', 'subset_type', 'subset_cand'
            :return:
            """
            with jsonlines.open(input_file, 'w') as writer:
                for instance_id, chat, sent, query in all_chat_message_info:
                    instance_id = str(instance_id) + '-' + uuid.uuid4().hex
                    line = {
                        "custom_id": instance_id,
                        "method": "POST",
                        "url": "/v1/chat/ds-test",  # /v1/chat/completions
                        "body": {
                            "model": 'batch-test-model',  # api_cfg['model']
                            "messages": chat,
                            "stream": kwargs.get('stream', False),
                            "top_p": kwargs.get('top_p', 0.5),
                            "temperature": kwargs.get('temperature', 0.1),
                            "max_tokens": kwargs.get('max_tokens', 100),
                        }
                    }
                    writer.write(line)
            logger.info(f"Init input file {input_file} success!\n")

        def upload_file(file_path):
            """
            upload the input file to the server
            :param file_path: the path of the input file
            :return:
            """
            logger.info(f"Uploading JSONL file containing requests...")
            file_object = client.files.create(file=Path(file_path), purpose="batch")
            logger.info(f"Uploading success! Got file ID: {file_object.id}\n")
            return file_object.id

        def create_batch_job(input_file_id):
            """
            create a batch job based on the input file ID
            :param input_file_id: the ID of the input file
            :return:
            """
            logger.info(f"Creating Batch Task based on file ID...")
            # if using Embedding model, set 'endpoint' with '/v1/embeddings'
            batch = client.batches.create(
                input_file_id=input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h")
            logger.info(f"Creating a Batch Task success! Got Batch Task ID: {batch.id}\n")
            return batch.id

        def check_job_status(batch_id):
            """
            check the job status of the batch task
            :param batch_id: the ID of the batch task
            :return:
            """
            logger.info(f"Checking the job status of the Batch Task (id {batch_id})...")
            batch = client.batches.retrieve(batch_id=batch_id)
            logger.info(f"Batch Task status: {batch.status}\n")
            return batch.status

        def get_output_id(batch_id):
            """
            get the output file ID of the successful execution request in the Batch Task
            :param batch_id: the ID of the batch task
            :return:
            """
            logger.info(f"Getting output file ID of the successful execution request in the Batch Task (id {batch_id})...")
            batch = client.batches.retrieve(batch_id=batch_id)
            logger.info(f"Output file ID: {batch.output_file_id}\n")
            return batch.output_file_id

        def get_error_id(batch_id):
            """
            get the output file ID of the unsuccessful executing requests in the Batch Task
            :param batch_id: the ID of the batch task
            :return:
            """
            logger.info(f"Getting output file ID for executing error requests in the Batch task (id {batch_id})...")
            batch = client.batches.retrieve(batch_id=batch_id)
            logger.info(f"Errot file ID: {batch.error_file_id}\n")
            return batch.error_file_id

        def download_results(output_file_id, output_file_path):
            """
            download the successful results of the batch task
            :param output_file_id: the ID of the output file
            :param output_file_path: the local path of the output file to save output
            :return:
            """
            logger.info(f"downloading successful results of the batch task...")
            content = client.files.content(output_file_id)
            # print part of the content for testing
            logger.info(f"Print the first 1000 characters of the successful request result: {content.text[:1000]}...\n")
            # save results to local
            content.write_to_file(output_file_path)
            logger.info(f"save successful results to {output_file_path}\n")

        def download_errors(error_file_id, error_file_path):
            """
            download the unsuccessful results of the batch task
            :param error_file_id: the ID of the error file
            :param error_file_path: the local path of the error file to save error
            :return:
            """
            logger.info(f"downloading unsuccessful results of the batch task...")
            content = client.files.content(error_file_id)
            # print part of the content for testing
            logger.info(f"Print the first 1000 characters of the unsuccessful request result: {content.text[:1000]}...\n")
            # save error content  to local
            content.write_to_file(error_file_path)
            logger.info(f"save error results to {error_file_path}\n")

        def extract_outputs(output_file):
            """
            extract the outputs from the output file
            :param output_file: the local path of the output file
            :return:
            """
            outputs = []
            with jsonlines.open(output_file) as reader:
                for line in reader:
                    outputs.append(line['response']['body']['choices'][0]['message']['content'])
            return outputs

        # 1. init file path
        batch_infer_dir = anno_cfg['setting_parent_dir'].format(dataset_name=kwargs['dataset_name'])
        batch_infer_dir = os.path.join(batch_infer_dir, 'batch_infer', kwargs['task_dir'])
        if not os.path.exists(batch_infer_dir):
            os.makedirs(batch_infer_dir)
        annotator_name = kwargs['annotator_name']
        input_file = os.path.join(batch_infer_dir, f"{annotator_name}_input.jsonl")  # file input
        output_file = os.path.join(batch_infer_dir, f"{annotator_name}_output.jsonl")  # file output
        error_file = os.path.join(batch_infer_dir, f"{annotator_name}_error.jsonl")  # error file

        # 2. init input file, a jsonline file
        if not os.path.exists(input_file):
            logger.info(f"Input file doesn't exist. Init input file {input_file}...")
            init_input_file(input_file, all_chat_message_info, anno_style=kwargs['anno_style'])
        if os.path.exists(output_file):  # if the output file exists, read it and return
            logger.info(f"Output file exist! Read cache from: {output_file}...")
            outputs = extract_outputs(output_file)
            return outputs
        try:
            # Step 1: 上传包含请求信息的JSONL文件,得到输入文件ID,如果您需要输入OSS文件,可将下行替换为：input_file_id = "实际的OSS文件URL或资源标识符"
            input_file_id = upload_file(input_file)
            # Step 2: 基于输入文件ID,创建Batch任务
            batch_id = create_batch_job(input_file_id)
            # Step 3: 检查Batch任务状态直到结束
            status = ""
            while status not in ["completed", "failed", "expired", "cancelled"]:
                status = check_job_status(batch_id)
                logger.info(f"waiting fot the batch task (id {batch_id}) complete...")
                time.sleep(10)  # 等待10秒后再次查询状态
            # 如果任务失败,则打印错误信息并退出
            if status == "failed":
                batch = client.batches.retrieve(batch_id)
                logger.error(f"Batch task fail. Error message:{batch.errors}\n")
                logger.error(f"refer to error code: https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
                return
            # Step 4: 下载结果：如果输出文件ID不为空,则打印请求成功结果的前1000个字符内容，并下载完整的请求成功结果到本地输出文件;
            # 如果错误文件ID不为空,则打印请求失败信息的前1000个字符内容,并下载完整的请求失败信息到本地错误文件.
            output_file_id = get_output_id(batch_id)
            if output_file_id:
                download_results(output_file_id, output_file)
            error_file_id = get_error_id(batch_id)
            if error_file_id:
                download_errors(error_file_id, error_file)
                logger.error(f"refer to error code: https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            logger.error(f"refer to error code: https://help.aliyun.com/zh/model-studio/developer-reference/error-code")

        outputs = extract_outputs(output_file)
        return outputs

    def annotate_by_one(self,dataset, anno_cfg, **kwargs):
        """
        Annotate the dataset by one specific annotator.
        :param dataset: the dataset to be annotated
        :param anno_cfg: the configuration of the annotation settings
        :param kwargs: other arguments, including
            2) dataset_name, the name of the dataset
            3) eval, whether to evaluate the annotation quality for each annotator
            4) cache, whether to cache the annotation results
            5) prompt_type, the type of the prompt, including single_type, mt_fs, and so on
            6) sampling_strategy, the strategy to sample the test subset
            7) dialogue_style, the style of the dialogue 'batch_qa' or 'multi_qa'
            8) ignore_sent, whether to ignore the sentence in the chat message. If True, the sentence will be shown as '***'.
            9) label_mention_map_portion, the portion of the corrected label-mention pair. Default is 1, which means all the label-mention pairs are correct.
            10) seed, the random seed
            11) start_row, the start row in the worksheet to continue to add metrics. If < 0, we don't write metrics to excel file.
        :return:
        """

        dataset_name = kwargs['dataset_name']
        # 0. Some settings
        # 0.1 get cache dir for result
        annotate_flag = True  # whether to annotate the dataset from scratch
        cache_dir = anno_cfg['cache_dir'].format(dataset_name=dataset_name)

        # 0.1.1 label format dir
        label_format_dir = f'span_{self.natural_flag}'

        # 0.1.2 label description dir
        label_des_dir = '{}_description'.format(anno_cfg['des_format'])

        if kwargs['prompt_type'] == 'sb_fs' or kwargs['prompt_type'] == 'sc_fs':
            subset_size = '-size_{}'.format(anno_cfg['subset_size'])
            repeat_num = '-rep_{}'.format(anno_cfg['repeat_num'])
            prompt_type_dir = os.path.join(kwargs['prompt_type'], '{}{}{}'.format(kwargs['prompt_type'], subset_size, repeat_num))
        elif kwargs['prompt_type'] == 'mt_fs':
            demo_times = 'rep_{}'.format(anno_cfg['demo_times'])
            prompt_type_dir = os.path.join(kwargs['prompt_type'], demo_times)
        else:
            prompt_type_dir = kwargs['prompt_type']

        # 0.1.3 test subset sampling strategy dir
        sub_samp_dir = '{}_sampling'.format(kwargs['sampling_strategy'])

        # 0.1.4 dialogue style dir
        dialogue_style_dir = '{}'.format(kwargs['dialogue_style'])

        # 0.1.5 annotator name
        if self.annotator.use_api:
            model_name = self.annotator.api_cfg['model']
        else:
            model_name = self.annotator.annotator_cfg['name']
        annotator_name =  model_name + '-' + anno_cfg['name']

        if anno_cfg['k_shot'] > 0:  # the 'ignore_sent' and 'label_mention_map_portion' is accessible when we use few-shot setting
            if kwargs['ignore_sent']:
                annotator_name += '-is'
            if kwargs['label_mention_map_portion'] < 1:
                annotator_name += '-lmp_{}'.format(kwargs['label_mention_map_portion'])
        if kwargs['seed']:
            annotator_name += '-{}'.format(kwargs['seed'])

        # task dir for this annotation
        task_dir = os.path.join(label_format_dir, prompt_type_dir, label_des_dir, sub_samp_dir, dialogue_style_dir, model_name)
        res_cache_dir = os.path.join(cache_dir, task_dir, annotator_name)
        logger.info(f'result cache dir: {res_cache_dir}')

        try:
            logger.info(f'Trying to load cache file from {res_cache_dir}')
            cache_result = load_from_disk(res_cache_dir)
            annotate_flag = False
        except FileNotFoundError:
            logger.info(f'No cache file found in {res_cache_dir}')
            if not os.path.exists(res_cache_dir):
                os.makedirs(res_cache_dir)

        # 0.2 annotation style setting
        anno_style = kwargs['prompt_type'].split('_')[0]
        if anno_style == 'single' or anno_style == 'st':
            anno_style = 'single_type'
        elif anno_style == 'mt':
            anno_style = 'multi_type'
        elif anno_style == 'sb':
            anno_style = 'subset_type'
        elif anno_style == 'sc':
            anno_style = 'subset_cand'

        # 0.3 other parameters to evaluate effieciency (only work for batch_qa with local annotator)
        prompt_lens_all = []
        execution_time_all = []
        query_count = 0
        efficiency_paras = None

        # annotation process
        if annotate_flag:
            # 1. Init the chat messages
            init_chat_template_methods = {
                'single_type': self._st_fs_msg,
                'multi_type': self._mt_fs_msg,
                'subset_type': self._subset_type_fs_msg,
                'subset_cand': self._subset_cand_fs_msg
            }

            if anno_cfg.get('k_shot', -1) >= 0:
                chat_msg_template = init_chat_template_methods[anno_style](
                    annotator_cfg=self.annotator.annotator_cfg,
                    anno_cfg=anno_cfg,
                    use_api=self.annotator.use_api,
                    **kwargs
                )

            # 2. batch process
            # 2.1 yield batch data
            if len(chat_msg_template) > 1:
                # when we use subset type prompt or single type prompt, chat_msg_template is a list of chat message template
                # we set the batch size to the number of label subsets
                # As a result, we can process one instance for each label subset in one batch
                batch_size = len(chat_msg_template)
            else:
                batch_size = self.annotator.annotator_cfg['anno_bs']

            all_chat_message_info = self._generate_chat_msg(
                instances=dataset,
                annotator_cfg=self.annotator.annotator_cfg,
                anno_cfg=anno_cfg,
                chat_msg_template=chat_msg_template,
                anno_style=anno_style,
                dialogue_style=kwargs['dialogue_style']
            )  # an element of all_chat_message_info is a tuple like (instance_id, chat_message, sentence, query)

            # 2.2 init the result
            pred_spans = [] # store the predicted spans and its label
            instance_results = []  # store the sentence, pred spans, gold results for each instance. Only used for multi/sigle type prompt
            outputs = []  # outputs (they are string) for each instance using api|annotator
            all_instance_ids = []  # store the instance ids for each instance.
            all_chat_messagges = []  # store the chat messages for each instance.
            all_sententes = []  # store the sentences for each instance.
            all_queries = []  # store the queries for each instance.

            # unpack the all_chat_message_info
            for instance_id, chat_message, sentence, query in all_chat_message_info:
                all_instance_ids.append(instance_id)
                all_chat_messagges.append(chat_message)
                all_sententes.append(sentence)
                all_queries.append(query)

            # 2.3 use batch inference with api.
            # Prioritize judging and executing this code segment
            if self.annotator.use_api and self.annotator.batch_infer:
                logger.info('using batch inference with api to annotate...')
                # use batch inference
                # 2.3.1 get the response of the annotator
                outputs = self.get_batch_response(
                    client=self.annotator.client,
                    all_chat_message_info=zip(all_instance_ids, all_chat_messagges, all_sententes, all_queries),
                    api_cfg=self.annotator.api_cfg,
                    anno_cfg=anno_cfg,
                    anno_style=anno_style,
                    dataset_name=dataset_name,
                    task_dir=task_dir,
                    annotator_name=annotator_name,
                    stream=self.annotator.annotator_cfg['stream'],
                    temperature=self.annotator.annotator_cfg['anno_temperature'],
                    top_p=self.annotator.annotator_cfg['anno_top_p'],
                    max_tokens=self.annotator.annotator_cfg['anno_max_tokens']
                )

            # 2.4 when not using batch inference, get the response of the annotator
            if not self.annotator.batch_infer and kwargs.get('dialogue_style') == 'batch_qa':
                if self.annotator.use_api:
                    # use LLM API and batch_qa
                    logger.info('using api with batch_qa to annotate...')
                    for chat in tqdm(all_chat_messagges, desc=f'annotating by {annotator_name}, dataset {dataset_name}'):
                        output = self.get_response(client=self.annotator.client,
                                                   chat_message=chat,
                                                   api_cfg=self.annotator.api_cfg,
                                                   stream=self.annotator.annotator_cfg['stream'],
                                                   temperature=self.annotator.annotator_cfg['anno_temperature'],
                                                   top_p=self.annotator.annotator_cfg['anno_top_p'],
                                                   max_tokens=self.annotator.annotator_cfg['anno_max_tokens'])
                        outputs.append(output)
                else:
                    # use local annotator and batch_qa
                    # we should use tokenizer.apply_chat_template to add generation template to the chats explicitly
                    for batch_chats in tqdm(
                            fu.batched(all_chat_messagges, batch_size),
                            desc=f'annotating by {annotator_name}, dataset {dataset_name}'
                    ):
                        templated_batch_chats = self.annotator.anno_tokenizer.apply_chat_template(
                            batch_chats, add_generation_prompt=True,tokenize=False
                        )
                        start_time = time.time()
                        batch_outputs = self.annotator.anno_model.generate(templated_batch_chats, self.annotator.sampling_params)  # annotate
                        end_time = time.time()
                        exce_time = end_time - start_time
                        execution_time_all.append(exce_time)
                        logger.info(f'execution time for a batch: {exce_time} s')
                        logger.info(f'execution time for a instance: {exce_time / len(batch_outputs)} s')

                        # for test
                        # test_answer = []
                        # for output in batch_outputs:
                        #     test_answer.append({'prompt': output.prompt, 'output': output.outputs[0].text})
                        prompt_lens = [len(e.prompt_token_ids) for e in batch_outputs]
                        prompt_lens_all += prompt_lens

                        logger.info(f'prompt average length: {sum(prompt_lens) // len(prompt_lens)} tokens')

                        output_lens = [len(e.outputs[0].token_ids) for e in batch_outputs]
                        logger.info(f'output average length: {sum(output_lens) // len(output_lens)} tokens')

                        batch_outputs = [e.outputs[0].text for e in batch_outputs]
                        query_count += len(batch_outputs)
                        outputs += batch_outputs
            elif not self.annotator.batch_infer and kwargs['dialogue_style'] == 'multi_qa':
                multi_qa_chat = []

                for batch in tqdm(
                    fu.batched(zip(all_instance_ids, all_chat_messagges, all_sententes, all_queries), batch_size),
                    desc=f'annotating by {annotator_name}, dataset {dataset_name}'
                ):
                    for idx, (instance_id, chat, sent, query) in enumerate(batch):
                        # chat[0] is the user role message
                        if idx == 0:
                            multi_qa_chat.append(chat[0])
                        else:
                            multi_qa_chat.append({"role": "user", "content": query})
                        if self.annotator.use_api:  # use LLM API and multi_qa
                            output_text = self.get_response(client=self.annotator.client,
                                                            chat_message=multi_qa_chat,
                                                            api_cfg=self.annotator.api_cfg,
                                                            stream=self.annotator.annotator_cfg['stream'],
                                                            temperature=self.annotator.annotator_cfg['anno_temperature'],
                                                            top_p=self.annotator.annotator_cfg['anno_top_p'],
                                                            max_tokens=self.annotator.annotator_cfg['anno_max_tokens'])
                        else:  # use local annotator and multi_qa
                            templated_multi_qa_chat = self.annotator.anno_tokenizer.apply_chat_template(multi_qa_chat, add_generation_prompt=True,tokenize=False)
                            tmp_outputs = self.annotator.anno_model.generate(templated_multi_qa_chat, self.annotator.sampling_params)  # len(tmp_outputs) == 1

                            prompt_lens = [len(e.prompt_token_ids) for e in tmp_outputs]
                            logger.info(f'prompt average length: {sum(prompt_lens) // len(prompt_lens)} tokens')

                            output_lens = [len(e.outputs[0].token_ids) for e in tmp_outputs]
                            logger.info(f'output average length: {sum(output_lens) // len(output_lens)} tokens')

                            output_texts = [e.outputs[0].text for e in tmp_outputs]  # len(output_texts) == 1
                            output_text = output_texts[0]
                        outputs.append(output_text)

                        # process the output for each turn in the multi_qa
                        out_spans = self._process_output(output_text, sent, anno_style=anno_style)
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

            # 2.5 process the output using mini_batch
            for batch in tqdm(
                    fu.batched(zip(all_instance_ids, all_chat_messagges, outputs, all_sententes), batch_size),
                    desc=f'process output by {annotator_name}, dataset {dataset_name}'
            ):
                # an element of batch is a tuple like (instance_id, chat_message, output, sentence)
                if anno_style == 'single_type':
                    tmp_pred_spans = []
                    for out_idx, (instance_id, chat_message, output, sent) in enumerate(batch):
                        out_spans = self._process_output(output, sent, anno_style=anno_style)
                        if len(out_spans) == 0:
                            continue
                        # instances in a batch is corresponding to different label
                        # e.g., the first instance in a batch is corresponding to the first label, i.e., id2label[0]
                        # the second instance in a batch is corresponding to the second label, i.e., id2label[1]
                        # However, in the self.label2id or id2label, the first label is 'O', we should skip it
                        # So , the output label id is the out_idx + 1
                        out_label_id = out_idx + 1
                        tmp_pred_spans += [(*out_span, str(out_label_id)) for out_span in set(out_spans)]
                    pred_spans += tmp_pred_spans
                    instance_results.append({'id': instance_id, 'sent': sent, 'pred_spans': tmp_pred_spans})
                else:
                    for out_idx, (instance_id, chat_message, output, sent) in enumerate(batch):
                        # if multi_qa, we have processed the output immediately after getting the response in 2.4
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

            # 3. cache and evaluate the annotation result
            # y_true is shape of [(start, end (excluded), gold_mention_span, gold_label_id), ...]
            # pred_spans is shape of [(start, end (excluded), pred_mention_span, pred_label_id), ...],
            # they are not one-to-one, so we convert them into 2-d list
            y_true = [span_label for spans_labels in dataset['spans_labels'] for span_label in spans_labels]  # flatten the spans_labels
            res_y_true = [y_true]
            res_pred_spans = [pred_spans]
            res_outputs = [outputs]

            if len(instance_results) > 0:  # for multi/single type prompt, we have instance_results
                ins_res = [instance_results]
                cache_result = {
                    "y_true": res_y_true, # the gold span and its label, shaped like [(start, end (excluded), gold_mention_span, gold_label_id), ...]
                    "pred_spans": res_pred_spans, # the predicted spans and its label, shaped like [(start, end (excluded), pred_mention_span, pred_label_id), ...]
                    "output_text": res_outputs, # the output text for each instance, shaped like [out_text_0, out_text1, ...]
                    "instance_results": ins_res  # the sentence, pred spans, gold results for each instance, shaped like [{'sent': sent_0, 'pred_spans': [(start, end, span, label), ...]}, ...]
                }
            else:  # for other settings, we do not have instance_results
                cache_result = {
                    "y_true": res_y_true,  # the gold span and its label, shaped like [(start, end (excluded), gold_mention_span, gold_label_id), ...]
                    "pred_spans": res_pred_spans,  # the predicted spans and its label, shaped like [(start, end (excluded), pred_mention_span, pred_label_id), ...]
                    "output_text": res_outputs  # # the output text for each instance, shaped like [out_text_0, out_text1, ...]
                }

            cache_result = Dataset.from_dict(cache_result)
            logger.info(f'Save cache result to {res_cache_dir}')

            efficiency_paras = None
            if kwargs['dialogue_style'] == 'batch_qa' and not self.annotator.use_api:
                prompt_avg_len = sum(prompt_lens_all) // query_count
                avg_time_per_batch = sum(execution_time_all) / len(execution_time_all)
                avg_time_per_ins = sum(execution_time_all) / query_count
                efficiency_paras = {'prompt_avg_len': prompt_avg_len,
                                    'avg_time_per_batch': avg_time_per_batch,
                                    'avg_time_per_ins': avg_time_per_ins
                                    }
                logger.info(f'prompt average length: {prompt_avg_len} tokens')
                logger.info(f'average execution time for a batch: {avg_time_per_batch} s')
                logger.info(f'average execution time for a instance: {avg_time_per_ins} s')

            if kwargs['cache']:
                cache_result.save_to_disk(res_cache_dir)

        # 4. evaluation
        if kwargs['eval']:
            # important! we must remove the '0'('O' label) span from the pred_spans before evaluation
            pred_spans = [span for span in cache_result['pred_spans'][0] if int(span[-1]) != self.label2id['O']]
            self.evaluate(cache_result['y_true'][0], pred_spans,
                          anno_cfg=anno_cfg,
                          dataset=dataset,
                          dataset_name=dataset_name,
                          annotator_name=annotator_name,
                          task_dir=task_dir,
                          anno_style=anno_style,
                          prompt_type=kwargs['prompt_type'],
                          efficiency_paras=efficiency_paras)
            start_row = kwargs['start_row']
            if kwargs['start_row'] > 0:
                start_row = self.write_metrics_to_excel(anno_cfg=anno_cfg,
                                                        start_row=start_row,
                                                        task_dir=task_dir,
                                                        dataset_name=dataset_name,
                                                        annotator_name=annotator_name,
                                                        label_mention_map_portion=kwargs['label_mention_map_portion'])

        return  start_row

    def evaluate(self, y_true, y_pred, anno_cfg, **kwargs):
        """
        Evaluate and save evaluation results by an annotator.
        :param y_true: y_true stores the gold span and their labels in a tuple,
        shaped like [(start, end (excluded), gold_mention_span, gold_label_id), ...].
        :param y_pred: y_pred stores the predicted spans and their labels.
        :param anno_cfg: the configuration of the annotation settings
        :param kwargs: other arguments, including
            1) dataset, the dataset used to be evaluated
            2) dataset_name,  the name of the dataset.
            3) annotator_name,  the name of the annotator LLM.
            4) task_dir, the directory of the task
            5) prompt_type, the type of the prompt, including single_type, mt_fs, raw, and so on
            6) efficiency_paras, the efficiency parameters, including prompt_avg_len, avg_time_per_batch, avg_time_per_ins
        :return:
        """
        eval_dir = anno_cfg['eval_dir'].format(dataset_name=kwargs['dataset_name'])
        res_cache_dir = os.path.join(eval_dir, kwargs['task_dir'])
        if not os.path.exists(res_cache_dir):
            os.makedirs(res_cache_dir)
        res_file = os.path.join(res_cache_dir, '{}_res.txt'.format(kwargs['annotator_name']))
        res_by_class_file = os.path.join(res_cache_dir, '{}_res_by_class.csv'.format(kwargs['annotator_name']))
        logger.info(f'saved the evaluation results to {res_file}')
        logger.info(f'saved the evaluation results by class to {res_by_class_file}')

        # compute span-level metrics
        eval_results = fu.compute_span_f1(copy.deepcopy(y_true),  copy.deepcopy(y_pred))
        fu.compute_span_f1_by_labels(copy.deepcopy(y_true), copy.deepcopy(y_pred), id2label=self.id2label, res_file=res_by_class_file)

        if anno_cfg['k_shot'] > 0:  # LSPI and LC are only available for few-shot setting
            lspi, lc, lm_beta = self.get_label_measure(kwargs['dataset'], anno_cfg, kwargs['dataset_name'], prompt_type=kwargs['prompt_type'])
            eval_results['lspi'] = lspi
            eval_results['lc'] = lc
            eval_results['lm'] = lm_beta
        with open(res_file, 'w') as f:
            for metric, res in eval_results.items():
                f.write(f'{metric}: {res}\n')

        if kwargs['efficiency_paras'] is not None:
            eff_file = os.path.join(res_cache_dir, '{}_efficiency.txt'.format(kwargs['annotator_name']))
            with open(eff_file, 'w') as f:
                for metric, res in kwargs['efficiency_paras'].items():
                    f.write(f'{metric}: {res}\n')
        return

    def get_label_measure(self, test_dataset, anno_cfg, dataset_name, prompt_type='sc_fs', beta=1):
        """
        get the label space per instance and label coverage
        :param test_dataset: dataset used to be evaluated
        :param anno_cfg: the configuration of the annotation settings
        :param dataset_name: the name of the dataset
        :param prompt_type: the type of the prompt, including sc_fs, mt_fs
        :param beta: the weight for lspi, lc. lm_beta = (1+b^2)*LSPI*LC/(b^2*LSPI+LC)
        :return:
        """
        all_labels = list(self.label2id.keys())
        if 'O' in all_labels:
            all_labels.remove('O')
        k_shot = anno_cfg['k_shot']
        outputs = []  # store all output labels for every instance
        if prompt_type == 'sc_fs':
            logger.info(f'subset_size: {anno_cfg["subset_size"]}, repeat_num: {anno_cfg["repeat_num"]}')
            label_subsets = fu.get_label_subsets(all_labels, anno_cfg['subset_size'], anno_cfg['repeat_num'])

        if k_shot != 0:
            # get the support set file
            anno_cfg['support_set_dir'] = anno_cfg['support_set_dir'].format(dataset_name=dataset_name)
            k_shot_file = os.path.join(anno_cfg['support_set_dir'], f'span_{self.natural_flag}', f'train_support_set_{k_shot}_shot.jsonl')

            # get the label sets from demonstration
            if prompt_type == 'sc_fs':  # subset candidate prompt with few-shot setting
                for label_subset in label_subsets:
                    empty_num = 0  # count the number of empty outputs
                    with jsonlines.open(k_shot_file) as reader:
                        for line in reader:
                            output = []
                            for start, end, entity_mention, label_id in line['spans_labels']:
                                label = self.id2label[int(label_id)]
                                if label in label_subset:
                                    output.append(label)
                            if len(output) == 0:
                                empty_num += 1
                                continue
                            else:
                                outputs.append(output)

                    if empty_num > 0:  # random select empty outputs
                        select_num = len(label_subsets) if empty_num > len(label_subsets) else empty_num
                        for _ in range(select_num):
                            outputs.append([])
            elif prompt_type == 'mt_fs' or prompt_type == 'st_fs':  # multi type| single type prompt with few-shot setting
                with jsonlines.open(k_shot_file) as reader:
                    for line in reader:
                        output = []
                        for start, end, entity_mention, label_id in line['spans_labels']:
                            label = self.id2label[int(label_id)]
                            output.append(label)
                        outputs.append(output)

                assert 'demo_times' in anno_cfg.keys(), "The demo_times is required for 'multi_type_prompt'. Defualt 1"
                original_outputs = copy.deepcopy(outputs)
                for _ in range(anno_cfg['demo_times']):
                    for _ in original_outputs:
                        outputs.append(output)
            # get the gold label sets from test dataset
            gold_outputs = []
            for spans_labels in test_dataset['spans_labels']:
                output = []
                for start, end, entity_mention, label_id in spans_labels:
                    label = self.id2label[int(label_id)]
                    output.append(label)
                gold_outputs.append(output)
            lspi = fu.compute_lspi(outputs)  # label space per instance
            lc = fu.compute_label_coverage(outputs, gold_outputs)  # label coverage
            lm_beta = (1 + beta ** 2) * lspi * lc / (beta ** 2 * lspi + lc)
            return lspi, lc, lm_beta

    def write_metrics_to_excel(self, anno_cfg, **kwargs):
        """
        write metrics to excel files
        :param kwargs: other arguments, including
            1) start_row: the row index we start. If the worksheet is new, start_row is 2. elif start_row > 2, it's the last row we add data.
            2) dataset_name,  the name of the dataset.
            3) annotator_name,  the name of the annotator LLM.
            4) task_dir, the directory of the task
            5) label_mention_map_portion, specify the portion of the label mention mapping
        :return:
        """
        start_row = kwargs['start_row']
        eval_dir = anno_cfg['eval_dir'].format(dataset_name=kwargs['dataset_name'])
        res_cache_dir = os.path.join(eval_dir, kwargs['task_dir'])
        if not os.path.exists(res_cache_dir):
            os.makedirs(res_cache_dir)
        res_file = os.path.join(res_cache_dir, '{}_res.txt'.format(kwargs['annotator_name']))
        logger.info(f'write metrics ({res_file}) to excel')
        with open(res_file, 'r') as f:
            eval_results = f.readlines()

        metric_num = len(eval_results)  # the number of metrics
        if start_row == 2:  # it's a new worksheet
            self.worksheet.write(0, 0, 'label_mention_map_portion')
            self.worksheet.write(0, 1, 'rep_num')
            self.worksheet.write(0, 2, '5-shot')
            self.worksheet.write(0, metric_num + 4, '1-shot')
            if anno_cfg['k_shot'] == 5:  # 5-shot
                head_row, head_col = 1, 3  # headers start from (1, 3)
                data_row, data_col = 2, 3  # datas start from (2, 3)
            else:  # 1-shot
                head_row, head_col = 1, metric_num + 4  # headers start from (1, metric_num + 4)
                data_row, data_col = 2, metric_num + 4  # datas start from (2, metric_num + 4)
        elif start_row > 2:  # we continue to write data to the older worksheet
            if anno_cfg['k_shot'] == 5:  # 5-shot
                head_row, head_col = 1, 3  # headers start from (1, 3)
                data_row, data_col = start_row, 3  # datas start from (start_row, 3)
            else:  # 1-shot
                head_row, head_col = 1, metric_num + 4  # headers start from (1, metric_num + 4)
                data_row, data_col = start_row, metric_num + 4  # datas start from (start_row, metric_num + 4)

        if 'repeat_num' in anno_cfg.keys():
            rep_num = anno_cfg['repeat_num']
        elif 'demo_times':
            rep_num = anno_cfg['demo_times']
        self.worksheet.write(data_row, 0, kwargs['label_mention_map_portion'])
        self.worksheet.write(data_row, 1, rep_num)
        self.worksheet.write(data_row, 2, kwargs['annotator_name'])
        for line in eval_results:
            line = line.strip()
            line = line.split(' ')
            metric, res = line[0], float(line[1])
            if start_row == 2:  # header
                self.worksheet.write(head_row, head_col, metric)
                head_col += 1
            self.worksheet.write(data_row, data_col, res)
            data_col += 1
        data_row += 1
        return  data_row