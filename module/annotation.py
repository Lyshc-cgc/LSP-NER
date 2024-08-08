import copy
import json
import math
import random
import re
import os
import xlsxwriter

import jsonlines
from vllm import LLM, SamplingParams
from tenacity import retry, retry_if_exception_type, wait_random
from openai import OpenAI, AsyncOpenAI
from datasets import load_from_disk, Dataset, load_dataset
from tqdm import tqdm
from module import func_util as fu
from module.label import Label

logger = fu.get_logger('Annotation')

class Annotation(Label):
    """
    The Annotation class is used to annotate the data.
    """
    def __init__(self, annotator_cfg, api_cfg, labels_cfg, natural_form=False, just_test=False):
        """
        Initialize the annotation model.
        :param annotator_cfg: the configuration of the local annotator model.
        :param api_cfg: the configuration of the LLM API.
        :param labels_cfg: the configuration of the label_cfgs.
        :param natural_form: whether the labels are in natural language form.
        :param just_test: If True, we just want to test something in the Annotation, so that we don't need to load the model.
        """
        # 0. cfg initialization
        super().__init__(labels_cfg, natural_form)
        self.annotator_cfg = annotator_cfg
        self.api_cfg = api_cfg if api_cfg else None
        self.use_api = True if api_cfg else False
        self.natural_flag = 'natural' if natural_form else 'bio'  # use natural labels or bio labels
        self.workbook = xlsxwriter.Workbook('metrics.xlsx')  # write metric to excel
        self.worksheet = self.workbook.add_worksheet()  # default 'Sheet 1'

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

            if 'task_label' in kwargs.keys():
                task_label = kwargs['task_label']
                task_prompt = task_prompt.format(task_label=task_label)
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

        if annotator_cfg['chat']:  # for qwen, it has 'system' role message
            chat_message = [{"role": "system", "content": sys_prompt}]
            if kwargs['use_api'] and self.api_cfg['model'] == 'qwen-long':
                # see https://help.aliyun.com/document_detail/2788814.html?spm=a2c4g.2788811.0.0.1440240aUbuyYI#b7f81199e2laz
                # when use qwen-long, we should add an extra system message for role-play to the chat_message
                chat_message = [{'role': 'system', 'content': anno_cfg['system_role']}] + chat_message
        else:  # for mistral, it doesn't have 'system' role message
            chat_message = [{"role": "user", "content": sys_prompt}]
        return chat_message

    def _st_fs_msg(self, annotator_cfg, anno_cfg, **kwargs) -> list[None | dict[str, str]]:
        """
        Init the chat messages for the annotation models to extract entity directly by annotators according the single_type_prompt.
        Init examples from the support set sampled from the dataset.

        :param annotator_cfg: The parameters of the annotation model.
        :param anno_cfg: the configuration of the annotation settings
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
                                instance = anno_cfg['instance_template'].format(label=label, sentence=sentence, output=output)
                                examples += f'{index + 1})\n{instance}\n'
                                index += 1
                            else:  # no gold mentions
                                empty_sents.append(sentence)

                    if len(empty_sents) > 0:  # randomly select empty outputs
                        select_num = 3 if len(empty_sents) > 3 else len(empty_sents)
                        for sentence in random.sample(empty_sents, select_num):
                            output = '"' + sentence + '"'
                            instance = anno_cfg['instance_template'].format(sentence=sentence, output=output)
                            examples += f'{index + 1})\n{instance}\n'
                            index += 1
                    chat_msg_template_list.append(
                        self._init_chat_msg_template(examples,
                                                     annotator_cfg=annotator_cfg,
                                                     anno_cfg=anno_cfg,
                                                     use_api=self.use_api,
                                                     task_label=target_label)
                    )
            else:
                for target_label in all_labels:
                    chat_msg_template_list.append(
                        self._init_chat_msg_template(examples,
                                                     annotator_cfg=annotator_cfg,
                                                     anno_cfg=anno_cfg,
                                                     use_api=self.use_api,
                                                     task_label=target_label)
                    )
        return chat_msg_template_list

    def _mt_fs_msg(self, annotator_cfg, anno_cfg, **kwargs) -> list[None | dict[str, str]]:
        """
        Init the chat messages for the annotation models to extract entity directly by annotators according the multi_type_prompt.
        Init examples from the support set sampled from the dataset.

        :param annotator_cfg: The parameters of the annotation model.
        :param anno_cfg: the configuration of the annotation settings
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
                        instance = anno_cfg['instance_template'].format(sentence=sentence, output=output)
                        example_list.append(instance)

                index = 0
                assert 'demo_times' in anno_cfg.keys(), "The demo_times is required for 'multi_type_prompt'. Defualt 1"
                examples = ''  # store the examples input to context
                for _ in range(anno_cfg['demo_times']):
                    for instance in example_list:
                        examples += f'{index + 1})\n{instance}\n'
                        index += 1
        return self._init_chat_msg_template(examples, annotator_cfg=annotator_cfg, anno_cfg=anno_cfg, use_api=self.use_api)

    def _subset_type_fs_msg(self, annotator_cfg, anno_cfg, **kwargs):
        """
        Init the chat messages for the annotation models using subset types with few-shot settings.
        Init examples from the support set sampled from the dataset.

        :param annotator_cfg: The parameters of the annotation model.
        :param anno_cfg: the configuration of the annotation settings
        :param kwargs: other parameters, including,
            1) dataset_name: the name of the dataset.
            2) dialogue_style: the style of the dialogue. 'batch_qa' or 'multi_qa'
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
                subset_size = math.floor(len(all_labels) * anno_cfg['subset_size'])
            else:
                subset_size = anno_cfg['subset_size']
            logger.info(f"cfg subset_size:{anno_cfg['subset_size']}, subset_size: {subset_size}")
            label_subsets = fu.get_label_subsets(all_labels, subset_size, anno_cfg['repeat_num'])
            examples = None
            k_shot = anno_cfg['k_shot']
            chat_msg_template_list = []  # store the chat message template for each label subset
            if k_shot != 0:
                anno_cfg['support_set_dir'] = anno_cfg['support_set_dir'].format(dataset_name=kwargs['dataset_name'])
                k_shot_file = os.path.join(anno_cfg['support_set_dir'], f'span_{self.natural_flag}',f'train_support_set_{k_shot}_shot.jsonl')

                instance_template = anno_cfg['instance_template'][kwargs['dialogue_style']]
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
                                                     use_api=self.use_api,
                                                     labels=label_subset,
                                                     dialogue_style=kwargs['dialogue_style'])
                    )
            else:
                for label_subset in label_subsets:
                    chat_msg_template_list.append(
                        self._init_chat_msg_template(examples,
                                                     annotator_cfg=annotator_cfg,
                                                     anno_cfg=anno_cfg,
                                                     use_api=self.use_api,
                                                     labels=label_subset,
                                                     dialogue_style=kwargs['dialogue_style'])
                    )

        return chat_msg_template_list

    def _subset_cand_fs_msg(self, annotator_cfg, anno_cfg, **kwargs):
        """
        Init the chat messages for the annotation models using subset candidate prompt with few-shot settings.
        Init examples from the support set sampled from the dataset.

        :param annotator_cfg: The parameters of the annotation model.
        :param anno_cfg: the configuration of the annotation settings
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
            else:
                subset_size = anno_cfg['subset_size']
            logger.info(f"cfg subset_size:{anno_cfg['subset_size']}, subset_size: {subset_size}")
            label_subsets = fu.get_label_subsets(all_labels, subset_size, anno_cfg['repeat_num'])
            examples = None
            k_shot = anno_cfg['k_shot']
            if k_shot != 0:
                anno_cfg['support_set_dir'] = anno_cfg['support_set_dir'].format(dataset_name=kwargs['dataset_name'])
                k_shot_file = os.path.join(anno_cfg['support_set_dir'], f'span_{self.natural_flag}',f'train_support_set_{k_shot}_shot.jsonl')

                instance_template = anno_cfg['instance_template'][kwargs['dialogue_style']]
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
                                            use_api=self.use_api,
                                            dialogue_style=kwargs['dialogue_style'])

    def _generate_chat_msg(self, instances, annotator_cfg, anno_cfg, chat_msg_template, anno_style, dialogue_style):
        """
        For batch chat.
        Generate chat messages for each instance. Meanwhile, init the labels for each instance.

        :param instances: The instances to be annotated.
        :param annotator_cfg: The parameters of the annotation model.
        :param anno_cfg: the configuration of the annotation settings
        :param chat_msg_template: The chat message template for the annotating model.
        :param anno_style: The annotation style including 'single_type', 'multi_type', 'cand_mention_type', 'subset_type' or 'raw'
        :param dialogue_style, the style of the dialogue 'batch_qa' or 'multi_qa'
        :return:
        """
        for instance_id, tokens, spans_labels in zip(instances['id'], instances['tokens'], instances['spans_labels']):
            if anno_style == 'single_type':
                # generate chat message using the single_type_prompt to extract entity directly by annotators
                for chat_msg_temp in chat_msg_template:
                    sentence = ' '.join(tokens)
                    chat_message = copy.deepcopy(chat_msg_temp)
                    query = anno_cfg['instance_template'].format(sentence=sentence, output='')
                    if dialogue_style == 'batch_qa':
                        user_prompt = '\n### Query\n' + query
                    else:
                        user_prompt = query

                    if annotator_cfg['chat']:
                        chat_message.append({"role": "user", "content": user_prompt})
                    else:
                        user_prompt = chat_message[-1]["content"] + user_prompt  # concat the query to the system prompt
                        chat_message[-1]["content"] = user_prompt  # replace the original user prompt
                    yield instance_id, chat_message, sentence
            elif anno_style == 'multi_type':
                # generate chat message using the multi_type_prompt to extract entity directly by annotators
                sentence = ' '.join(tokens)
                chat_message = copy.deepcopy(chat_msg_template)
                query = anno_cfg['instance_template'].format(sentence=sentence, output='')
                if dialogue_style == 'batch_qa':
                    user_prompt = '\n### Query\n' + query
                else:
                    user_prompt = query

                if annotator_cfg['chat']:
                    chat_message.append({"role": "user", "content": user_prompt})
                else:
                    user_prompt = chat_message[-1]["content"] + user_prompt  # concat the query to the system prompt
                    chat_message[-1]["content"] = user_prompt  # replace the original user prompt
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
                    if annotator_cfg['chat']:
                        chat_message.append({"role": "user", "content": user_prompt})
                    else:
                        user_prompt = chat_message[-1]["content"] + user_prompt  # concat the query to the system prompt
                        chat_message[-1]["content"] = user_prompt  # replace the original user prompt
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
                if annotator_cfg['chat']:
                    chat_message.append({"role": "user", "content": user_prompt})
                else:
                    user_prompt = chat_message[-1]["content"] + user_prompt  # concat the query to the system prompt
                    chat_message[-1]["content"] = user_prompt  # replace the original user prompt
                yield instance_id, chat_message, sentence

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
            logger.info('--------------- get response ---------------')
            completion = client.chat.completions.create(
                model=self.api_cfg['model'],
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

    def annotate_by_one(self,dataset, anno_cfg, **kwargs):
        """
        Annotate the dataset by one specific annotator.
        :param dataset: the dataset to be annotated
        :param anno_cfg: the configuration of the annotation settings
        :param kwargs: other arguments, including
            2) dataset_name, the name of the dataset
            3) eval, whether to evaluate the annotation quality for each annotator
            4) cache, whether to cache the annotation results
            5) prompt_type, the type of the prompt, including single_type, mt_fs, raw, and so on
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
        if self.use_api:
            model_name = self.api_cfg['model']
        else:
            model_name = self.annotator_cfg['name']
        annotator_name =  model_name + '-' + anno_cfg['name']

        if kwargs['ignore_sent']:
            annotator_name += '-is'
        if kwargs['label_mention_map_portion'] < 1:
            annotator_name += '-lmp_{}'.format(kwargs['label_mention_map_portion'])
        if kwargs['seed']:
            annotator_name += '-{}'.format(kwargs['seed'])
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

        # annotation process
        if annotate_flag:
            # 1. Init the chat messages
            if anno_style == 'single_type':
                if 'k_shot' in anno_cfg.keys() and anno_cfg['k_shot'] >= 0:  # single_type_prompt with few-shot setting
                    chat_msg_template = self._st_fs_msg(annotator_cfg=self.annotator_cfg, anno_cfg=anno_cfg, **kwargs)
            elif anno_style == 'multi_type':
                if 'k_shot' in anno_cfg.keys() and anno_cfg['k_shot'] >= 0:
                    chat_msg_template = self._mt_fs_msg(annotator_cfg=self.annotator_cfg, anno_cfg=anno_cfg, **kwargs)
            elif anno_style == 'subset_type':
                if 'k_shot' in anno_cfg.keys() and anno_cfg['k_shot'] >= 0:
                    chat_msg_template = self._subset_type_fs_msg(annotator_cfg=self.annotator_cfg, anno_cfg=anno_cfg, **kwargs)
            elif anno_style == 'subset_cand':
                if 'k_shot' in anno_cfg.keys() and anno_cfg['k_shot'] >= 0:
                    chat_msg_template = self._subset_cand_fs_msg(annotator_cfg=self.annotator_cfg, anno_cfg=anno_cfg, **kwargs)

            # 2. batch process
            # 2.1 yield batch data
            if len(chat_msg_template) > 1:
                # when we use subset type prompt or single type prompt, chat_msg_template is a list of chat message template
                # we set the batch size to the number of label subsets
                # As a result, we can process one instance for each label subset in one batch
                batch_size = len(chat_msg_template)
            else:
                batch_size = self.annotator_cfg['anno_bs']
            pbar = tqdm(fu.batched(self._generate_chat_msg(instances=dataset,
                                                           annotator_cfg=self.annotator_cfg,
                                                           anno_cfg=anno_cfg,
                                                           chat_msg_template=chat_msg_template,
                                                           anno_style=anno_style,
                                                           dialogue_style=kwargs['dialogue_style']),
                                   batch_size),
                        desc=f'annotating by {annotator_name}, dataset {dataset_name}')

            res_labels, res_label_ids = [], []  # store the output labels and label ids

            y_true = []
            pred_spans = [] # store the predicted spans and its label
            output_texts = []  # store the output text for each instance
            instance_results = []  # store the sentence, pred spans, gold results for each instance. Only used for multi/sigle type prompt
            for batch_id, batch in enumerate(pbar):
                batch_spans = []  # store the span for each batch
                batch_instance_ids = []  # store the instance ids for each batch
                batch_chats = []  # store the chats for each batch
                batch_sents = []  # store the sentences for each batch
                batch_res_label_ids = []  # store the output label ids for each batch to evaluate

                if anno_style == 'raw':
                    # batch is a tuple like ((instance_id_0, span_0, chat_0),(instance_id_1,span_1, chat_1)...)
                    # we  get all gold span labels after annotation
                    for instance_id, span, chat, sent in batch:
                        batch_instance_ids.append(instance_id)
                        batch_spans.append(span)
                        batch_chats.append(chat)
                        batch_sents.append(sent)
                else:  # single_type, multi_type, cand_mention_type, subset_type, subset_cand
                    # batch is a tuple like ((instance_id_0, chat_0, sent_0), (instance_id_1, chat_1, sent_1),...)
                    for instance_id, chat, sent in batch:
                        batch_instance_ids.append(instance_id)
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
                if anno_style == 'single_type':
                    tmp_pred_spans = []
                    for out_idx, (instance_id, output, sent) in enumerate(zip(batch_instance_ids, outputs, batch_sents)):
                        output_texts.append(output)
                        if 'analysis' in anno_cfg.keys() and anno_cfg['analysis']:
                            out_spans = self._process_output(output, sent, anno_style=anno_style, analysis=anno_cfg['analysis'])
                        else:
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
                    for out_idx, (instance_id, output, sent) in enumerate(zip(batch_instance_ids, outputs, batch_sents)):
                        output_texts.append(output)
                        if anno_style == 'multi_type' or anno_style == 'cand_mention_type' or anno_style == 'subset_type' or anno_style == 'subset_cand':
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

            # 3. cache and evaluate the annotation result
            # y_true is shape of [(start, end (excluded), gold_mention_span, gold_label_id), ...]
            # pred_spans is shape of [(start, end (excluded), pred_mention_span, pred_label_id), ...],
            # they are not one-to-one, so we convert them into 2-d list
            y_true = [span_label for spans_labels in dataset['spans_labels'] for span_label in spans_labels]  # flatten the spans_labels
            res_y_true = [y_true]
            res_pred_spans = [pred_spans]
            res_output_texts = [output_texts]

            if len(instance_results) > 0:  # for multi/single type prompt, we have instance_results
                ins_res = [instance_results]
                cache_result = {
                    "y_true": res_y_true, # the gold span and its label, shaped like [(start, end (excluded), gold_mention_span, gold_label_id), ...]
                    "pred_spans": res_pred_spans, # the predicted spans and its label, shaped like [(start, end (excluded), pred_mention_span, pred_label_id), ...]
                    "output_text": res_output_texts, # the output text for each instance, shaped like [out_text_0, out_text1, ...]
                    "instance_results": ins_res  # the sentence, pred spans, gold results for each instance, shaped like [{'sent': sent_0, 'pred_spans': [(start, end, span, label), ...]}, ...]
                }
            else:  # for other settings, we do not have instance_results
                cache_result = {
                    "y_true": res_y_true,  # the gold span and its label, shaped like [(start, end (excluded), gold_mention_span, gold_label_id), ...]
                    "pred_spans": res_pred_spans,  # the predicted spans and its label, shaped like [(start, end (excluded), pred_mention_span, pred_label_id), ...]
                    "output_text": res_output_texts  # # the output text for each instance, shaped like [out_text_0, out_text1, ...]
                }

            cache_result = Dataset.from_dict(cache_result)
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
                          prompt_type=kwargs['prompt_type'])
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
        Evaluate the annotation results by an annotator.
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
        lspi, lc = self.get_lspi_lc(kwargs['dataset'], anno_cfg, kwargs['dataset_name'], prompt_type=kwargs['prompt_type'])
        eval_results['lspi'] = lspi
        eval_results['lc'] = lc
        with open(res_file, 'w') as f:
            for metric, res in eval_results.items():
                f.write(f'{metric}: {res}\n')

        return

    def get_lspi_lc(self, test_dataset, anno_cfg, dataset_name, prompt_type='sc_fs'):
        """
        get the label space per instance and label coverage
        :param test_dataset: dataset used to be evaluated
        :param anno_cfg: the configuration of the annotation settings
        :param dataset_name: the name of the dataset
        :param prompt_type: the type of the prompt, including sc_fs, mt_fs
        :return:
        """
        all_labels = list(self.label2id.keys())
        if 'O' in all_labels:
            all_labels.remove('O')
        k_shot = anno_cfg['k_shot']
        outputs = []  # store all output labels for every instance
        if prompt_type == 'sc_fs':
            logger.info(f'subset_size: {anno_cfg["subset_size"]}, repeat_num: {anno_cfg["repeat_num"]}')
            if 0 < anno_cfg['subset_size'] < 1:
                subset_size = math.floor(len(all_labels) * anno_cfg['subset_size'])
            else:
                subset_size = anno_cfg['subset_size']
            label_subsets = fu.get_label_subsets(all_labels, subset_size, anno_cfg['repeat_num'])
        elif prompt_type == 'mt_fs':
            logger.info(f'demo_times: {anno_cfg["demo_times"]}')

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
            elif prompt_type == 'mt_fs':  # multi type prompt with few-shot setting
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
            return lspi, lc

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
            self.worksheet.write(0, 1, '5-shot')
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