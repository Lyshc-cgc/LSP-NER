import copy
import json
import random
import re
import os
import asyncio

import jsonlines
import wandb
import torch
import numpy as np
from tenacity import retry, wait_random, stop_after_attempt
from zhipuai import ZhipuAI
from openai import OpenAI, AsyncOpenAI
from datasets import load_from_disk, Dataset
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, matthews_corrcoef, cohen_kappa_score
from module import func_util as fu
from module.label import Label

class Annotation(Label):
    """
    The Annotation class is used to annotate the data.
    """
    def __init__(self, anno_cfg, api_cfg, labels_cfg):
        super().__init__(labels_cfg)
        self.anno_config = anno_cfg
        self.api_cfg = api_cfg if api_cfg else None
        self.use_api = True if api_cfg else False
        self.annotators_cfg = self.anno_config['annotators']
        self.annotator_ids = dict()
        for idx, anno_cfg in enumerate(self.annotators_cfg):
            self.annotator_ids[anno_cfg['name']] = idx
    
    def _init_chat_msg_template(self, examples, dataset_name, use_api=False) -> list[None | dict[str, str]]:
        """
        Get examples and init the chat messages for the annotation models.
        :param examples: the examples to be shown to annotators.
        :param dataset_name: the name of the dataset
        :param use_api: whether to use LLM API as annotator
        :return:
        """
        if examples:
            kwargs = {'examples': examples}
        else:
            kwargs = dict()
        if 'system_role' in self.anno_config.keys() and self.anno_config['system_role']:
            kwargs.update({'system_role': self.anno_config['system_role']})
        if 'task_prompt' in self.anno_config.keys() and self.anno_config['task_prompt']:
            kwargs.update({'task_prompt': self.anno_config['task_prompt']})
        if 'types_prompt' in self.anno_config.keys() and self.anno_config['types_prompt']:
            # different dataset uses different types_prompt
            kwargs.update({'types_prompt': self.anno_config['types_prompt'][dataset_name]})
        if 'guidelines' in self.anno_config.keys() and self.anno_config['guidelines']:
            kwargs.update({'guidelines': self.anno_config['guidelines']})
        sys_prompt = self.anno_config['prompt_template'].format(**kwargs)

        chat_message = [{"role": "system", "content": sys_prompt}]
        if use_api and self.api_cfg['model'] == 'qwen-long':
            # see https://help.aliyun.com/document_detail/2788814.html?spm=a2c4g.2788811.0.0.1440240aUbuyYI#b7f81199e2laz
            # when use qwen-long, we should add an extra system message for role-play to the chat_message
            chat_message = [{'role': 'system', 'content': self.anno_config['system_role']}] + chat_message
        return chat_message

    def _pipeline_msg(self, annotator_cfg, dataset_name, use_api=False) -> list[None | dict[str, str]]:
        """
        Get examples and init the chat messages template for the annotation models using 2-stage pipeline.

        :param annotator_cfg: The parameters of the annotation model.
        :param dataset_name: the name of the dataset.
        :param use_api: whether to use LLM API as annotator
        :return:
        """
        if annotator_cfg['chat']:
            pass
        else:
            if 'examples' in self.anno_config.keys() and self.anno_config['examples'][dataset_name]:
                examples = ''
                for idx, example in enumerate(self.anno_config['examples'][dataset_name]):
                    instance = self.anno_config['instance_template'].format(sentence=example['sentence'], output=example['output'])
                    examples += f'{idx + 1})\n{instance}\n'
            else:  # 0-shot
                examples = None
        return self._init_chat_msg_template(examples, use_api=use_api, dataset_name=dataset_name)

    def _st_msg(self, annotator_cfg, dataset_name, use_api=False) -> list[None | dict[str, str]]:
        """
        Get examples and init the chat messages for the annotation models to extract entity directly by annotators according the single_type_prompt.

        :param annotator_cfg: The parameters of the annotation model.
        :param dataset_name: the name of the dataset.
        :param use_api: whether to use LLM API as annotator
        :return:
        """
        if annotator_cfg['chat']:
            pass
        else:
            if 'examples' in self.anno_config.keys() and self.anno_config['examples'][dataset_name]:
                examples = ''
                for idx, example in enumerate(self.anno_config['examples'][dataset_name]):
                    instance = self.anno_config['instance_template'].format(label=example['label'],
                                                                       sentence=example['sentence'],
                                                                       output=example['output'])
                    examples += f'{idx + 1})\n{instance}\n'
            else:   # 0-shot
                examples = None
        return self._init_chat_msg_template(examples, use_api=use_api, dataset_name=dataset_name)

    def _pipeline_fs_msg(self, annotator_cfg, dataset_name, use_api=False) -> list[None | dict[str, str]]:
        """
        Get examples and init the chat messages for the annotation models using 2-stage pipeline with few-shot settings.
        Init examples from the support set sampled from the dataset.

        :param annotator_cfg: The parameters of the annotation model.
        :param dataset_name: the name of the dataset.
        :param use_api: whether to use LLM API as annotator.
        :return:
        """
        if annotator_cfg['chat']:
            pass
        else:
            examples = ''
            spans_o_class = []  # store the spans of 'O' class
            span_nums = []  # count the number of gold spans for each example

            index = 0  # index of the examples
            # 1. for Non-O class
            # k-shot examples to show to the annotator
            k_shot = self.anno_config['k_shot']
            self.anno_config['support_set_dir'] = self.anno_config['support_set_dir'].format(dataset_name=dataset_name)
            if self.anno_config['gold_span']:
                k_shot_file = os.path.join(self.anno_config['support_set_dir'], 'gold_span', f'train_support_set_{k_shot}_shot.jsonl')
            else:
                k_shot_file = os.path.join(self.anno_config['support_set_dir'], 'span', f'train_support_set_{k_shot}_shot.jsonl')
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
                        instance = self.anno_config['instance_template'].format(sentence=sentence, output=output)
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
            sampled_idx = torch.topk(probability, k = self.anno_config['k_shot']).indices.tolist()  # the instance indices to be sampled
            # get the sampled spans of 'O' class and filter the empty spans
            sampled_idx = list(filter(lambda x: len(spans_o_class[x]) > 0, sampled_idx))
            sampled_idx = sampled_idx[:self.anno_config['k_shot']]  # Get the first k_shot elements
            with jsonlines.open(k_shot_file) as reader:
                for idx, line in enumerate(reader):
                    if idx in sampled_idx:
                        start, end, entity_mention = random.choice(list(spans_o_class[idx]))  # randomly select a span in this instance
                        start, end = int(start), int(end)
                        sentence = ' '.join(line['tokens'][:start] + ['[ ', entity_mention, ' ]'] + line['tokens'][end:])
                        output = '{"answer": "O"}'
                        instance = self.anno_config['instance_template'].format(sentence=sentence, output=output)
                        examples += f'{index + 1})\n{instance}\n'
                        index += 1

        return self._init_chat_msg_template(examples, use_api=use_api, dataset_name=dataset_name)

    def _st_fs_msg(self, annotator_cfg, dataset_name, use_api=False) -> list[None | dict[str, str]]:
        """
        Init the chat messages for the annotation models to extract entity directly by annotators according the single_type_prompt.
        Init examples from the support set sampled from the dataset.

        :param annotator_cfg: The parameters of the annotation model.
        :param dataset_name: the name of the dataset.
        :param use_api: whether to use LLM API as annotator
        :return:
        """
        if annotator_cfg['chat']:
            pass
        else:
            examples = ''
            index = 0  # index of the examples
            k_shot = self.anno_config['k_shot']
            self.anno_config['support_set_dir'] = self.anno_config['support_set_dir'].format(dataset_name=dataset_name)
            if self.anno_config['gold_span']:
                k_shot_file = os.path.join(self.anno_config['support_set_dir'], 'gold_span', f'train_support_set_{k_shot}_shot.jsonl')
            else:
                k_shot_file = os.path.join(self.anno_config['support_set_dir'], 'span', f'train_support_set_{k_shot}_shot.jsonl')
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
                            instance = self.anno_config['instance_template'].format(label=label, sentence=sent, output=output)
                            examples += f'{index + 1})\n{instance}\n'
                            index += 1
        return self._init_chat_msg_template(examples, use_api=use_api, dataset_name=dataset_name)

    def _mt_fs_msg(self, annotator_cfg, dataset_name, use_api=False) -> list[None | dict[str, str]]:
        """
        Init the chat messages for the annotation models to extract entity directly by annotators according the multi_type_prompt.
        Init examples from the support set sampled from the dataset.

        :param annotator_cfg: The parameters of the annotation model.
        :param dataset_name: the name of the dataset.
        :param use_api: whether to use LLM API as annotator
        :return:
        """
        if annotator_cfg['chat']:
            pass
        else:
            examples = ''
            index = 0
            k_shot = self.anno_config['k_shot']
            self.anno_config['support_set_dir'] = self.anno_config['support_set_dir'].format(dataset_name=dataset_name)
            if self.anno_config['gold_span']:
                k_shot_file = os.path.join(self.anno_config['support_set_dir'], 'gold_span', f'train_support_set_{k_shot}_shot.jsonl')
            else:
                k_shot_file = os.path.join(self.anno_config['support_set_dir'], 'span', f'train_support_set_{k_shot}_shot.jsonl')
            with jsonlines.open(k_shot_file) as reader:
                for line in reader:
                    sentence = ' '.join((line['tokens']))
                    output = '['
                    for start, end, entity_mention, label_id in line['spans_labels']:
                        label = self.id2label[int(label_id)]
                        output += f'("{label}", "{entity_mention}"),'
                    output += ']'
                    instance = self.anno_config['instance_template'].format(sentence=sentence, output=output)
                    examples += f'{index + 1})\n{instance}\n'
                    index += 1

        return self._init_chat_msg_template(examples, use_api=use_api, dataset_name=dataset_name)

    def _generate_chat_msg(self, instances, chat_msg_template):
        """
        Generate chat messages for each instance. Meanwhile, init the labels for each instance.

        :param instances: The instances to be annotated.
        :param chat_msg_template: The chat message template for the annotating model.
        :return:
        """
        for tokens, spans, spans_labels in zip(instances['tokens'], instances['spans'], instances['spans_labels']):
            if 'single_type_prompt' in self.anno_config.keys() and self.anno_config['single_type_prompt']:
                # generate chat message using the single_type_prompt to extract entity directly by annotators
                sentence = ' '.join(tokens)
                for label, label_id in self.label2id.items():
                    if label == 'O':
                        continue
                    chat_message = copy.deepcopy(chat_msg_template)
                    query = self.anno_config['instance_template'].format(label=label, sentence=sentence, output='')
                    user_prompt = '\n### Query\n' +  query
                    chat_message.append({"role": "user", "content": user_prompt})
                    yield label_id, chat_message, sentence
            elif 'multi_type_prompt' in self.anno_config.keys() and self.anno_config['multi_type_prompt']:
                sentence = ' '.join(tokens)
                chat_message = copy.deepcopy(chat_msg_template)
                query = self.anno_config['instance_template'].format(sentence=sentence, output='')
                user_prompt = '\n### Query\n' + query
                chat_message.append({"role": "user", "content": user_prompt})
                yield chat_message, sentence
            else:
                # do not generate chat message given entity
                for idx, (start, end, entity_mention) in enumerate(spans):
                    start, end = int(start), int(end)
                    sentence = ' '.join(tokens[:start] + ['[ ', entity_mention, ' ]'] + tokens[end:])

                    chat_message = copy.deepcopy(chat_msg_template)
                    query = self.anno_config['instance_template'].format(sentence=sentence, output='')
                    user_prompt = '\n### Query\n' + query
                    chat_message.append({"role": "user", "content": user_prompt})
                    # if self.anno_config['gold_span'] is True, label is the ground truth label id
                    # yield the ID of the sentences, the mention span, the label id of the entity mention and the chat message
                    # else, label is the tuple of the ground truth span and its label, like (start, end, gold_mention_span, gold_label_id)
                    # yield the ID of the sentences, the mention span, gold span and the chat message
                    span = (str(start), str(end), entity_mention)
                    if self.anno_config['gold_span']:  # use the gold span from the annotation
                        # In this case, entity mentions and gold labels are one-to-one
                        label = spans_labels[idx]
                        yield span, label, chat_message, sentence

                    else: # get the span from scratch by spaCy parsers
                        # In this case, entity mention and gold spans with labels are not one-to-one
                        yield span, chat_message, sentence

    @staticmethod
    def _process_output(output_text, sentence, **kwargs):
        """
        process the output text to get the label of the entity mention.
        :param output_text: the text output by the annotator model
        :param sentence: the query sentence
        :param kwargs: other parameters including single_type_prompt, and multi_type_prompt
            single_type_prompt: whether to use the single_type_prompt to extract entity directly by annotators
            multi_type_prompt: whether to use the multi_type_prompt to extract entity directly by annotators
        :return: if the single_type_prompt is True, return the predicted spans and their labels.
        """
        import ast

        output_text = output_text.strip().replace('\_', '_')

        # kwargs['single_type_prompt'] is True, we recognize all entity mention in the output_text given the type
        if 'single_type_prompt' in kwargs.keys() and kwargs['single_type_prompt']:

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
        elif 'multi_type_prompt' in kwargs.keys() and kwargs['multi_type_prompt']:
            out_spans = []
            pattern = r'\[(.*?)\]'  # the pattern to extract a list string
            result = re.search(pattern, output_text, re.DOTALL)  # only find the first list string
            try:
                tmp_spans = ast.literal_eval(result.group(0).strip())  # tmp_spans shapes like [(type 0, mention0),...]
                tmp_spans = filter(lambda e: len(e) == 2, tmp_spans)  # filter the invalid spans
            except Exception:  # the output_text is not valid
                tmp_spans = []

            for label, mention in tmp_spans:
                founded_spans = fu.find_span(sentence, mention)
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

    def get_response(self, client, chat_message, **kwargs):
        """
        Get the response of the annotator using api.
        :param client: the client of the LLM API
        :param chat_message: the chat message to be sent to the api.
        :return:
        """
        # https://help.aliyun.com/zh/dashscope/developer-reference/compatibility-of-openai-with-dashscope/?spm=a2c4g.11186623.0.0.5fef6e432o2fr6

        try:
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
        except Exception as e:
            print(e)
            output = ''
        return output

    async def get_all_batch_response(self, batch_jobs, **kwargs):
        """
        run all batch jobs and wait for their completion
        :param batch_jobs:
        :param kwargs:
        :return:
        """
        res = await asyncio.gather(
            *(self.get_batch_response(batch_id=batch_id,
                                      batch_chat_msg=batch_chat_msg,
                                      **kwargs) for batch_id, batch_chat_msg in batch_jobs))
        return res

    async def get_batch_response(self, client, batch_id, batch_chat_msg, batch_size, **kwargs):
        """
        Get the response of the annotator using batch api. This is specific for GLM API.
        https://open.bigmodel.cn/dev/howuse/batchapi
        :param client: the client of the LLM API
        :param batch_id: the index of the batch
        :param batch_chat_msg: batch chat messages to be sent to the api.
        :param batch_size: batch size
        :param kwargs:
        :return:
        """
        # 1. create request file
        request_file = os.path.join(os.getcwd(), 'request', f'/batch_{batch_id}.jsonl')
        for idx, chat_msg in enumerate(batch_chat_msg):
            with jsonlines.open(request_file, 'w') as writer:
                custom_id = batch_id * batch_size + idx
                writer.write({"custom_id": f"req_{custom_id}",
                              "method": "POST",
                              "url": self.api_cfg['endpoint'],
                              "body": {
                                  "model": self.api_cfg['model'],
                                  "messages": chat_msg,
                                  "top_p": kwargs['top_p'],
                                  "temperature": kwargs['temperature'],
                                  "max_tokens": kwargs['max_tokens']
                                }
                              }
                             )

        # 2. update the request file to get the file id
        uploaded_req = client.files.create(
            file=open(request_file, 'rb'),
            purpose="batch"
        )
        print(f'uploaded request file id: {uploaded_req.id}')
        # 3. create the batch task
        batch_task = client.batches.create(
            input_file_id=uploaded_req.id,
            endpoint=self.api_cfg['endpoint'],
            completion_window="24h",  # only support '24h'
            metadata={
                "description": "named entity recognition"
            }
        )
        print(f'batch task id: {batch_task.id}, status: {batch_task.status}')
        # todo, debug
        while True:
            batch_task = client.batches.retrieve(batch_task.id)
            print(f'batch task id: {batch_task.id}, status: {batch_task.status}')
            if batch_task.status == 'completed':
                # save the response to the disk
                response_file = os.path.join(os.getcwd(), 'response', f'/batch_{batch_id}.jsonl')
                content = client.files.content(batch_task.output_file_id)
                content.write_to_file(response_file)
                break
            await asyncio.sleep(10)
        return batch_task

    def annotate_by_one(self, dataset, queue, dataset_name, **annotator_cfg):
        """
        Annotate the dataset by one specific annotator.
        :param dataset: the dataset to be annotated
        :param annotator_cfg: the config of the annotator
        :param queue: the queue to store the annotation result
        :param dataset_name: the name of the dataset
        :return:
        """
        from vllm import LLM, SamplingParams

        # 0. Some settings
        # 0.1 init wandb
        if 'gold_span' in self.anno_config.keys() and self.anno_config['gold_span']:
            wandb.init(
                project='ontonotes5_annotation_by_llm',
                config=annotator_cfg
            )

        # 0.2 GPU setting
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # avoid parallelism in tokenizers
        # set GPU device
        if self.anno_config['cuda_devices'] == 'all':
            # set the GPU can be used
            cuda_devices = [str(i) for i in range(torch.cuda.device_count())]
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_devices)
        elif len(self.anno_config['cuda_devices']) == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.anno_config['cuda_devices'])
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.anno_config['cuda_devices']
        gpu_num = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))

        # 0.3 check the cache result of this annotator
        annotate_flag = True  # whether to annotate the dataset from scratch
        if self.use_api:
            annotator_name = self.api_cfg['model']+ '-' + annotator_cfg['name']
        else:
            model_name = annotator_cfg['checkpoint'].split('/')[-1].split('-')[0]
            annotator_name =  model_name + '-' + annotator_cfg['name']

        self.anno_config['cache_dir'] = self.anno_config['cache_dir'].format(dataset_name=dataset_name)
        if 'gold_span' in self.anno_config.keys() and self.anno_config['gold_span']:
            this_cache_dir = os.path.join(self.anno_config['cache_dir'], 'gold_span', annotator_name)
        else:
            this_cache_dir = os.path.join(self.anno_config['cache_dir'], 'span', annotator_name)

        try:
            cache_result = load_from_disk(this_cache_dir)
            queue.put(cache_result)
            annotate_flag = False
        except FileNotFoundError:
            os.makedirs(this_cache_dir, exist_ok=True)

        # annotation process
        if annotate_flag:
            # 1. Init the chat messages
            if 'single_type_prompt' in self.anno_config.keys() and self.anno_config['single_type_prompt']:
                if 'k_shot' in self.anno_config.keys() and self.anno_config['k_shot']:  # single_type_prompt with few-shot setting
                    chat_msg_template = self._st_fs_msg(annotator_cfg=annotator_cfg, dataset_name=dataset_name,use_api=self.use_api)
                else:  # single_type_prompt without few-shot setting
                    chat_msg_template = self._st_msg(annotator_cfg=annotator_cfg, dataset_name=dataset_name,use_api=self.use_api)
            elif 'multi_type_prompt' in self.anno_config.keys() and self.anno_config['multi_type_prompt']:
                if 'k_shot' in self.anno_config.keys() and self.anno_config['k_shot']:
                    chat_msg_template = self._mt_fs_msg(annotator_cfg=annotator_cfg, dataset_name=dataset_name,use_api=self.use_api)
            else:
                if 'k_shot' in self.anno_config.keys() and self.anno_config['k_shot']:  # 2-stage pipeline with few-shot setting
                    chat_msg_template = self._pipeline_fs_msg(annotator_cfg=annotator_cfg, dataset_name=dataset_name,use_api=self.use_api)
                else:  # 2-stage pipeline without few-shot setting
                    chat_msg_template = self._pipeline_msg(annotator_cfg=annotator_cfg, dataset_name=dataset_name,use_api=self.use_api)

            # 2. Import the annotating model
            if self.use_api and not self.api_cfg['batch_api']:
                client = OpenAI(
                    api_key=os.getenv(self.api_cfg['api_key']),
                    base_url=self.api_cfg['base_url'],  # base_url
                )
            elif self.use_api and self.api_cfg['batch_api']:
                client = ZhipuAI(
                    api_key=os.getenv(self.api_cfg['api_key']),
                    base_url=self.api_cfg['base_url'],  # base_url
                )
            else:
                # if not use api, we employ local annotator using vllm
                # https://docs.vllm.ai/en/latest/getting_started/quickstart.html
                anno_model = LLM(model=annotator_cfg['checkpoint'],
                                 tensor_parallel_size=gpu_num,
                                 dtype=annotator_cfg['dtype'],
                                 gpu_memory_utilization=annotator_cfg['gpu_memory_utilization'],
                                 trust_remote_code=True)
                sampling_params = SamplingParams(temperature=annotator_cfg['anno_temperature'],
                                                 top_p=annotator_cfg['anno_top_p'],
                                                 max_tokens=annotator_cfg['anno_max_tokens'],
                                                 repetition_penalty=annotator_cfg['repetition_penalty'])

                # get anno_model's tokenizer to apply the chat template
                # https://github.com/vllm-project/vllm/issues/3119
                anno_tokenizer = anno_model.llm_engine.tokenizer.tokenizer

            # 3. batch process
            # yield span, span_label, chat_message
            dataset = dataset.shuffle(seed=42).select(range(200))
            pbar = tqdm(fu.batched(self._generate_chat_msg(instances=dataset,
                                                           chat_msg_template=chat_msg_template, ),
                                   annotator_cfg['anno_bs']),
                        desc=f'annotating by {annotator_name}')

            res_labels, res_label_ids = [], []  # store the output labels and label ids

            # if self.anno_config['gold_span'] is True, we use gold span from annotation
            # y_true stores the ground truth label ids, shaped like [label_id_0, label_id_1, ...]
            # else, we get the span from scratch by spaCy and stanza parsers
            # y_true stores the gold span and its label in a tuple, shaped like [(start, end, gold_mention_span, gold_label_id), ...]
            y_true = []
            pred_spans = [] # store the predicted spans and its label
            output_text = []  # store the output text for each instance
            batch_jobs = []  # store the batch jobs for the batch process, specific for glm batch api
            for batch_id, batch in enumerate(pbar):
                batch_spans = []  # store the span for each batch
                batch_labels = []  # store the gold labels for each batch
                batch_chats = []  # store the chats for each batch
                batch_sents = []  # store the sentences for each batch
                batch_res_label_ids = []  # store the output label ids for each batch to evaluate

                # 3.1 store different information according to the different annotation settings
                if 'single_type_prompt' in self.anno_config.keys() and self.anno_config['single_type_prompt']:
                    # if self.anno_config['single_type_prompt'] is True, batch is a tuple like ((label_0, chat_0, sent_0), (label_1, chat_1, sent_1),...)
                    for label_id, chat, sent in batch:
                        batch_labels.append(label_id)
                        batch_chats.append(chat)
                        batch_sents.append(sent)
                elif 'multi_type_prompt' in self.anno_config.keys() and self.anno_config['multi_type_prompt']:
                    # if self.anno_config['multi_type_prompt'] is True, batch is a tuple like ((chat_0, sent_0), (chat_1, sent_1),...)
                    for chat, sent in batch:
                        batch_chats.append(chat)
                        batch_sents.append(sent)
                else:
                    if self.anno_config['gold_span']:
                        # if self.anno_config['gold_span'] is true,
                        # batch is a tuple like ((span_0, label_0, chat_0, sent_0),(span_1, label_1, chat_1, sent_1)...)
                        for span, label, chat, sent in batch:
                            # if self.anno_config['gold_span'] is True, label is the ground truth label id
                            # else, label is the tuple of the ground truth span and its label, like (start, end, gold_mention_span, gold_label_id)
                            batch_spans.append(span)
                            batch_labels.append(label)
                            y_true.append(label)
                            batch_chats.append(chat)
                            batch_sents.append(sent)
                    else:
                        # else, batch is a tuple like ((span_0, chat_0),(span_1, chat_1)...)
                        # we  get all gold span labels after annotation
                        for span, chat, sent in batch:
                            batch_spans.append(span)
                            batch_chats.append(chat)
                            batch_sents.append(sent)

                # 3.2 get the response of the annotator
                outputs = None
                if self.use_api and not self.api_cfg['batch_api']:
                    outputs = []
                    for chat in batch_chats:
                        output = self.get_response(client=client,
                                                   chat_message=chat,
                                                   stream=annotator_cfg['stream'],
                                                   temperature=annotator_cfg['anno_temperature'],
                                                   top_p=annotator_cfg['anno_top_p'],
                                                   max_tokens=annotator_cfg['anno_max_tokens'])
                        if output == '':
                            continue
                        outputs.append(output)
                elif self.use_api and self.api_cfg['batch_api']:
                    # batch process using glm
                    # get the batch response using glm
                    batch_jobs.append((batch_id, batch_chats))
                else:
                    # we should use tokenizer.apply_chat_template to add generation template to the chats explicitly
                    templated_batch_chats = anno_tokenizer.apply_chat_template(batch_chats, add_generation_prompt=True, tokenize=False)
                    outputs = anno_model.generate(templated_batch_chats, sampling_params)  # annotate
                    # for test
                    # test_answer = []
                    # for output in outputs:
                    #     test_answer.append({'prompt': output.prompt, 'output': output.outputs[0].text})
                    outputs = [e.outputs[0].text for e in outputs]

                # 3.3 process directly the output if not using bacth api
                if outputs:
                    for out_idx, (output, sent) in enumerate(zip(outputs, batch_sents)):
                        output_text.append(output)
                        if 'single_type_prompt' in self.anno_config.keys() and self.anno_config['single_type_prompt']:
                            if 'analysis' in self.anno_config.keys() and self.anno_config['analysis']:
                                out_spans = self._process_output(output, sent, single_type_prompt=self.anno_config['single_type_prompt'], analysis=self.anno_config['analysis'])
                            else:
                                out_spans = self._process_output(output, sent, single_type_prompt=self.anno_config['single_type_prompt'])
                            if len(out_spans) == 0:
                                continue
                            out_label_id = batch_labels[out_idx]
                            pred_spans += [(*out_span, str(out_label_id)) for out_span in set(out_spans)]
                        elif 'multi_type_prompt' in self.anno_config.keys() and self.anno_config['multi_type_prompt']:
                            out_spans = self._process_output(output, sent, multi_type_prompt=self.anno_config['multi_type_prompt'])
                            if len(out_spans) == 0:
                                continue
                            for start, end, span, label in set(out_spans):
                                if label not in self.label2id.keys():
                                    label = 'O'
                                out_label_id = self.label2id[label]
                                pred_spans.append((str(start), str(end), span, str(out_label_id)))
                        else:
                            out_label = self._process_output(output, sent)
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

                    # 3.4 evaluate batch results
                    if 'gold_span' in self.anno_config.keys() and self.anno_config['gold_span']:
                        wandb.log({'f1-macro': f1_score(y_true=batch_labels, y_pred=batch_res_label_ids, average='macro', zero_division=0),
                                   'precision-weighted': precision_score(y_true=batch_labels, y_pred=batch_res_label_ids, average='weighted', zero_division=0),
                                   'recall-weighted': recall_score(y_true=batch_labels, y_pred=batch_res_label_ids, average='weighted', zero_division=0),
                                   'accuracy & f1-micro': accuracy_score(y_true=batch_labels, y_pred=batch_res_label_ids),
                                   'matthews_corrcoef': matthews_corrcoef(y_true=batch_labels, y_pred=batch_res_label_ids),
                                   'cohen_kappa_score': cohen_kappa_score(y1=batch_labels, y2=batch_res_label_ids)
                                     })

            # 3.4 process the batch results if using batch api
            if not outputs:

                asyncio.run(self.get_all_batch_response(batch_jobs,
                                                        client=client,
                                                        batch_size=annotator_cfg['anno_bs'],
                                                        stream=annotator_cfg['stream'],
                                                        temperature=annotator_cfg['anno_temperature'],
                                                        top_p=annotator_cfg['anno_top_p'],
                                                        max_tokens=annotator_cfg['anno_max_tokens']))
                outputs = []
                response_dir = os.path.join(os.getcwd(), 'response')
                for path , dir_lst, file_lst in response_dir:
                    for file in file_lst:
                        with jsonlines.open(os.path.join(path, file), 'r') as reader:
                            for line in reader:
                                outputs.append(line['response']['body']['choices'][0]['message']['content'])

            # 5. cache and evaluate the annotation result
            if 'gold_span' in self.anno_config.keys() and self.anno_config['gold_span']:
                cache_result = {
                    "y_true": y_true,  # the ground truth label ids, shaped like [label_id_0, label_id_1, ...]
                    "labels": res_labels,  # the output labels, shaped like [label_0, label_1, ...]
                    "label_ids": res_label_ids,  # the output label ids, shaped like [label_id_0, label_id_1, ...]
                    "output_text": output_text, # shape like [out_text_0, out_text1, ...]
                }

            else:
                # y_true is shape of [(start, end, gold_mention_span, gold_label_id), ...]
                # pred_spans is shape of [(start, end, pred_mention_span, pred_label_id), ...],
                # they are not one-to-one, so we convert them into 2-d list
                res_y_true, res_pred_spans = [], []
                y_true = [span_label for spans_labels in dataset['spans_labels'] for span_label in spans_labels]
                res_y_true.append(y_true)
                res_pred_spans.append(pred_spans)

                res_output_text = [output_text]
                cache_result = {
                    "y_true": res_y_true,  # the gold span and its label, shaped like [(start, end, gold_mention_span, gold_label_id), ...]
                    "pred_spans": res_pred_spans,  # the predicted spans and its label, shaped like [(start, end, pred_mention_span, pred_label_id), ...]
                    "output_text": res_output_text
                }

            cache_result = Dataset.from_dict(cache_result)
            queue.put(cache_result)
            cache_result.save_to_disk(this_cache_dir)

        # 6. evaluation
        if self.anno_config['gold_span']:
            self.evaluate(cache_result['y_true'], cache_result['label_ids'], dataset_name, annotator_name)
        else:
            # remove the '0'('O' label) span from the pred_spans
            pred_spans = [span for span in cache_result['pred_spans'][0] if int(span[-1]) != self.label2id['O']]
            self.evaluate(cache_result['y_true'][0], pred_spans, dataset_name, annotator_name)
        return cache_result


    def annotate_augmented(self, dataset, quality=True, **kwargs):
        """
        Augmented annotation.
        1) First, we get candidate entity mentions (denoted as Ec) using constituency parsing.
        2) Then, we annotate the dataset using multi type prompt. The recognized entity mentions are denoted as 'Er'.
        3) Third, we filter out candidate entity mentions containing Er from Ec and annotate the remaining entity mentions
        using pipeline annotation.
        :param dataset:
        :param quality:
        :param kwargs:
        :return:
        """
        # todo

    def annotate_by_all(self, dataset, quality=True, **kwargs):
        """
        Annotate the dataset by all annotators.
        :param dataset: the dataset to be annotated
        :param quality: whether to evaluate the quality of the annotations
        :return:
        """
        from multiprocessing import Process, Queue

        # 1. start process for each annotator
        queue = Queue()
        for annotator_cfg in self.annotators_cfg:
            p = Process(target=self.annotate_by_one, args=(dataset, queue, kwargs['dataset_name']), kwargs=annotator_cfg)
            p.start()
            p.join()  # wait for the process to finish so that we can release the GPU memory for the next process

        quality_data = []  # store the prediction for each annotator
        while not queue.empty():
            result = queue.get()
            if self.anno_config['gold_span']:
                quality_data.append(result['label_ids'])
            else:
                quality_data.append(result['pred_spans'])

        # 2. evaluate the annotation quality
        if quality:
            self.anno_config['eval_dir'] = self.anno_config['eval_dir'].format(dataset_name=kwargs['dataset_name'])
            if self.anno_config['gold_span']:
                qual_res_file = os.path.join(self.anno_config['eval_dir'], 'gold_span', 'quality_res.txt')
            else:
                qual_res_file = os.path.join(self.anno_config['eval_dir'], 'span','quality_res.txt')
            quality_data = np.array(quality_data)  # quality_data with shape (num_annotators, num_instances)
            # quality_data with shape (num_instances, num_annotators)
            # transpose the quality_data to get the shape (num_instances, num_annotators)
            quality_res = fu.eval_anno_quality(quality_data.T)
            with open(qual_res_file, 'w') as f:
                for metric, res in quality_res.items():
                    f.write(f'{metric}: {res}\n')

    def evaluate(self, y_true, y_pred, dataset_name, annotator_name: str):
        """
        Evaluate the annotation results by an annotator.
        :param y_true: if self.anno_config['gold_span'] is True, we use gold span from annotation, y_true stores the ground truth label ids, shaped like
        [label_id_0, label_id_1, ...]. Else, we get the span from scratch by parser, y_true stores the gold span and their labels in a tuple,
        shaped like [(start, end, gold_mention_span, gold_label_id), ...].
        :param y_pred: if self.anno_config['gold_span'] is True, y_pred stores the predicted label ids. Else, y_pred stores the predicted spans and their labels.
        :param dataset_name: the name of the dataset.
        :param annotator_name: the name of the annotator LLM.
        :return:
        """
        self.anno_config['eval_dir'] = self.anno_config['eval_dir'].format(dataset_name=dataset_name)

        if self.anno_config['gold_span']:
            # compute all classification metrics
            eval_results = {'f1-macro': f1_score(y_true=y_true, y_pred=y_pred, average='macro'),
                            'precision-weighted': precision_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0),
                            'recall-weighted': recall_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0),
                            'accuracy & f1-micro': accuracy_score(y_true=y_true, y_pred=y_pred),
                            'matthews_corrcoef': matthews_corrcoef(y_true=y_true, y_pred=y_pred),
                            'cohen_kappa_score': cohen_kappa_score(y1=y_true, y2=y_pred)}
            res_cache_dir = os.path.join(self.anno_config['eval_dir'], 'gold_span')
        else:
            # compute span-level metrics
            eval_results = fu.compute_span_f1(y_true,  y_pred)
            res_cache_dir = os.path.join(self.anno_config['eval_dir'], 'span')

        if not os.path.exists(res_cache_dir):
            os.makedirs(res_cache_dir, exist_ok=True)
        res_file = os.path.join(res_cache_dir, f'{annotator_name}_res.txt')
        with open(res_file, 'w') as f:
            for metric, res in eval_results.items():
                f.write(f'{metric}: {res}\n')