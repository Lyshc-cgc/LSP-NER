import copy
import json
import random
import re
import os

import jsonlines
import wandb
import torch
from openai import OpenAI
import numpy as np
from datasets import load_from_disk, Dataset
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, matthews_corrcoef, cohen_kappa_score
from module import func_util as fu
from module.label import Label

class Annotation(Label):
    """
    The Annotation class is used to annotate the data.
    """
    def __init__(self, anno_cfg):
        super().__init__()
        self.anno_config = fu.get_config(anno_cfg)
        self.annotators_cfg = self.anno_config['annotators']
        self.annotator_ids = dict()
        for idx, anno_cfg in enumerate(self.annotators_cfg):
            self.annotator_ids[anno_cfg['name']] = idx

    def _init_usr_prompt(self, examples) -> str:
        """
        Init the user's prompt for the annotation models.
        :param examples: the examples to be shown to annotators.
        :return:
        """
        kwargs = {'examples': examples}
        if 'system_role' in self.anno_config.keys() and self.anno_config['system_role']:
            kwargs.update({'system_role': self.anno_config['system_role']})
        if 'task_prompt' in self.anno_config.keys() and self.anno_config['task_prompt']:
            kwargs.update({'task_prompt': self.anno_config['task_prompt']})
        if 'types_prompt' in self.anno_config.keys() and self.anno_config['types_prompt']:
            kwargs.update({'types_prompt': self.anno_config['types_prompt']})
        if 'guidelines' in self.anno_config.keys() and self.anno_config['guidelines']:
            kwargs.update({'guidelines': self.anno_config['guidelines']})

        usr_prompt = self.anno_config['prompt_template'].format(**kwargs)
        return usr_prompt

    def _init_chat_message(self, anno_cfg) -> list[None | dict[str, str]]:
        """
        Init the chat messages for the annotation models using 2-stage pipeline.

        :param anno_cfg: The parameters of the annotation model.
        :return:
        """
        if anno_cfg['chat']:
            pass
        else:
            examples = ''
            for idx, example in enumerate(self.anno_config['examples']):
                instance = self.anno_config['instance_template'].format(sentence=example['sentence'], output=example['output'])
                examples += f'{idx + 1})\n{instance}'
            usr_prompt = self._init_usr_prompt(examples)
            chat_message = [{"role": "user", "content": usr_prompt}]
        return chat_message

    def _init_chat_message_st(self, anno_cfg) -> list[None | dict[str, str]]:
        """
        Init the chat messages for the annotation models to extract entity directly by annotators according the single_type_prompt.

        :param anno_cfg: The parameters of the annotation model.
        :return:
        """
        if anno_cfg['chat']:
            pass
        else:
            examples = ''
            for idx, example in enumerate(self.anno_config['examples']):
                instance = self.anno_config['instance_template'].format(label=example['label'],
                                                                   sentence=example['sentence'],
                                                                   output=example['output'])
                examples += f'{idx + 1})\n{instance}\n'
            usr_prompt = self._init_usr_prompt(examples)
            chat_message = [{"role": "user", "content": usr_prompt}]
        return chat_message

    def _init_chat_message_fs(self, anno_cfg) -> list[None | dict[str, str]]:
        """
        Init the chat messages for the annotation models using 2-stage pipeline with few-shot settings.
        Init examples from the support set sampled from the dataset.

        :param anno_cfg: The parameters of the annotation model.
        :return:
        """
        if anno_cfg['chat']:
            pass
        else:
            examples = ''
            spans_o_class = []  # store the spans of 'O' class
            span_nums = []  # count the number of gold spans for each example

            index = 0  # index of the examples
            # 1. for Non-O class
            # k-shot examples to show to the annotator
            k_shot = self.anno_config['k_shot']
            if self.anno_config['gold_span']:
                k_shot_file = os.path.join(self.anno_config['support_set_dir'], 'gold_span', f'train_support_set_{k_shot}_shot.jsonl')
            else:
                k_shot_file = os.path.join(self.anno_config['support_set_dir'], 'span', f'train_support_set_{k_shot}_shot.jsonl')
            with jsonlines.open(self.anno_config['']) as reader:
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
            sampled_idx = torch.topk(probability, k = 2*self.anno_config['k_shot']).indices.tolist()  # the instance indices to be sampled
            # get the sampled spans of 'O' class and filter the empty spans
            sampled_idx = list(filter(lambda x: len(spans_o_class[x]) > 0, sampled_idx))
            sampled_idx = sampled_idx[:self.anno_config['k_shot']]  # Get the first k_shot elements
            with jsonlines.open(self.anno_config['k_shot_file']) as reader:
                for idx, line in enumerate(reader):
                    if idx in sampled_idx:
                        start, end, entity_mention = random.choice(list(spans_o_class[idx]))  # randomly select a span in this instance
                        start, end = int(start), int(end)
                        sentence = ' '.join(line['tokens'][:start] + ['[ ', entity_mention, ' ]'] + line['tokens'][end:])
                        output = '{"answer": "O"}'
                        instance = self.anno_config['instance_template'].format(sentence=sentence, output=output)
                        examples += f'{index + 1})\n{instance}\n'
                        index += 1

            usr_prompt = self._init_usr_prompt(examples)
            chat_message = [{"role": "user", "content": usr_prompt}]
        return chat_message

    def _init_chat_message_st_fs(self, anno_cfg) -> list[None | dict[str, str]]:
        """
        Init the chat messages for the annotation models to extract entity directly by annotators according the single_type_prompt.
        Init examples from the support set sampled from the dataset.

        :param anno_cfg: The parameters of the annotation model.
        :return:
        """
        if anno_cfg['chat']:
            pass
        else:
            examples = ''
            index = 0  # index of the examples
            k_shot = self.anno_config['k_shot']
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
            usr_prompt = self._init_usr_prompt(examples)
            chat_message = [{"role": "user", "content": usr_prompt}]
        return chat_message

    def _generate_chat_messages(self, instances, chat_message_template):
        """
        Generate chat messages for each instance. Meanwhile, init the labels for each instance.

        :param instances: The instances to be annotated.
        :param chat_message_template: The chat message template for the annotating model.
        :return:
        """
        for tokens, spans, spans_labels in zip(instances['tokens'], instances['spans'], instances['spans_labels']):
            if 'single_type_prompt' in self.anno_config.keys() and self.anno_config['single_type_prompt']:
                # generate chat message using the single_type_prompt to extract entity directly by annotators
                sentence = ' '.join(tokens)
                for label, label_id in self.label2id.items():
                    chat_message = copy.deepcopy(chat_message_template)
                    query = self.anno_config['instance_template'].format(label=label, sentence=sentence, output='')
                    user_prompt = chat_message[-1]['content'] + '\n' + query
                    chat_message[-1] = {"role": "user", "content": user_prompt}
                    yield label_id, chat_message
            else:
                # do not generate chat message given entity
                for idx, (start, end, entity_mention) in enumerate(spans):
                    start, end = int(start), int(end)
                    sentence = ' '.join(tokens[:start] + ['[ ', entity_mention, ' ]'] + tokens[end:])

                    chat_message = copy.deepcopy(chat_message_template)
                    query = self.anno_config['instance_template'].format(sentence=sentence, output='')
                    user_prompt = chat_message[-1]['content'] + '\n' + query
                    chat_message[-1] = {"role": "user", "content": user_prompt}
                    # if self.anno_config['gold_span'] is True, label is the ground truth label id
                    # yield the ID of the sentences, the mention span, the label id of the entity mention and the chat message
                    # else, label is the tuple of the ground truth span and its label, like (start, end, gold_mention_span, gold_label_id)
                    # yield the ID of the sentences, the mention span, gold span and the chat message
                    span = (str(start), str(end), entity_mention)
                    if self.anno_config['gold_span']:  # use the gold span from the annotation
                        # In this case, entity mentions and gold labels are one-to-one
                        label = spans_labels[idx]
                        yield span, label, chat_message

                    else: # get the span from scratch by spaCy parsers
                        # In this case, entity mention and gold spans with labels are not one-to-one
                        yield span, chat_message

    @staticmethod
    def _process_output_text(output_text, **kwargs):
        """
        process the output text to get the label of the entity mention.
        :param output_text: the text output by the annotator model
        :param kwargs: other parameters including single_type_prompt, and multi_type_prompt
            single_type_prompt: whether to use the single_type_prompt to extract entity directly by annotators
            multi_type_prompt: whether to use the multi_type_prompt to extract entity directly by annotators
        :return: if the single_type_prompt is True, return the predicted spans and their labels.
        """

        output_text = output_text.strip().replace('\_', '_')

        # kwargs['single_type_prompt'] is True, we recognize all entity mention given the type
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
            pattern = r'@@(\b.*?\b)##'
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
                    except (json.decoder.JSONDecodeError, KeyError):
                        continue
            return out_label

    def get_response(self, chat_message):
        """
        Get the response of the annotator using api.
        :return:
        """
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),  # API_KEY
            base_url=self.api_config['base_url'],  # base_url
        )
        completion = client.chat.completions.create(
            model=self.api_config['model'],
            messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                      {'role': 'user', 'content': '你是谁？'}]
        )
        pass


    def annotate_by_one(self, dataset, queue, **anno_cfg):
        """
        Annotate the dataset by one specific annotator.
        :param dataset: the dataset to be annotated
        :param anno_cfg: the config of the annotator
        :param queue: the queue to store the annotation result
        :return:
        """
        import ray
        from vllm import LLM, SamplingParams

        # 0. Some settings
        # 0.1 init wandb
        if 'gold_span' in self.anno_config.keys() and self.anno_config['gold_span']:
            wandb.init(
                project='ontonotes5_annotation_by_llm',
                config=anno_cfg
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
        annotator_name = anno_cfg['name']
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
                    chat_message_template = self._init_chat_message_st_fs(anno_cfg)
                else:  # single_type_prompt without few-shot setting
                    chat_message_template = self._init_chat_message_st(anno_cfg)
            else:
                if 'k_shot' in self.anno_config.keys() and self.anno_config['k_shot']:  # 2-stage pipeline with few-shot setting
                    chat_message_template = self._init_chat_message_fs(anno_cfg)
                else:  # 2-stage pipeline without few-shot setting
                    chat_message_template = self._init_chat_message(anno_cfg)

            # 2. Import the annotating model
            # https://docs.vllm.ai/en/latest/getting_started/quickstart.html

            anno_model = LLM(model=anno_cfg['checkpoint'],
                             tensor_parallel_size=gpu_num,
                             dtype=anno_cfg['dtype'],
                             gpu_memory_utilization=anno_cfg['gpu_memory_utilization'],
                             trust_remote_code=True)
            sampling_params = SamplingParams(temperature=anno_cfg['anno_temperature'],
                                             top_p=anno_cfg['anno_top_p'],
                                             max_tokens=anno_cfg['anno_max_tokens'],
                                             repetition_penalty=anno_cfg['repetition_penalty'])

            # get anno_model's tokenizer to apply the chat template
            # https://github.com/vllm-project/vllm/issues/3119
            anno_tokenizer = anno_model.llm_engine.tokenizer.tokenizer

            # 3. batch process
            # yield span, span_label, chat_message
            dataset = dataset.select(range(100))
            pbar = tqdm(fu.batched(self._generate_chat_messages(instances=dataset,
                                                                chat_message_template=chat_message_template,),
                                   anno_cfg['anno_bs']),
                        desc=f'annotating by {annotator_name}')

            res_labels, res_label_ids = [], []  # store the output labels and label ids

            # if self.anno_config['gold_span'] is True, we use gold span from annotation
            # y_true stores the ground truth label ids, shaped like [label_id_0, label_id_1, ...]
            # else, we get the span from scratch by spaCy and stanza parsers
            # y_true stores the gold span and its label in a tuple, shaped like [(start, end, gold_mention_span, gold_label_id), ...]
            y_true = []
            pred_spans = [] # store the predicted spans and its label
            output_text = []  # store the output text for each instance
            for batch in pbar:
                batch_spans, batch_labels, batch_chats = [], [], []  # store the span, the gold label ids / the gold spans, chat for each batch
                batch_res_label_ids = []  # store the output label ids for each batch to evaluate
                if 'single_type_prompt' in self.anno_config.keys() and self.anno_config['single_type_prompt']:
                    # if self.anno_config['single_type_prompt'] is True, batch is a tuple like ((label_0, chat_0), (label_1, chat_1),...)
                    for label_id, chat in batch:
                        batch_labels.append(label_id)
                        batch_chats.append(chat)
                else:
                    if self.anno_config['gold_span']:
                        # if self.anno_config['gold_span'] is true,
                        # batch is a tuple like ((span_0, label_0, chat_0),(span_1, label_1, chat_1)...)
                        for span, label, chat in batch:
                            # if self.anno_config['gold_span'] is True, label is the ground truth label id
                            # else, label is the tuple of the ground truth span and its label, like (start, end, gold_mention_span, gold_label_id)
                            batch_spans.append(span)
                            batch_labels.append(label)
                            y_true.append(label)
                            batch_chats.append(chat)
                    else:
                        # else, batch is a tuple like ((span_0, chat_0),(span_1, chat_1)...)
                        # we  get all gold span labels after annotation
                        for span, chat in batch:
                            batch_spans.append(span)
                            batch_chats.append(chat)

                # we should use tokenizer.apply_chat_template to add generation template to the chats explicitly
                templated_batch_chats = anno_tokenizer.apply_chat_template(batch_chats, add_generation_prompt=True, tokenize=False)
                outputs = anno_model.generate(templated_batch_chats, sampling_params)  # annotate
                # for test
                test_answer = []
                for output in outputs:
                    test_answer.append({'prompt': output.prompt, 'output': output.outputs[0].text})
                for out_idx, output in enumerate(outputs):
                    output_text.append(output.outputs[0].text)
                    if 'single_type_prompt' in self.anno_config.keys() and self.anno_config['single_type_prompt']:
                        if 'analysis' in self.anno_config.keys() and self.anno_config['analysis']:
                            out_spans = self._process_output_text(output.outputs[0].text, single_type_prompt=self.anno_config['single_type_prompt'], analysis=self.anno_config['analysis'])
                        else:
                            out_spans = self._process_output_text(output.outputs[0].text, single_type_prompt=self.anno_config['single_type_prompt'])
                        if len(out_spans) == 0:
                            continue
                        out_label_id = batch_labels[out_idx]
                        pred_spans += [(*out_span, str(out_label_id)) for out_span in set(out_spans)]
                    else:
                        out_label = self._process_output_text(output.outputs[0].text)
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
                # print(pred_spans)

                # evaluate batch results
                if 'gold_span' in self.anno_config.keys() and self.anno_config['gold_span']:
                    wandb.log({'f1-macro': f1_score(y_true=batch_labels, y_pred=batch_res_label_ids, average='macro', zero_division=0),
                               'precision-weighted': precision_score(y_true=batch_labels, y_pred=batch_res_label_ids, average='weighted', zero_division=0),
                               'recall-weighted': recall_score(y_true=batch_labels, y_pred=batch_res_label_ids, average='weighted', zero_division=0),
                               'accuracy & f1-micro': accuracy_score(y_true=batch_labels, y_pred=batch_res_label_ids),
                               'matthews_corrcoef': matthews_corrcoef(y_true=batch_labels, y_pred=batch_res_label_ids),
                               'cohen_kappa_score': cohen_kappa_score(y1=batch_labels, y2=batch_res_label_ids)
                                 })

            ray.shutdown()

            # 5. cache and evaluate the annotation result
            res_output_text = [output_text]
            if 'gold_span' in self.anno_config.keys() and self.anno_config['gold_span']:
                cache_result = {
                    "y_true": y_true,  # the ground truth label ids, shaped like [label_id_0, label_id_1, ...]
                    "labels": res_labels,  # the output labels, shaped like [label_0, label_1, ...]
                    "label_ids": res_label_ids,  # the output label ids, shaped like [label_id_0, label_id_1, ...]
                    "output_text": res_output_text
                }

            else:
                # y_true is shape of [(start, end, gold_mention_span, gold_label_id), ...]
                # pred_spans is shape of [(start, end, pred_mention_span, pred_label_id), ...],
                # they are not one-to-one, so we convert them into 2-d list
                res_y_true, res_pred_spans = [], []
                y_true = [span_label for spans_labels in dataset['spans_labels'] for span_label in spans_labels]
                res_y_true.append(y_true)
                res_pred_spans.append(pred_spans)

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
            self.evaluate(cache_result['y_true'], cache_result['label_ids'], annotator_name)
        else:
            # remove the '0'('O' label) span from the pred_spans
            pred_spans = [span for span in cache_result['pred_spans'][0] if int(span[-1]) != self.label2id['O']]
            self.evaluate(cache_result['y_true'][0], pred_spans, annotator_name)
        return cache_result

    def annotate_by_all(self, dataset, quality=True,**kwargs):
        """
        Annotate the dataset by all annotators.
        :param dataset: the dataset to be annotated
        :param quality: whether to evaluate the quality of the annotations
        :return:
        """
        from multiprocessing import Process, Queue

        # 1. start process for each annotator
        queue = Queue()
        for anno_cfg in self.annotators_cfg:
            p = Process(target=self.annotate_by_one, args=(dataset, queue), kwargs=anno_cfg)
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

    def evaluate(self, y_true, y_pred, annotator_name: str):
        """
        Evaluate the annotation results by an annotator.
        :param y_true: if self.anno_config['gold_span'] is True, we use gold span from annotation, y_true stores the ground truth label ids, shaped like
        [label_id_0, label_id_1, ...]. Else, we get the span from scratch by parser, y_true stores the gold span and their labels in a tuple,
        shaped like [(start, end, gold_mention_span, gold_label_id), ...].
        :param y_pred: if self.anno_config['gold_span'] is True, y_pred stores the predicted label ids. Else, y_pred stores the predicted spans and their labels.
        :param annotator_name: the name of the annotator LLM.
        :return:
        """

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