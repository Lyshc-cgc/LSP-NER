
import json
import re
import os
import copy
import wandb
import torch
import numpy as np
from vllm import LLM, SamplingParams
from datasets import load_dataset, load_from_disk, Dataset
from func_util import batched, get_config, eval_anno_quality
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

_LABELS = {"O": 0, "B-CARDINAL": 1, "B-DATE": 2, "I-DATE": 3, "B-PERSON": 4, "I-PERSON": 5, "B-NORP": 6, "B-GPE": 7,
           "I-GPE": 8, "B-LAW": 9, "I-LAW": 10, "B-ORG": 11, "I-ORG": 12, "B-PERCENT": 13, "I-PERCENT": 14,
           "B-ORDINAL": 15, "B-MONEY": 16, "I-MONEY": 17, "B-WORK_OF_ART": 18, "I-WORK_OF_ART": 19, "B-FAC": 20,
           "B-TIME": 21, "I-CARDINAL": 22, "B-LOC": 23, "B-QUANTITY": 24, "I-QUANTITY": 25, "I-NORP": 26, "I-LOC": 27,
           "B-PRODUCT": 28, "I-TIME": 29, "B-EVENT": 30, "I-EVENT": 31, "I-FAC": 32, "B-LANGUAGE": 33, "I-PRODUCT": 34,
           "I-ORDINAL": 35, "I-LANGUAGE": 36}

class Label:
    """
    The Label class is used to store the labels, label ids.
    """
    def __init__(self):
        self.labels = _LABELS
        self.label2id = dict()
        self.id2label = dict()
        self.covert_tag2id = dict()  # covert the original label (tag) id to the new label id. e.g., {2:2,3:2} means 2 (B-DATE) and 3 (I-DATE) are the same (DATE, id=2).
        self.init_labels()

    def init_labels(self):
        idx = 0
        for k, v in _LABELS.items():
            if k.startswith('B-') or k.startswith('I-'):
                label = k.replace('B-', '').replace('I-', '')
            else:
                label = k
            if label not in self.label2id.keys():
                self.label2id[label] = idx
                self.id2label[idx] = label
                idx += 1
            if v not in self.covert_tag2id.keys():
                # e.g., {2:2,3:2} means 2 (B-DATE) and 3 (I-DATE) are the same (DATE, id=2).
                self.covert_tag2id[v] = self.label2id[label]

class Processor(Label):
    """
    The Processor class is used to process the data.
    """
    def __init__(self, data_cfg):
        super().__init__()
        self.config = get_config(data_cfg)
        self.num_proc = self.config['num_proc']

    def _data_format(self, instances):
        """
        Format the data in the required format.
        From 'BIO' tag into span format.
        original format https://huggingface.co/datasets/tner/ontonotes5
        :param instances:
        :return:
        """
        res_spans = []
        res_spans_labels = []
        for inst_id, (tokens, tags) in enumerate(zip(instances['tokens'], instances['tags'])):
            instance_spans = []  # store spans for each instance
            instance_spans_labels = []  # store labels for each span of each instance
            idx = 0
            span = []  # store tokens in a span
            pre_tag = 0  # the previous tag
            start, end = 0, 0  # the start/end index for a span
            while idx < len(tokens):
                tag = tags[idx]
                if tag != 0:
                    if pre_tag != 0 and self.covert_tag2id[tag] == self.covert_tag2id[pre_tag]:  # the token is in the same span
                        # append the token into the same span
                        span.append(tokens[idx])
                        end = idx + 1  # exclusive
                    else:  # the previous is a 'O' token or previous token is not in the same span
                        # store the previous span
                        if len(span) > 0:
                            instance_spans.append((str(start), str(end), ' '.join(span)))
                            span_tag = tags[start]  # the label of the span, we use the label of the first token in the span
                            instance_spans_labels.append(self.covert_tag2id[span_tag])
                        # init a new span
                        span.clear()
                        span.append(tokens[idx])
                        start = idx
                        end = idx + 1  # exclusive
                # the token is an 'O' token
                pre_tag = tag
                idx += 1
            # store the last span
            if len(span) > 0:
                instance_spans.append((str(start), str(end), ' '.join(span)))
                instance_spans_labels.append(self.covert_tag2id[tags[start]])
            res_spans.append(instance_spans)
            res_spans_labels.append(instance_spans_labels)
        return {
            'tokens': instances['tokens'],
            'tags': instances['tags'],
            'spans': res_spans,
            'spans_labels': res_spans_labels
        }

    def process(self):
        # 1. check and load the cache file
        try:
            formated_dataset = load_from_disk(self.config['save_dir'])
        except FileNotFoundError:
            # 2. format datasets to get span from scratch
            raw_dataset = load_dataset(self.config['data_path'], num_proc=self.config['num_proc'])
            formated_dataset = raw_dataset.map(self._data_format,batched=True,num_proc=self.num_proc)

            os.makedirs(self.config['save_dir'], exist_ok=True)
            formated_dataset.save_to_disk(self.config['save_dir'])

        if self.config['split'] is not None:
            formated_dataset = formated_dataset[self.config['split']]

        if self.config['shuffle']:
            formated_dataset = formated_dataset.shuffle()

        return formated_dataset

class Annotation(Label):
    """
    The Annotation class is used to annotate the data.
    """
    def __init__(self, anno_cfg):
        super().__init__()
        self.config = get_config(anno_cfg)
        self.annotators_cfg = self.config['annotators']
        self.annotator_ids = dict()
        for idx, anno_cfg in enumerate(self.annotators_cfg):
            self.annotator_ids[anno_cfg['name']] = idx

    @staticmethod
    def _init_chat_message(annotator_name: str, **anno_model_cfg) -> list[None | dict[str, str]]:
        """
        Init the chat messages for the annotation models.

        :param annotator_name: The name of the annotation model.
        :param anno_model_cfg: The parameters of the annotation model.
        :return:
        """
        if annotator_name == 'Qwen':
            # for Qwen and Yi-ft, we need to input the system's prompt and the user's prompt
            # https://huggingface.co/Qwen/Qwen1.5-72B-Chat-GPTQ-Int8
            # https://huggingface.co/TheBloke/nontoxic-bagel-34b-v0.2-GPTQ
            chat_message = [
                {"role": "system", "content": anno_model_cfg['anno_sys_prompt_batch']},
            ]
            for example in anno_model_cfg['anno_examples_batch']:
                usr_content = anno_model_cfg['anno_usr_prompt_batch'].format(sentence=example['sentence'])
                chat_message.append({"role": "user", "content": usr_content})
                chat_message.append({"role": "assistant", "content": example['output']})
        else:
            # e.g., for mistral and Yi, we only need to input the user's prompt
            chat_message = []
        return chat_message

    @staticmethod
    def _generate_chat_messages(instances, anno_usr_prompt_batch, types_prompt, chat_message_template):
        """
        Used for stage2.
        Generate chat messages for each instance. Meanwhile, init the labels for each instance.

        :param instances: The instances to be annotated.
        :param anno_usr_prompt_batch: The user prompt for the annotating model.
        :param types: The candidate type words for the annotating model.
        :param chat_message_template: The chat message template for the annotating model.
        :return:
        """
        for sent_id, (tokens, spans, spans_labels) in enumerate(zip(instances['tokens'], instances['spans'], instances['spans_labels'])):
            for span_id, ((start, end, entity_mention), span_label) in enumerate(zip(tuple(spans), spans_labels)):
                start, end = int(start), int(end)
                sentence = ' '.join(tokens[:start] + ['[ ', entity_mention, ' ]'] + tokens[end:])

                chat_message = copy.deepcopy(chat_message_template)
                usr_content = anno_usr_prompt_batch.format(sentence=sentence, types=types_prompt)
                chat_message.append({"role": "user", "content": usr_content})
                # yield the ID of the sentences, the ID of the entity mention, the label id of the entity mention and the chat message
                yield sent_id, span_id, span_label, chat_message

    def annotate_by_one(self, dataset, **anno_cfg):
        """
        Annotate the dataset by one specific annotator.
        :param dataset: the dataset to be annotated
        :param anno_cfg: the config of the annotator
        :return:
        """
        # 0. Some settings
        # 0.1 init wandb
        wandb.init(
            project='ontonotes5_annotation_by_llm',
            config=anno_cfg
        )

        # 0.2 GPU setting
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # avoid parallelism in tokenizers
        # set GPU device
        if self.config['cuda_devices'] == 'all':
            # set the GPU can be used
            cuda_devices = [str(i) for i in range(torch.cuda.device_count())]
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_devices)
        elif len(self.config['cuda_devices']) == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config['cuda_devices'])
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config['cuda_devices']
        gpu_num = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))

        # 0.3 check the cache result of this annotator
        annotator_name = anno_cfg['name']
        this_cache_dir = os.path.join(self.config['cache_dir'], annotator_name)
        try:
            cache_result = load_from_disk(this_cache_dir)
            return cache_result
        except FileNotFoundError:
            os.makedirs(this_cache_dir, exist_ok=True)

        # 1. Import the annotating model
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

        # 2. Init the chat messages
        chat_message_template = self._init_chat_message(annotator_name, **anno_cfg)

        # 3. batch process
        # yield sent_id, span_id, span_label, chat_message
        pbar = tqdm(batched(self._generate_chat_messages(dataset,
                                                         anno_cfg['anno_usr_prompt_batch'],
                                                         self.config['types_prompt'],
                                                         chat_message_template),
                            anno_cfg['anno_bs']),desc=f'annotating by {annotator_name}')

        json_pattern = [r'\{\{(.*?)\}\}', r'\{(.*?)\}']  # the pattern to extract JSON string
        res_labels, res_label_ids = [], []  # store the output labels and label ids
        y_true = []  # store the ground truth label ids
        for batch in pbar:  # batch is a tuple like ((sent_id_0, span_id_0, span_label_0, chat_0),(sent_id_1, span_id_1, span_label_1, chat_1)...)
            batch_sent_ids, batch_span_ids, batch_span_labels, batch_chats = [], [], [], []  # store the sent_id, span_id, the ground truth label ids, chat for each batch
            batch_res_label_ids = []  # store the output label ids for each batch to evaluate
            for sent_id, span_id, span_label, chat in batch:
                batch_sent_ids.append(sent_id)
                batch_span_ids.append(span_id)
                batch_span_labels.append(span_label)
                y_true.append(span_label)
                batch_chats.append(chat)

            # we should use tokenizer.apply_chat_template to add generation template to the chats explicitly
            templated_batch_chats = anno_tokenizer.apply_chat_template(batch_chats, add_generation_prompt=True, tokenize=False)
            outputs = anno_model.generate(templated_batch_chats, sampling_params)  # annotate
            # for test
            test_answer = []
            for output in outputs:
                test_answer.append({'prompt': output.prompt, 'output': output.outputs[0].text})
            for sent_id, span_id, output in zip(batch_sent_ids, batch_span_ids, outputs):
                # extract JSON string from output.outputs[0].text
                output_text = output.outputs[0].text.strip()
                result = None
                if output_text.startswith('{{'):
                    result = re.search(json_pattern[0], output_text,re.DOTALL)  # only extract the first JSON string
                elif output_text.startswith('{'):
                    result = re.search(json_pattern[1], output_text, re.DOTALL)

                out_label = 'O'
                if result:
                    try:
                        tmp = result.string.replace('\_', '_')
                        out_label = json.loads(tmp)['answer'].strip()
                    except json.JSONDecodeError:
                        out_label = 'O'  # we assign 'O' to label to this span if we cannot extract the JSON string, so that we can continue the loop

                if out_label not in self.label2id.keys():
                    # check if the output label is redundant
                    tmp_out_label_0 = out_label.split(' ')[0]
                    tmp_out_label_1 = out_label.split(',')[0]
                    if tmp_out_label_0 in self.label2id.keys():
                        out_label = tmp_out_label_0
                    elif tmp_out_label_1 in self.label2id.keys():
                        out_label = tmp_out_label_1
                    else:
                        out_label = 'O'
                    # print(f'prompt: {output.prompt}')
                    # print(f'>>> out_label:"{out_label}"')

                out_label_id = self.label2id[out_label]
                res_labels.append(out_label)
                res_label_ids.append(out_label_id)
                batch_res_label_ids.append(out_label_id)

            # evaluate batch results
            wandb.log({'f1-micro': f1_score(y_true=batch_span_labels, y_pred=batch_res_label_ids, average='micro'),
                       'precision-micro': precision_score(y_true=batch_span_labels, y_pred=batch_res_label_ids, average='micro'),
                       'recall-micro': recall_score(y_true=batch_span_labels, y_pred=batch_res_label_ids, average='micro'),
                       'accuracy': accuracy_score(y_true=batch_span_labels, y_pred=batch_res_label_ids),
                       })

        # 5. cache the annotation result
        cache_result = {
            "y_true": y_true,  # the ground truth label ids
            "labels": res_labels,
            "label_ids": res_label_ids
        }
        cache_result = Dataset.from_dict(cache_result)
        cache_result.save_to_disk(this_cache_dir)
        return cache_result

    def annotate_by_all(self, dataset, eval=True, quality=True):
        """
        Annotate the dataset by all annotators.
        :param dataset: the dataset to be annotated
        :param eval: whether to evaluate the annotation results
        :param quality: whether to evaluate the quality of the annotations
        :return:
        """
        quality_data = []
        for anno_cfg in self.annotators_cfg:
            result = self.annotate_by_one(dataset, **anno_cfg)
            quality_data.append(result['label_ids'])
            if eval:
                self.evaluate(result['y_true'], result['label_ids'], anno_cfg['name'])
        if quality:
            quality_data = np.array(quality_data)  # quality_data with shape (num_annotators, num_instances)
            # quality_data with shape (num_instances, num_annotators)
            # transpose the quality_data to get the shape (num_instances, num_annotators)
            eval_anno_quality(quality_data.T)

    def evaluate(self, y_true, y_pred, annotator_name: str):
        """
        Evaluate the annotation results by an annotator.
        :param y_true:
        :param y_pred:
        :param annotator_name:
        :return:
        """
        # compute all evaluation metrics
        eval_results = {'f1-micro': f1_score(y_true=y_true, y_pred=y_pred, average='micro'),
                        'precision-micro': precision_score(y_true=y_true, y_pred=y_pred, average='micro'),
                        'recall-micro': recall_score(y_true=y_true, y_pred=y_pred, average='micro'),
                        'accuracy': accuracy_score(y_true=y_true, y_pred=y_pred)}
        if not os.path.exists(self.config['eval_dir']):
            os.makedirs(self.config['eval_dir'], exist_ok=True)
        res_file = os.path.join(self.config['eval_dir'], f'{annotator_name}_res.txt')
        with open(res_file, 'w') as f:
            for metric, res in eval_results.items():
                f.write(f'{metric}: {res}\n')

def main():
    config = get_config('config.yml')

    # 1. pre-process the data
    processor = Processor(config['data_cfg'])
    formated_dataset = processor.process()

    # 2. annotate the data by LLMs
    annotation = Annotation(config['anno_cfg'])
    annotation.annotate_by_all(formated_dataset, eval=True)

if __name__ == '__main__':
    main()