import copy
import json
import re
import os
import wandb
import torch
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
        self.config = fu.get_config(anno_cfg)
        self.annotators_cfg = self.config['annotators']
        self.annotator_ids = dict()
        for idx, anno_cfg in enumerate(self.annotators_cfg):
            self.annotator_ids[anno_cfg['name']] = idx

    def _init_chat_message(self, anno_cfg) -> list[None | dict[str, str]]:
        """
        Init the chat messages for the annotation models.

        :param anno_cfg: The parameters of the annotation model.
        :return:
        """
        if anno_cfg['chat']:
            # for Qwen and Yi-ft, we need to input the system's prompt and the user's prompt
            # https://huggingface.co/Qwen/Qwen1.5-72B-Chat-GPTQ-Int8
            # https://huggingface.co/TheBloke/nontoxic-bagel-34b-v0.2-GPTQ
            sys_prompt = self.config['prompt_template_chat'].format(system_role=self.config['system_role'],
                                                                    task_prompt=self.config['task_prompt'],
                                                                    types_prompt=self.config['types_prompt'])
            chat_message = [
                {"role": "system", "content": sys_prompt},
            ]
            for example in self.config['examples']:
                instance = self.config['instance_template'].format(sentence=example['sentence'], output='')
                chat_message.append({"role": "user", "content": instance})
                chat_message.append({"role": "assistant", "content": example['output']})
            query = self.config['instance_template'].format(sentence='{sentence}', output='')
            chat_message.append({"role": "user", "content": query})
        else:
            examples = ''
            for idx, example in enumerate(self.config['examples']):
                instance = self.config['instance_template'].format(sentence=example['sentence'], output=example['output'])
                examples += f'{idx + 1})\n{instance}'
            query = self.config['instance_template'].format(sentence='{sentence}', output='')
            usr_prompt = self.config['prompt_template'].format(system_role=self.config['system_role'],
                                                               task_prompt=self.config['task_prompt'],
                                                               types_prompt=self.config['types_prompt'],
                                                               examples=examples,
                                                               query=query)
            # e.g., for mistral and Yi, we only need to input the user's prompt
            chat_message = [{"role": "user", "content": usr_prompt}]
        return chat_message

    @staticmethod
    def _generate_chat_messages(instances, chat_message_template):
        """
        Generate chat messages for each instance. Meanwhile, init the labels for each instance.

        :param instances: The instances to be annotated.
        :param chat_message_template: The chat message template for the annotating model.
        :return:
        """
        for sent_id, (tokens, spans, spans_labels) in enumerate(zip(instances['tokens'], instances['spans'], instances['spans_labels'])):
            for (start, end, entity_mention), label in zip(spans, spans_labels):
                start, end = int(start), int(end)
                sentence = ' '.join(tokens[:start] + ['[ ', entity_mention, ' ]'] + tokens[end:])

                chat_message = copy.deepcopy(chat_message_template)
                query = chat_message[-1]['content'].format(sentence=sentence)
                chat_message[-1] = {"role": "user", "content": query}
                # if self.config['gold_span'] is True, label is the ground truth label id
                # yield the ID of the sentences, the mention span, the label id of the entity mention and the chat message
                # else, label is the tuple of the ground truth span and its label, like (start, end, gold_mention_span, gold_label)
                # yield the ID of the sentences, the mention span, gold span and the chat message
                span = (str(start), str(end), entity_mention)
                yield sent_id, span, label, chat_message

    @staticmethod
    def _process_output_text(output_text):
        """
        process the output text to get the label of the entity mention.
        :param output_text: the text output by the annotator model
        :return:
        """
        json_pattern = [r'\{\{(.*?)\}\}', r'\{(.*?)\}']  # the pattern to extract JSON string
        output_text = output_text.strip().replace('\_', '_')
        out_label = 'O'  # we assign 'O' to label to this span if we cannot extract the JSON string

        # extract JSON string from output.outputs[0].text
        for pattern in json_pattern:
            result = re.search(pattern, output_text, re.DOTALL)  # only extract the first JSON string
            if result:
                try:
                    out_label = json.loads(result.group())['answer'].strip()
                    break
                except json.JSONDecodeError:
                    continue
        return out_label

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
        annotate_flag = True  # whether to annotate the dataset from scratch
        annotator_name = anno_cfg['name']
        if self.config['gold_span']:
            this_cache_dir = os.path.join(self.config['cache_dir'], 'gold_span', annotator_name)
        else:
            this_cache_dir = os.path.join(self.config['cache_dir'], 'span', annotator_name)

        try:
            cache_result = load_from_disk(this_cache_dir)
            queue.put(cache_result)
            annotate_flag = False
        except FileNotFoundError:
            os.makedirs(this_cache_dir, exist_ok=True)

        # annotation process
        if annotate_flag:
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
            chat_message_template = self._init_chat_message(anno_cfg)

            # 3. batch process
            # yield sent_id, span, span_label, chat_message
            pbar = tqdm(fu.batched(self._generate_chat_messages(dataset,chat_message_template
                                                             ),anno_cfg['anno_bs']),desc=f'annotating by {annotator_name}')

            res_labels, res_label_ids = [], []  # store the output labels and label ids

            # if self.config['gold_span'] is True, we use gold span from annotation
            # y_true stores the ground truth label ids
            # else, we get the span from scratch by spaCy and stanza parsers
            # y_true stores the gold span and its label in a tuple
            y_true = []
            pred_spans = [] # store the predicted spans and its label

            for batch in pbar:  # batch is a tuple like ((sent_id_0, span_0, label_0, chat_0),(sent_id_1, span_1, label_1, chat_1)...)
                batch_sent_ids, batch_spans, batch_labels, batch_chats = [], [], [], []  # store the sent_id, span, the gold label ids / the gold spans, chat for each batch
                batch_res_label_ids = []  # store the output label ids for each batch to evaluate
                batch_pred_spans = []  # store the predicted spans and its label for each batch
                for sent_id, span, label, chat in batch:
                    # if self.config['gold_span'] is True, label is the ground truth label id
                    # else, label is the tuple of the ground truth span and its label, like (start, end, gold_mention_span, gold_label)
                    batch_sent_ids.append(sent_id)
                    batch_spans.append(span)
                    batch_labels.append(label)
                    y_true.append(label)
                    batch_chats.append(chat)

                # we should use tokenizer.apply_chat_template to add generation template to the chats explicitly
                templated_batch_chats = anno_tokenizer.apply_chat_template(batch_chats, add_generation_prompt=True, tokenize=False)
                outputs = anno_model.generate(templated_batch_chats, sampling_params)  # annotate
                # for test
                # test_answer = []
                # for output in outputs:
                #     test_answer.append({'prompt': output.prompt, 'output': output.outputs[0].text})
                for sent_id, span, output in zip(batch_sent_ids, batch_spans, outputs):
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
                    batch_pred_spans.append((*span, str(out_label_id)))
                    pred_spans.append((*span, str(out_label_id)))

                # evaluate batch results
                if self.config['gold_span']:
                    wandb.log({'f1-macro': f1_score(y_true=batch_labels, y_pred=batch_res_label_ids, average='macro', zero_division=0),
                               'precision-weighted': precision_score(y_true=batch_labels, y_pred=batch_res_label_ids, average='weighted', zero_division=0),
                               'recall-weighted': recall_score(y_true=batch_labels, y_pred=batch_res_label_ids, average='weighted', zero_division=0),
                               'accuracy & f1-micro': accuracy_score(y_true=batch_labels, y_pred=batch_res_label_ids),
                               'matthews_corrcoef': matthews_corrcoef(y_true=batch_labels, y_pred=batch_res_label_ids),
                               'cohen_kappa_score': cohen_kappa_score(y1=batch_labels, y2=batch_res_label_ids)
                               })
                else:
                    batch_res = fu.compute_span_f1(batch_pred_spans, batch_labels)
                    wandb.log({'precision': batch_res['precision'],
                               'recall': batch_res['recall'],
                               'f1-micro': batch_res['f1']})

            ray.shutdown()

            # 5. cache and evaluate the annotation result
            if self.config['gold_span']:
                cache_result = {
                    "y_true": y_true,  # the ground truth label ids
                    "labels": res_labels,
                    "label_ids": res_label_ids
                }

            else:
                cache_result = {
                    "y_true": y_true,  # the gold span and its label
                    "pred_spans": pred_spans
                }

            cache_result = Dataset.from_dict(cache_result)
            queue.put(cache_result)
            cache_result.save_to_disk(this_cache_dir)

        # 6. evaluation
        if self.config['gold_span']:
            self.evaluate(cache_result['y_true'], cache_result['label_ids'], annotator_name)
        else:
            self.evaluate(cache_result['y_true'], cache_result['pred_spans'], annotator_name)
        return cache_result

    def annotate_by_all(self, dataset, quality=True):
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
            if self.config['gold_span']:
                quality_data.append(result['label_ids'])
            else:
                quality_data.append(result['pred_spans'])

        # 2. evaluate the annotation quality
        # todo, for the case that gold_span is False, we cannot evaluate directly the quality of the annotation. the results need to be transformed to the value format
        if quality:
            if self.config['gold_span']:
                qual_res_file = os.path.join(self.config['eval_dir'], 'gold_span', 'quality_res.txt')
            else:
                qual_res_file = os.path.join(self.config['eval_dir'], 'span', 'quality_res.txt')
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
        :param y_true:
        :param y_pred:
        :param annotator_name:
        :return:
        """
        # compute all evaluation metrics
        if self.config['gold_span']:
            eval_results = {'f1-macro': f1_score(y_true=y_true, y_pred=y_pred, average='macro'),
                            'precision-weighted': precision_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0),
                            'recall-weighted': recall_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0),
                            'accuracy & f1-micro': accuracy_score(y_true=y_true, y_pred=y_pred),
                            'matthews_corrcoef': matthews_corrcoef(y_true=y_true, y_pred=y_pred),
                            'cohen_kappa_score': cohen_kappa_score(y1=y_true, y2=y_pred)}
            res_cache_dir = os.path.join(self.config['eval_dir'], 'gold_span')
        else:
            eval_results = fu.compute_span_f1(y_true, y_pred)
            res_cache_dir = os.path.join(self.config['eval_dir'], 'span')

        if not os.path.exists(res_cache_dir):
            os.makedirs(res_cache_dir, exist_ok=True)
        res_file = os.path.join(res_cache_dir, f'{annotator_name}_res.txt')
        with open(res_file, 'w') as f:
            for metric, res in eval_results.items():
                f.write(f'{metric}: {res}\n')