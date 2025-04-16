import os
import copy
import random
import math
import jsonlines

from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Dataset
import module.func_util as fu
from module.label import Label

class Processor(Label):
    """
    The Processor class is used to process the data.
    """
    def __init__(self, data_cfg, labels_cfg, natural_form=False):
        """
        Initialize the Processor class.
        :param data_cfg: the data processing config from the config file.
        :param labels_cfg: the configuration of the label_cfgs.
        :param natural_form: whether the labels are in natural language form.
        """
        super().__init__(labels_cfg, natural_form)
        self.config = data_cfg
        self.natural_flag = 'natural' if natural_form else 'bio'  # use natural-form or bio-form
        self.logger = fu.get_sync_logger()  # use sync logger for data processing

    def _get_span_and_tags(self, tokens, tags, language='en'):
        """
        Get the span and span tags of the sentence, given the tokens and token tags.
        :param tokens: tokens of the sentence
        :param tags: tags for each token
        :param language: the language of the tokens.
        :return:
        """
        assert language in ('en', 'zh')
        instance_spans = []  # store spans for each instance
        instance_spans_labels = []  # store labels for each span of each instance
        idx = 0
        span = []  # store tokens in a span
        pre_tag = 0  # the previous tag

        # the start/end index for a span
        # if language is English, the start/end index is the token index
        # if language is Chinese, the start/end index is the character index
        start, end = 0, 0
        characters_length = 0  # the length of the characters that we have processed
        join_character = ' ' if self.config['language'] == 'en' else ''
        span_tag = None
        while idx < len(tokens):
            tag = tags[idx]
            if tag != 0:
                if pre_tag != 0 and self.covert_tag2id[tag] == self.covert_tag2id[pre_tag]:  # the token is in the same span
                    # append the token into the same span
                    span.append(tokens[idx])
                    span_tag = self.covert_tag2id[tag]
                    if language == 'en':
                        end = idx + 1  # exclusive
                    else:
                        # for Chinese, we need to consider the characters length
                        end = characters_length + len(tokens[idx])  # exclusive

                else:  # the previous is a 'O' token or previous token is not in the same span
                    # store the previous span
                    if len(span) > 0:
                        instance_spans.append((str(start), str(end), join_character.join(span)))
                        instance_spans_labels.append(span_tag)
                    # init a new span
                    span.clear()
                    span.append(tokens[idx])
                    span_tag = self.covert_tag2id[tag]
                    if language == 'en':
                        start = idx
                        end = idx + 1  # exclusive
                    else:
                        # for Chinese, we need to consider the characters length
                        start = characters_length
                        end = characters_length + len(tokens[idx])  # exclusive

            pre_tag = tag
            characters_length += len(tokens[idx])
            idx += 1

        # store the last span
        if len(span) > 0:
            instance_spans.append((str(start), str(end), join_character.join(span)))
            instance_spans_labels.append(span_tag)
        return instance_spans, instance_spans_labels

    def data_format_span(self, instances):
        """
        Get the span from gold annotated instances. an instance is corresponding to a sentence

        :param instances: Dict[str, List], A batch of instances. an instance is a dict with keys 'tokens', 'ner_tags',
        'starts', 'ends'
        :return:
        """

        # init the result
        res_tokens = []  # store the tokens of the instances
        res_tags = []  # store the tags of the instances
        res_spans_labels = []  # store the gold spans and labels of the instances

        # main process
        tokens_filed, ner_tags_field = self.config['tokens_field'], self.config['ner_tags_field']
        all_raw_tokens, all_raw_tags = instances[tokens_filed], instances[ner_tags_field]

        if not self.config['nested']:  # flat ner
            pbar = zip(all_raw_tokens, all_raw_tags)
        else:  # nested
            start_position, end_position, spans = instances['starts'], instances['ends'], instances['spans']
            pbar = zip(all_raw_tokens, start_position, end_position, spans, all_raw_tags)
        for instance in pbar:
            if not self.config['nested']:  # flat NER
                raw_tokens, raw_tags = instance
                if len(raw_tokens) != len(raw_tags):
                    # for those flat datasets, we need to filter out those instances with different length of tokens and tags
                    continue
                # 2.1.2 get gold spans and its labels
                gold_spans, gold_spans_tags = self._get_span_and_tags(raw_tokens, raw_tags, self.config['language'])

                # element in gold_spans is in the shape of (str(start), str(end) (excluded), span)
                # element in gold_spans_tags is tag id
                # the elements' shape of res_spans_labels is like [(start, end (excluded), gold_mention_span, span_label_id)...]

                res_tokens.append(raw_tokens)
                res_tags.append(raw_tags)
                res_spans_labels.append([(*gs, str(gst)) for gs, gst in zip(gold_spans, gold_spans_tags)])

            else:  # nested NER
                raw_tokens, starts, ends, spans, raw_tags = instance
                # 2.1.1 (optional) get the tag directly from the raw dataset
                gold_spans = []  # store gold spans for this instance
                for start, end, span, label in zip(starts, ends, spans, raw_tags):
                    label_id = None
                    try:
                        label_id = int(label)  # if label is a number
                    except ValueError:
                        label_id = self.covert_tag2id[label]  # if label is a string
                    # end position is excluded
                    gold_spans.append((str(start), str(end), span, str(label_id)))
                # the elements' shape of res_spans_labels is like [(start, end (excluded), gold_mention_span, span_label_id)...]
                res_spans_labels.append(gold_spans)
                res_tokens.append(raw_tokens)
                res_tags.append([])


        return {
            'tokens': res_tokens,
            'tags': res_tags,
            'spans_labels': res_spans_labels,  # store the gold spans and labels of the instances, shape like (start, end (excluded), gold_mention_span, gold_label_id)
        }

    def data_format_span_from_docs(self, documents):
        """
        Get the span from gold annotated documents. We should convert each sentence of a document to a single instance.

        :param documents: Dict[str, List], A batch of documents.
        :return:
        """

        # init the result
        res_tokens = []  # store the tokens of the instances
        res_tags = []  # store the tags of the instances
        res_spans_labels = []  # store the gold spans and labels of the instances

        # main process
        tokens_filed, ner_tags_field = self.config['tokens_field'], self.config['ner_tags_field']
        for doc in documents['sentences']:
            for instance in doc:
                raw_tokens, raw_tags = instance[tokens_filed], instance[ner_tags_field]
                # flat NER
                if len(raw_tokens) != len(raw_tags):
                    # for those flat datasets, we need to filter out those instances with different length of tokens and tags
                    continue

                # get gold spans and its labels
                gold_spans, gold_spans_tags = self._get_span_and_tags(raw_tokens, raw_tags, self.config['language'])

                # element in gold_spans is in the shape of (str(start), str(end) (excluded), span)
                # element in gold_spans_tags is tag id
                # the elements' shape of res_spans_labels is like [(start, end (excluded), gold_mention_span, span_label_id)...]

                res_tokens.append(raw_tokens)
                res_tags.append(raw_tags)
                res_spans_labels.append([(*gs, str(gst)) for gs, gst in zip(gold_spans, gold_spans_tags)])

        return {
            'tokens': res_tokens,
            'tags': res_tags,
            'spans_labels': res_spans_labels,
            # store the gold spans and labels of the instances, shape like (start, end (excluded), gold_mention_span, span_label_id)
        }

    def statistics(self, dataset, include_none=False):
        """
        Get the statistics of the dataset.
        :param dataset: the dataset to be analyzed.
        :param include_none: whether to include the instances without any golden entity spans. True means to include.
        :return: the statistics of the dataset.
        """
        # get the statistics of the dataset
        # check the cached
        # 1.1 get the entity number of each label

        label_nums = {k: 0 for k in self.label2id.keys() if k != 'O'}  # store the number of entities for each label
        label_indices = {k: [] for k in self.label2id.keys() if k != 'O' }  # store the index of instances for each label

        if include_none:
            label_nums['none'], label_indices['none'] = 0, []  # store the number and index of instances without any golden entity spans

        for instance in dataset:
            if include_none and len(instance['spans_labels']) == 0:
                label_nums['none'] += 1
                label_indices['none'].append(instance['id'])
                continue

            for spans_label in instance['spans_labels']:
                # shape like (start, end, gold_mention_span, gold_label_id)
                label_id = int(spans_label[-1])
                label = self.id2label[label_id]
                label_nums[label] += 1
                label_indices[label].append(instance['id'])

        # remove dunplicate indices
        for k, v in label_indices.items():
            label_indices[k] = list(set(v))

        sum_labels = sum(label_nums.values())
        label_dist = {k: v / sum_labels for k, v in label_nums.items()}

        return {
            'label_nums': label_nums,
            'label_dist': label_dist,
            'label_indices': label_indices
        }

    def support_set_sampling(self, dataset, k_shot=1, sample_split='train'):
        """
        Sample k-shot support set from the dataset split.
        The sampled support set contains at least K examples for each of the labels.
        Refer to in the Support Set Sampling Algorithm in the Appendix B (P12) of the paper https://arxiv.org/abs/2203.08985
        or in the Algorithm 1 in the A.2 (P14) of the paper https://arxiv.org/abs/2303.08559

        :param dataset: The dataset to be sampled.
        :param k_shot: The shot number of the support set.
        :param sample_split: The dataset split you want to sample from.
        :return: the support set containing k-shot index of examples for each of the labels.
        """
        def _update_counter(support_set, raw_counter):
            """
            Update the number for each label in the support set.
            :param support_set: the support_set
            :param raw_counter: the counter to record the number of entities for each label in the support set
            :return:
            """
            counter = {label: 0 for label in raw_counter.keys()}
            for idx in support_set:
                for spans_label in dataset['spans_labels'][idx]:
                    # spans_label shapes like (start, end, gold_mention_span, gold_label)
                    label_id = int(spans_label[-1])
                    label = self.id2label[label_id]
                    counter[label] += 1
            return counter

        # 1. init
        if sample_split not in dataset.keys():
            dataset = dataset['train']
        else:
            dataset = dataset[sample_split]

        label_nums = self.statistics(dataset)['label_nums']  # count the number of entities for each label
        label_nums = dict(sorted(label_nums.items(), key=lambda x: x[1], reverse=False))  # sort the labels by the number of entities by ascending order

        # add new_tags column
        # we extract span tags for each instance
        # an elements' shape of example['spans_labels'] is
        # [(str(start), str(end) (excluded), str(gold_mention_span), str(span_label_id)...]
        dataset = dataset.map(lambda example: {"span_tags": [int(e[-1]) for e in example['spans_labels']]})
        support_set = set()  # the support set
        counter = {label: 0 for label in label_nums.keys()}  # counter to record the number of entities for each label in the support set

        # init the candidate instances indices for each label
        candidate_idx = dict()

        for label in label_nums.keys():

            # filter out the instances without any golden entity spans
            label_id = self.label2id[label]
            tmp_ins = dataset.filter(lambda x: label_id in x['span_tags'] and len(x['spans_labels']) > 0)['id']
            candidate_idx.update({label: tmp_ins})

        # 2. sample
        self.logger.info(f"Sampling {k_shot}-shot support set from {sample_split} split...")
        for label in label_nums.keys():
            self.logger.info(f'Sampling {label} support set...')
            while counter[label] < k_shot:
                idxs = random.sample(candidate_idx[label], k=k_shot-counter[label])
                support_set.update(idxs)
                counter = _update_counter(support_set, counter)
                self.logger.info(f'support set statistic: {counter}')

        # 3. remove redundant instance
        raw_support_set = copy.deepcopy(support_set)
        for idx in tqdm(raw_support_set, desc='removing redundant instance'):
            tmp_support_set = copy.deepcopy(support_set)  # cache before removing instance idx
            support_set.remove(idx)
            counter = _update_counter(support_set, counter)
            # if we remove instance idx, leading to the number of entities for any label in the support set is less than k_shot
            # we should add instance idx back to the support set
            if len(list(filter(lambda e: e[1] < k_shot, counter.items()))) != 0:
                support_set = tmp_support_set

        counter = _update_counter(support_set, counter)
        return support_set, counter

    def subset_sampling(self, dataset: Dataset, size=200, sampling_strategy='random', seed=None):
        """
        Get the subset of the dataset according to sampling sampling_strategy.
        :param dataset: the dataset to be sampled to get subset.
        :param size: the size of the test subset.
        :param sampling_strategy: the sampling strategy.
            1) 'random' for random sampling. Select instances randomly. Each instance has the same probability of being selected.
            2) 'lab_uniform' for uniform sampling at label-level. Choice probability is uniform for each label.
            3) 'proportion' for proportion sampling. Choice probability is proportional to the number of entities for each label.
            4) 'shot_sample' for sampling test set like k-shot sampling. Each label has at least k instances.
        :param seed: the seed for random sampling. If None, a random seed will be used.
        :return:
        """
        assert sampling_strategy in ('random', 'lab_uniform', 'proportion', 'shot_sample')

        if sampling_strategy == 'random':
            if not seed or isinstance(seed, str):
                seed = random.randint(0, 512)
            self.logger.info(f"Random sampling with seed {seed}...")
            # https://huggingface.co/docs/datasets/process#shuffle
            # use Dataset.flatten_indices() to rewrite the entire dataset on your disk again to remove the indices mapping
            dataset_subset = dataset.shuffle(seed=seed).flatten_indices().select(range(size))

        elif sampling_strategy == 'proportion':
            statistics_res = self.statistics(dataset)
            label_dist,  label_indices= statistics_res['label_dist'], statistics_res['label_indices']
            choice_indices = []
            for label, proportion in label_dist.items():
                choice_num = math.ceil(proportion * size)
                choice_indices += random.sample(label_indices[label], choice_num)

            choice_indices = list(set(choice_indices))
            dataset_subset = dataset.select(choice_indices)

        elif sampling_strategy == 'lab_uniform':
            label_num = len(self.label2id.keys()) - 1  # exclude 'O' label
            statistics_res = self.statistics(dataset)
            label_indices = statistics_res['label_indices']
            choice_indices = []
            for label, indices in label_indices.items():
                choice_num = math.ceil(size / label_num)
                choice_indices += random.sample(indices, choice_num)
            choice_indices = list(set(choice_indices))
            dataset_subset = dataset.select(choice_indices)

        elif sampling_strategy == 'shot_sample':
            support_set, counter = self.support_set_sampling(dataset, k_shot=20, sample_split='train')
            dataset_subset = dataset.select(list(support_set))

        return dataset_subset

    def process(self):
        # 0. init config
        self.config['preprocessed_dir'] = self.config['preprocessed_dir'].format(dataset_name=self.config['dataset_name'])
        self.config['continue_dir'] = self.config['continue_dir'].format(dataset_name=self.config['dataset_name'])
        self.config['ss_cache_dir'] = self.config['ss_cache_dir'].format(dataset_name=self.config['dataset_name'])

        preprocessed_dir = os.path.join(self.config['preprocessed_dir'], f'span_{self.natural_flag}')
        process_func = self.data_format_span
        if self.config['dataset_name'] == 'ontonotes5_zh':
            process_func = self.data_format_span_from_docs

        # with_rank is used to determine whether to assign a value to the rank parameter in the map function
        continue_dir = os.path.join(self.config['continue_dir'], f'span_{self.natural_flag}')  # the directory to store the continued data to be annotated
        ss_cache_dir = os.path.join(self.config['ss_cache_dir'], f'span_{self.natural_flag}')  # the directory to cache the support set

        # 1. check and load the cached formatted dataset
        try:
            self.logger.info('Try to load the preprocessed dataset from the cache...')
            preprocessed_dataset = load_from_disk(preprocessed_dir)
        except FileNotFoundError:
            # 2. format datasets
            self.logger.info('No cache found, start to preprocess the dataset...')
            data_path = self.config['data_path'].format(dataset_name=self.config['dataset_name'])
            # raw dataset
            raw_dataset = load_dataset(
                data_path,
                name=self.config['cfg_name'],
                num_proc=self.config['num_proc'],
                trust_remote_code=True
            )

            preprocessed_dataset = raw_dataset.map(
                process_func,
                batched=True,
                batch_size=self.config['batch_size'],
                num_proc=self.config['num_proc'],
                remove_columns=raw_dataset['train'].column_names,
            )
            # add index column
            preprocessed_dataset = preprocessed_dataset.map(lambda example, index: {"id": index}, with_indices=True)  # add index column

            os.makedirs(self.config['preprocessed_dir'], exist_ok=True)
            preprocessed_dataset.save_to_disk(preprocessed_dir)

        # 3. sample the support set
        if self.config['support_set']:
            if not os.path.exists(ss_cache_dir):
                os.makedirs(ss_cache_dir)

            for k_shot in self.config['k_shot']:
                cache_ss_file_name = '{}_support_set_{}_shot.jsonl'.format(self.config['sample_split'], k_shot)
                cache_counter_file_name = '{}_counter_{}_shot.txt'.format(self.config['sample_split'], k_shot)
                support_set_file = os.path.join(ss_cache_dir, cache_ss_file_name)
                counter_file = os.path.join(ss_cache_dir, cache_counter_file_name)

                # check and load the cache
                if not os.path.exists(support_set_file):
                # 3.2 sample support set from scratch

                    support_set, counter = self.support_set_sampling(preprocessed_dataset, k_shot, self.config['sample_split'])
                    # cache the support set
                    with jsonlines.open(support_set_file, mode='w') as writer:
                        for idx in support_set:
                            tokens = preprocessed_dataset[self.config['sample_split']]['tokens'][idx]
                            tags = preprocessed_dataset[self.config['sample_split']]['tags'][idx]
                            spans_labels = preprocessed_dataset[self.config['sample_split']]['spans_labels'][idx]
                            writer.write({'id': idx, 'tokens': tokens, 'tags': tags, 'spans_labels': spans_labels})

                    # cache the counter
                    with open(counter_file, 'w') as writer:
                        for k, v in counter.items():
                            writer.write(f'{k}: {v}\n')

        # 4. shuffle, split and then save the formatted dataset
        # 4.1 check the cached result
        if self.config['continue']:
            try:
                dataset = load_from_disk(continue_dir)
                return dataset
            except FileNotFoundError:
                dataset = None

        # 4.2 get the specific split of the formatted dataset
        if self.config['split'] is not None:
            dataset = preprocessed_dataset[self.config['split']]

        # 4.3 shuffle the formatted dataset
        if self.config['shuffle']:
            dataset = dataset.shuffle()

        dataset.save_to_disk(continue_dir)

        return dataset
