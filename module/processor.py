import os
import copy
import random
import numpy as np
import faiss
import jsonlines

from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Dataset
import module.func_util as fu
from .label import Label
from .simcse import SimCSE

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
        self.sim_model = None

    def _init_sim_model(self, sim_path='../model/princeton-nlp/sup-simcse-roberta-large'):
        """
        Initialize the SimCSE model for retrieval-based support set sampling.
        :return:
        """
        if self.sim_model is None:
            self.sim_model = SimCSE(sim_path)

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

    def support_set_sampling(self, dataset, k_shot=1, sample_split='train', seed=None):
        """
        Sample k-shot support set from the dataset split.
        The sampled support set contains at least K examples for each of the labels.
        Refer to in the Support Set Sampling Algorithm in the Appendix B (P12) of the paper https://arxiv.org/abs/2203.08985
        or in the Algorithm 1 in the A.2 (P14) of the paper https://arxiv.org/abs/2303.08559

        :param dataset: The dataset to be sampled.
        :param k_shot: The shot number of the support set.
        :param sample_split: The dataset split you want to sample from.
        :param seed: The base seed for random sampling. If None, a random seed will be used.
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
        self.logger.info(f"Sampling {k_shot}-shot support set from {sample_split} split with seed {seed}...")
        for label in label_nums.keys():
            self.logger.info(f'Sampling {label} support set...')
            while counter[label] < k_shot:
                current_seed = seed if seed is not None else random.randint(0, 512)
                random.seed(current_seed)
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

    def retrival_support_set(self, dataset, k_shot, cache_dir, retrieval_base_size=-1, seed=None):
        """
        Retrieve top-k nearest neighbors for each test instance from the train dataset.

        :param dataset: The dataset containing 'train' and 'test' splits.
        :param k_shot: The number of nearest neighbors to retrieve.
        :param cache_dir: The directory to cache the computed embeddings.
        :param retrieval_base_size: The number of samples to randomly select from the train set for retrieval. If -1, use the entire train set.
        :param seed: The random seed for selecting the subset of the train set.
        :return: A list of indices representing the top-k nearest neighbors for each test instance.
        """

        # Define cache file paths
        train_cache_file = os.path.join(cache_dir, "train_embeddings.npy")
        test_cache_file = os.path.join(cache_dir, "test_embeddings.npy")
        join_character = ' ' if self.config['language'] == 'en' else ''

        # Load or compute train embeddings
        if os.path.exists(train_cache_file):
            train_embeddings = np.load(train_cache_file)
        else:
            train_sentences = [join_character.join(tokens) for tokens in dataset['train']['tokens']]
            train_embeddings = self.sim_model.encode(
                train_sentences,
                batch_size=256,
                normalize_to_unit=True,
                return_numpy=True
            )
            np.save(train_cache_file, train_embeddings)

        # If train_data_size is specified, randomly sample a subset of the train set
        if retrieval_base_size != -1:
            # if seed is not None:
            #     np.random.seed(seed)
            sample_indices = np.random.choice(train_embeddings.shape[0], retrieval_base_size, replace=False)
            train_embeddings = train_embeddings[sample_indices]
            train_subset = dataset['train'].select(sample_indices)
        else:
            train_subset = dataset['train']

        # Load or compute test embeddings
        if os.path.exists(test_cache_file):
            test_embeddings = np.load(test_cache_file)
        else:
            # self.config['split'] is the test or validation split
            test_sentences = [join_character.join(tokens) for tokens in dataset[self.config['split']]['tokens']]
            test_embeddings = self.sim_model.encode(
                test_sentences,
                batch_size=256,
                normalize_to_unit=True,
                return_numpy=True
            )
            np.save(test_cache_file, test_embeddings)

        # Build FAISS index for train embeddings
        # Build FAISS index for train embeddings (use GPU if available)
        train_embeddings = train_embeddings.astype('float32')
        test_embeddings = test_embeddings.astype('float32')
        cpu_index = faiss.IndexFlatIP(train_embeddings.shape[1])
        cpu_index.add(train_embeddings)

        try:
            # 初始化 GPU 资源并将 CPU 索引拷贝到 GPU（device 0）
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            index = gpu_index
            self.logger.info("Using FAISS GPU index on device 0")
        except Exception as e:
            index = cpu_index
            self.logger.warning(f"FAISS GPU not available, using CPU index: {e}")

        # Retrieve top-k nearest neighbors for each test instance
        self.logger.info(f"Retrieving {k_shot} nearest neighbors for test instances...")
        _, knn_indices = index.search(test_embeddings, k_shot)  # 乘4，保证demonstration的数量级一致
        self.logger.info("Retrieval completed.")

        # Map knn_indices back to the original dataset if size is specified
        # meanwhile, we convert numpy.int64 to int
        if retrieval_base_size != -1:
            knn_indices = [[int(sample_indices[int(idx)]) for idx in knn] for knn in knn_indices]
        else:
            knn_indices = [[int(idx) for idx in knn] for knn in knn_indices]
        return knn_indices


    def process(self, method = 'lsp', **kwargs):
        """
        Process the dataset.
        :param method: the method tot be evaluated. 'lsp' for label subset partition. 'retrieval' for retrieval-based method.
        :param kwargs: 用于'retrieval'方法的其他参数，例如
            1) retrieval_base_size: 检索的训练集大小。如果为-1，表示从整个训练集中检索支持集。
        :return:
        """
        assert method in ('lsp', 'retrieval', 'other')

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

        # the directory to cache the support set
        # 不同的方法使用不同的support set
        # 对于lsp方法，support set直接从train split中采样，采样得到的support set被放到一个jsonl文件中，测试集所有样本共享同一个support set
        # 对于retrival方法，利用向量相似度为测试集每个样本检索support set，jsonl文件中存储每个测试样本对应的support set的下标
        method_name = method
        if method == 'retrieval':
            retrieval_base_size = kwargs.get('retrieval_base_size', -1)
            if retrieval_base_size == -1:
                method_name = method + "_full"
            else:
                method_name = method + f"_{retrieval_base_size}"
        ss_cache_dir = os.path.join(self.config['ss_cache_dir'], f'span_{self.natural_flag}', method_name)

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
        support_set_info = None
        if self.config['support_set']:
            if not os.path.exists(ss_cache_dir):
                os.makedirs(ss_cache_dir)

            if method == 'retrieval':
                self._init_sim_model()  # initialize the SimCSE model for retrieval-based support set sampling

            for k_shot in self.config['k_shot']:
                for seed in self.config['seed']:
                    cache_ss_file_name = '{}_support_set_{}_shot_{}.jsonl'.format(self.config['sample_split'], k_shot, seed)
                    cache_counter_file_name = '{}_counter_{}_shot_{}.txt'.format(self.config['sample_split'], k_shot, seed)
                    if method == 'retrieval':
                        cache_ss_file_name = '{}_support_set_{}_shot.jsonl'.format(self.config['sample_split'], k_shot)
                        cache_counter_file_name = '{}_counters_{}_shot.txt'.format(self.config['sample_split'], k_shot)
                    support_set_file = os.path.join(ss_cache_dir, cache_ss_file_name)
                    counter_file = os.path.join(ss_cache_dir, cache_counter_file_name)

                    # check and load the cache
                    if not os.path.exists(support_set_file):
                    # 3.2 sample support set from scratch
                        self.logger.info(f'{support_set_file} does not exist, start to sample the support set...')
                        if method == 'retrieval':
                            support_sets = self.retrival_support_set(
                                preprocessed_dataset,
                                k_shot,
                                ss_cache_dir,
                                retrieval_base_size,
                                seed,
                            )  # support set for each test instance

                            # cace the support set
                            dir_path = os.path.dirname(support_set_file)
                            if dir_path and not os.path.exists(dir_path):
                                os.makedirs(dir_path)
                            with jsonlines.open(support_set_file, mode='w') as writer:

                                for support_set in support_sets:
                                    all_tokens, all_tags, all_spans_labels = [], [], []
                                    for idx in support_set:
                                        tokens = preprocessed_dataset[self.config['sample_split']]['tokens'][idx]
                                        tags = preprocessed_dataset[self.config['sample_split']]['tags'][idx]
                                        spans_labels =  preprocessed_dataset[self.config['sample_split']]['spans_labels'][idx]
                                        all_tokens.append(tokens)
                                        all_tags.append(tags)
                                        all_spans_labels.append(spans_labels)
                                    ids = [int(i) for i in support_set]
                                    writer.write({'ids':ids, 'tokens': all_tokens, 'tags': all_tags, 'spans_labels': all_spans_labels})
                        else:
                            support_set, counter = self.support_set_sampling(
                                preprocessed_dataset,
                                k_shot,
                                self.config['sample_split'],
                                seed,
                            )
                            # cache the support set
                            with jsonlines.open(support_set_file, mode='w') as writer:
                                for idx in support_set:
                                    tokens = preprocessed_dataset[self.config['sample_split']]['tokens'][idx]
                                    tags = preprocessed_dataset[self.config['sample_split']]['tags'][idx]
                                    spans_labels = preprocessed_dataset[self.config['sample_split']]['spans_labels'][idx]
                                    writer.write({'id': int(idx), 'tokens': tokens, 'tags': tags, 'spans_labels': spans_labels})

                            # cache the counter
                            with open(counter_file, 'w') as writer:
                                for k, v in counter.items():
                                    writer.write(f'{k}: {v}\n')

            support_set_info = {
                'dir': ss_cache_dir,
                'base_size': kwargs.get('size', -1)
            }

        # 4. shuffle, split and then save the formatted dataset
        # 4.1 check the cached result
        if self.config['continue']:
            try:
                dataset = load_from_disk(continue_dir)
                return dataset, support_set_info
            except FileNotFoundError:
                dataset = None

        # 4.2 get the specific split of the formatted dataset
        if self.config['split'] is not None:
            dataset = preprocessed_dataset[self.config['split']]

        # 4.3 shuffle the formatted dataset
        if self.config['shuffle']:
            dataset = dataset.shuffle()

        dataset.save_to_disk(continue_dir)

        # support_set informationz

        return dataset, support_set_info
