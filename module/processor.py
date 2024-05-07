import re
import os
import multiprocess
import module.func_util as fu
from datasets import load_dataset, load_from_disk
from module.label import Label

class Processor(Label):
    """
    The Processor class is used to process the data.
    """
    def __init__(self, data_cfg):
        super().__init__()
        self.config = fu.get_config(data_cfg)
        self.num_proc = self.config['num_proc']

    @staticmethod
    def _modify_spacy_tokenizer(nlp):
        """
        Used in the '_data_format_span' method.
        Modify the spaCy tokenizer to prevent it from splitting on '-' and '/'.
        Refer to https://spacy.io/usage/linguistic-features#native-tokenizer-additions

        :param nlp: The spaCy model.
        :return: The modified spaCy model.
        """
        from spacy.util import compile_infix_regex
        from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
        from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
        infixes = (
                LIST_ELLIPSES
                + LIST_ICONS
                + [
                    r"(?<=[0-9])[+\\-\\*^](?=[0-9-])",
                    r"(?<=[{al}{q}])\\.(?=[{au}{q}])".format(
                        al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
                    ),
                    # Commented out regex that splits on hyphens between letters:
                    # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
                    r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
                    # r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
                    r"(?<=[{a}])[:<>=/](?=[{a}])".format(a=ALPHA),
                    # Modified regex to only split words on '/' if it is preceded by a character
                ]
        )
        infix_re = compile_infix_regex(infixes)
        nlp.tokenizer.infix_finditer = infix_re.finditer
        return nlp

    def _get_span_and_tags(self, tokens, tags):
        """
        Get the span and span tags of the sentence, given the tokens and token tags.
        :param tokens: tokens of the sentence
        :param tags: tags for each token
        :return:
        """
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
            pre_tag = tag
            idx += 1
        # store the last span
        if len(span) > 0:
            instance_spans.append((str(start), str(end), ' '.join(span)))
            instance_spans_labels.append(self.covert_tag2id[tags[start]])
        return instance_spans, instance_spans_labels

    @staticmethod
    def _convert_ch_position(sent, target_spans):
        """
        convert the start/end character position of the span to the start/end position of the span.
        For instance, a 'I am a student.' is tokenized into ['I', 'am', 'a', 'student', '.'].
        The start/end character position of the span 'student' is 7/14, and the start/end position of the span is 3/4.
        :param sent: original sentence
        :param target_spans: A list of target span
        :return:
        """
        res_spans = []
        for start_ch_pos, end_ch_pos, span in target_spans:
            # stanza will replace some special characters with special tokens, e.g., '(' -> '-LRB-', '[' -> '-LSB-'
            # We need to replace them back to the original characters.
            # But we need to use escape character in the regular expression.
            # span = self._replace_special_tokens(span)

            if start_ch_pos == -1 and end_ch_pos == -1:  # -1 means the start/end character index is not available
                # Find the start character index and end character index of the first matched NP/NE span.
                re_span = re.escape(span)  # escape special characters in the span
                pattern = r"\b" + re_span + r"\b"  # match the whole word
                matches = re.finditer(pattern, sent)
                for match in matches:
                    start_ch_idx, end_ch_idx = match.start(), match.end()
                    # To get the start position of the first word of the matched NP span,
                    # we just need to count the number of spaces before the start character
                    start = sent[:start_ch_idx].count(' ')

                    # To get the end position of the last word of the matched NP span,
                    # we just need to count the number of spaces before the end character
                    end = sent[:end_ch_idx].count(' ') + 1  # end position of the NP span, excluded
                    res_spans.append((start, end, span))

        return res_spans

    @staticmethod
    def _eval_span_quality(dataset, split=None):
        """
        Evaluate the quality of the spans recognized by stanza and spaCy parsers.
        :param dataset: list, the dataset containing 4 fields ('tokens', 'tags' 'spans' and 'spans_labels'), where 'spans'
            are the spans recognized by the parsers. 'spans_labels' are the gold spans and their labels.
        :param split: str, the split name of the dataset.
        """
        if split is not None:
            split = [split]
        else:
            split = ['train', 'validation', 'test']

        for sp in split:
            dataset_split = dataset[sp]
            spans, spans_labels = dataset_split['spans'], dataset_split['spans_labels']
            spa_cons_string, sta_cons_string = dataset_split['spa_cons_string'], dataset_split['sta_cons_string']

            true_positive, false_positive, false_negative = 0, 0, 0
            for idx, (instance_spans, instance_spans_labels, sp_cs, st_cs) in enumerate(zip(spans, spans_labels, spa_cons_string, sta_cons_string)):
                # 1. get the gold spans and their labels
                gold_spans = [(start, end, span) for start, end, span, label in instance_spans_labels]

                # 2. get the predicted spans and their labels
                pred_spans = [(start, end, span) for start, end, span in instance_spans]

                # 3. compute the span F1
                for span_item in pred_spans:
                    if span_item in gold_spans:
                        true_positive += 1
                        gold_spans.remove(span_item)
                    else:
                        false_positive += 1

                # these entities are not predicted.
                if len(gold_spans) > 0:
                    print('-'*20)
                    print(f'remaining gold_spans: {gold_spans}')
                    print(f'spacy constituency tree: {sp_cs}')
                    print(f'stanza constituency tree: {st_cs}')
                false_negative += len(gold_spans)

            recall = true_positive / (true_positive + false_negative)
            precision = true_positive / (true_positive + false_positive)
            if recall + precision == 0:
                f1 = 0
            else:
                f1 = precision * recall * 2 / (recall + precision)

        return {
        'true_positive': true_positive,
        'false_positive': false_positive,
        'false_negative': false_negative,
        'recall': recall,
        'precision': precision,
        'f1': f1
    }

    def data_format_gold(self, instances):
        """
        Format the data in the required format.
        From 'BIO' tag into span format.
        original format https://huggingface.co/datasets/tner/ontonotes5
        We get gold spans directly from the annotations (ground truth).
        :param instances: Dict[str, List], A batch of instances.
        :return:
        """
        res_spans = []  # store the gold spans of the instances
        res_spans_labels = []  # store the label ids of the gold spans
        for inst_id, (tokens, tags) in enumerate(zip(instances['tokens'], instances['tags'])):
            instance_spans, instance_spans_labels = self._get_span_and_tags(tokens, tags)
            res_spans.append(instance_spans)
            res_spans_labels.append(instance_spans_labels)
        return {
            'tokens': instances['tokens'],
            'tags': instances['tags'],
            'spans': res_spans,  # the gold spans of the instances
            'spans_labels': res_spans_labels  # the label ids of the gold spans
        }

    def data_format_span(self, instances, rank=0):
        """
        Using stanza and spacy to get the span from scratch. Do not use gold annotated spans.
        :param instances: Dict[str, List], A batch of instances.
        :param rank: The rank of the current process. It will be automatically assigned a value when multiprocess is
            enabled in the map function.
        :return:
        """
        import stanza
        import spacy
        import benepar
        import torch
        from nltk.tree import Tree
        from spacy.training import Alignment

        # 0. some settings
        # 0.1 GPU settings
        if self.config['cuda_devices'] == 'all':
            # set the GPU can be used by stanza in this process
            os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % torch.cuda.device_count())

            # specify the GPU to be used by spaCy, which should be same as above
            # https://spacy.io/api/top-level#spacy.prefer_gpu
            spacy.prefer_gpu(rank % torch.cuda.device_count())
        else:
            cuda_devices = str(self.config['cuda_devices']).split(',')
            gpu_id = rank % len(cuda_devices)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices[gpu_id])

            # specify the GPU to be used by spaCy, which should be same as above
            spacy.prefer_gpu(int(cuda_devices[gpu_id]))

        # 0.2 spaCy setting
        # load a spaCy and a stanza model in each process
        spacy_nlp = spacy.load(name=self.config['spacy_model']['name'])
        spacy_nlp = self._modify_spacy_tokenizer(spacy_nlp)  # modify the spaCy tokenizer

        # add a benepar constituency parsing to spaCy pipeline
        # refer to https://spacy.io/universe/project/self-attentive-parser
        # and https://github.com/nikitakit/self-attentive-parser
        spacy_nlp.add_pipe('benepar', config={'model': 'benepar_en3_large'})

        # 0.3 stanza setting
        stanza_nlp = stanza.Pipeline(**self.config['stanza_model'], download_method=None)

        # 0.4 init the result
        res_spans = []  # store the spans of the instances, predicted by the spaCy and stanza parsers
        res_spans_labels = []  # store the gold spans and labels of the instances
        res_spa_cons_string = []  # store the constituency parse tree of the instances, predicted by the spaCy parser
        res_sta_cons_string = []  # store the constituency parse tree of the instances, predicted by the stanza parser

        # main process
        all_raw_tokens, all_raw_tags = instances['tokens'], instances['tags']
        # 1. Some preparations

        # 1.2. covert tokens to sentence
        sents = [' '.join(raw_tokens) for raw_tokens in all_raw_tokens]

        # 2. process by 2 parsers
        # refer to
        # 1) https://spacy.io/usage/processing-pipelines#processing
        # 2) https://spacy.io/api/language#pipe
        spacy_docs = list(spacy_nlp.pipe(sents))  # covert generator to list

        # refer to https://stanfordnlp.github.io/stanza/getting_started.html#processing-multiple-documents
        stanza_docs = stanza_nlp.bulk_process(sents)

        for sent, raw_tokens, raw_tags, spa_doc, sta_doc in zip(sents, all_raw_tokens, all_raw_tags, spacy_docs, stanza_docs):
            # 2.1 spaCy
            # 2.1.1 get tag and token alignment between sentence tokenized by spaCy and raw sentence
            # see details at https://spacy.io/usage/linguistic-features#aligning-tokenization
            spacy_tokens = [token.text for token in spa_doc]
            align = Alignment.from_strings(raw_tokens, spacy_tokens)
            aligned_tags = []  # the tags assigned to tokens output by spaCy
            sp_tokens_idx = -1
            for length in align.y2x.lengths:
                # the token tag of spacy_tokens at position sp_tokens_idx is aligned to the token tag of raw_tokens at position raw_tokens_idx
                sp_tokens_idx += length
                raw_tokens_idx = align.y2x.data[sp_tokens_idx]  # the map from spacy_tokens to raw_tokens is stored in align.y2x.data
                tag = raw_tags[raw_tokens_idx]
                aligned_tags.append(tag)  # covert original tags to ids of new tags

            # 2.1.2 get gold spans and its labels
            gold_spans, gold_spans_tags = self._get_span_and_tags(spacy_tokens, aligned_tags)
            res_spans_labels.append([(*gs, str(gst)) for gs, gst in zip(gold_spans, gold_spans_tags)])

            # 2.1.3 get NP chunk by spaCy. They are flat
            # store the start word index, end word index (excluded) and the text of the NP spans.
            # i.e., (start_word_idx, end_word_idx, span_text)
            # The method is included in the spaCy package.
            # refer to https://spacy.io/usage/linguistic-features#noun-chunks
            # and https://spacy.io/api/doc#noun_chunks
            spacy_result = [(chunk.start, chunk.end, chunk.text) for chunk in spa_doc.noun_chunks]

            # 2.1.4 get spans by spaCy constituency parsing
            # get constituency parse tree (String) of the sentence
            # refer to https://github.com/nikitakit/self-attentive-parser
            spa_cons_string = list(spa_doc.sents)[0]._.parse_string
            res_spa_cons_string.append(spa_cons_string)

            # Convert string to nltk.tree.Tree
            # refer to https://www.nltk.org/api/nltk.tree.html#nltk.tree.Tree.fromstring
            spa_cons_tree = Tree.fromstring(spa_cons_string)


            # filter out all the NP\CD\ subtrees
            # We can use a filter function to restrict the Tree.subtrees we want,
            # refer to https://www.nltk.org/api/nltk.tree.html#nltk.tree.Tree.subtrees
            spa_subtrees = [subtree.leaves() for subtree in spa_cons_tree.subtrees(lambda t: t.label() in self.config['cand_constituent'])]

            # init the spacy spans from Np subtrees
            # We initiate the start character index and end character index with -1.
            spa_subtrees_spans = [(-1, -1, ' '.join(subtree)) for subtree in spa_subtrees]
            spacy_result += self._convert_ch_position(sent, spa_subtrees_spans)

            # 2.2 stanza
            # 2.2.1 get spans by stanza
            # refer to https://stanfordnlp.github.io/stanza/constituency.html
            # Here, Constituency parser of Stanza will split compound words formed by hyphen (-) into several words
            # e.g., 'United-States' will be split into 'United', '-' and 'States'
            sta_cons_string = sta_doc.sentences[0].constituency  # constituency parse tree (String) of the sentence
            res_sta_cons_string.append(repr(sta_cons_string))
            sta_cons_tree = Tree.fromstring(repr(sta_cons_string))   # convert string to nltk.tree.Tree

            # filter out all the NP\CD subtrees
            # We can use a filter function to restrict the Tree.subtrees we want,
            sta_subtrees = [subtree.leaves() for subtree in sta_cons_tree.subtrees(lambda t: t.label() in self.config['cand_constituent'])]

            # get stanza spans
            stanza_spans = [(-1, -1, ' '.join(subtree)) for subtree in sta_subtrees]
            stanza_result = self._convert_ch_position(sent, stanza_spans)

            # 2.3. Select the union of two parsers' recognition results
            # convert start/end index to string, to be consistent with the format of spans. This operation ensures
            # that the tuple is successfully converted to pyarrow and then serialized into a JSON/JSONL array
            max_span_len = self.config['span_portion'] * len(sent)

            assert self.config['mode'] in ['strict', 'loose'], f"mode must be one of ('strict', loose)!"
            if self.config['mode'] == 'strict':
                # In strict (default) mode, we get spans based on intersection of spaCy and Stanza results.
                spans = [(str(start), str(end), text)
                         for start, end, text in list(set(spacy_result) & set(stanza_result))
                         if len(text) <= max_span_len  # filter out long span
                         ]
            else:
                # In loose mode, we get spans based on union of spaCy and Stanza results.
                spans = [(str(start), str(end), text)
                         for start, end, text in list(set(spacy_result) | set(stanza_result))
                         if len(text) <= max_span_len  # filter out long span
                         ]

            res_spans.append(spans)

        return {
            'tokens': instances['tokens'],
            'tags': instances['tags'],
            'spans': res_spans,  # the spans of the instances, predicted by the spaCy and stanza parsers, shape like (start, end, mention_span)
            'spans_labels': res_spans_labels,  # store the gold spans and labels of the instances, shape like (start, end, gold_mention_span, gold_label)
            'spa_cons_string': res_spa_cons_string,  # constituency parse tree of the instances, predicted by the spaCy parser
            'sta_cons_string': res_sta_cons_string  # constituency parse tree of the instances, predicted by the stanza parser
        }

    def process(self):
        # 0. init config
        if self.config['gold_span']:
            save_dir = os.path.join(self.config['save_dir'], 'gold_span')
            process_func = self.data_format_gold
            with_rank = False
            continue_dir = os.path.join(self.config['continue_dir'], 'gold_span')
        else:
            save_dir = os.path.join(self.config['save_dir'], 'span', self.config['mode'])
            process_func = self.data_format_span
            # with_rank is used to determine whether to assign a value to the rank parameter in the map function
            # we use rank number to specify the GPU device to be used by stanza and spaCy in the different processing
            with_rank = True
            # batch_size = self.config['batch_num_per_device'] * self.config['batch_size_per_device']
            continue_dir = os.path.join(self.config['continue_dir'], 'span', self.config['mode'])
            quality_res_dir = os.path.join(self.config['eval_dir'], 'span', self.config['mode'])
            if not os.path.exists(quality_res_dir):
                os.makedirs(quality_res_dir)

        # set 'spawn' start method in the main process to parallelize computation across several GPUs when using multi-processes in the map function
        # refer to https://huggingface.co/docs/datasets/process#map
        multiprocess.set_start_method('spawn')

        # 1. check and load the cached formatted dataset
        try:
            formated_dataset = load_from_disk(save_dir)
        except FileNotFoundError:
            # 2. format datasets to get span from scratch
            raw_dataset = load_dataset(self.config['data_path'], num_proc=self.config['num_proc'])
            formated_dataset = raw_dataset.map(process_func,
                                               batched=True,
                                               batch_size=self.config['batch_size'],
                                               num_proc=self.num_proc,
                                               with_rank=with_rank)

            os.makedirs(self.config['save_dir'], exist_ok=True)
            formated_dataset.save_to_disk(save_dir)

        # 3. if the self.config['gold_span'] is False, we get span from scratch by stanza and spaCy
        # we need to evaluate spans quality the formatted dataset
        if self.config['eval_quality']:
            quality_res = self._eval_span_quality(formated_dataset)
            quality_res_file = os.path.join(quality_res_dir, 'quality_res.txt')
            with open(quality_res_file, 'w') as f:
                for metric, res in quality_res.items():
                    f.write(f'{metric}: {res}\n')
            print(f"Span quality: {quality_res}")

        # 4. select, shuffle, split and then save the formatted dataset
        # 4.1 check the cached result
        if self.config['continue']:
            try:
                dataset = load_from_disk(continue_dir)
                return dataset
            except FileNotFoundError:
                dataset = None

        # 4.2 get the specific split of the formatted dataset
        if self.config['split'] is not None:
            dataset = formated_dataset[self.config['split']]

        # 4.3 shuffle the formatted dataset
        if self.config['shuffle']:
            dataset = dataset.shuffle()

        # 4.4 select the specific number of instances
        if self.config['select']:
            dataset = dataset.select(range(self.config['select']))
        dataset.save_to_disk(continue_dir)
        return dataset
