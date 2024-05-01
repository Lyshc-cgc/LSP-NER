import re
import os

import multiprocess
from datasets import load_dataset, load_from_disk
from module.func_util import get_config
from module.label import Label

class Processor(Label):
    """
    The Processor class is used to process the data.
    """
    def __init__(self, data_cfg):
        super().__init__()
        self.config = get_config(data_cfg)
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

    @staticmethod
    def _merge_compound_words(sents: list[list[str]]) -> list[str]:
        """
        Used in the '_data_format_span' method.
        1. Some compound words formed by hyphen ('-') had been split into several words, we need to merge them into a
            single word.
            e.g., ['United', '-', 'States'] -> ['United-States'], ['the', 'Semi-', 'finals'] -> ['the Semi-finals']
        2. In stanza, the fraction symbol will be separated by spaces.  We need to merge them together.
            e.g., ['The', 'FMS', '/', 'FMF', 'case'] -> ['The', 'FMS/FMF', 'case'],
                ['3', '/', '4ths', 'to', '9' , '/', '10ths', 'of', 'the', 'tenant', 'farmers', 'on', 'some', 'estates']
                -> ['3/4ths', 'to', '9/10ths', 'of', 'the, 'tenant', 'farmers, 'on', 'some', 'estates']

        :param sents: List[List[str]], a list of sentences, where each sentence is a list of words/tokens.
        :return: new_sents, List[str], a list of new sentences
        """
        new_sents = []
        for sent in sents:
            pos = 0  # word position
            while 0 <= pos < len(sent):
                word = sent[pos]
                if word == '-' or (word == '/' and pos >= 1 and sent[pos - 1].isdigit()):
                    # e.g., ['a', 'United', '-', 'States', 'b'] -> ['a', 'United-States', 'b']
                    # ['3', '/', '4ths'] -> ['3/4ths']
                    if pos - 1 >= 0 and pos + 2 < len(sent):
                        sent = sent[:pos - 1] + [''.join(sent[pos - 1: pos + 2])] + sent[pos + 2:]
                    elif pos - 1 >= 0 and pos + 2 >= len(sent):
                        sent = sent[:pos - 1] + [''.join(sent[pos - 1: pos + 2])]
                    else:  # pos - 1 < 0, i.e., pos == 0
                        pos += 2  # ignore this word
                    pos = pos - 1  # in this case, the position of the next new word is at the previous position
                elif not word.endswith('B-') and word != '--' and word.endswith(
                        '-'):  # e.g., ['a', 'the', 'Semi-', 'finals', 'b'] -> ['a', 'the Semi-finals', 'b']
                    # Special symbols (e.g., '-LRB-', '-LSB-') need to be excluded
                    if pos + 1 == len(sent):  # the last word
                        break
                    elif pos + 2 < len(sent):
                        sent = sent[:pos] + [''.join(sent[pos: pos + 2])] + sent[pos + 2:]
                    else:
                        sent = sent[:pos] + [''.join(sent[pos: pos + 2])]
                    # in this case, the position of the next new word is at the current pos, i.e., pos = pos
                elif not word.endswith('B-') and word != '--' and word.startswith(
                        '-'):  # e.g., ['a', 'the', 'Semi', '-finals', 'b'] -> ['a', 'the', 'Semi-finals', 'b']
                    if pos - 1 >= 0 and pos + 1 < len(sent):
                        sent = sent[:pos - 1] + [''.join(sent[pos - 1: pos + 1])] + sent[pos + 1:]
                    elif pos - 1 >= 0 and pos + 1 >= len(sent):
                        sent = sent[:pos - 1] + [''.join(sent[pos - 1: pos + 1])]
                    else:  # pos - 1 < 0, i.e., pos == 0
                        pos += 2  # ignore this word
                    pos = pos - 1  # in this case, the position of the next new word is at the previous position
                else:
                    pos += 1
            new_sents.append(' '.join(sent))
        return new_sents

    @staticmethod
    def _replace_special_tokens(sent: str) -> str:
        """
        Used in the '_data_format_span' method.
        Replace special tokens with original characters.
        e.g., '-LRB-' -> '(', '-RRB-' -> ')', '-LSB-' -> '[', '-RSB-' -> ']'

        :param sent: The sentence to be processed.
        :return: The processed sentences.
        """
        processed_sents = (sent.replace('-LRB-', '(')
                           .replace('-RRB-', ')')
                           .replace('-LSB-', '[')
                           .replace('-RSB-', ']')
                           .replace('-LCB-', '{')
                           .replace('-RCB-', '}')
                           )

        return processed_sents

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
            # the token is an 'O' token
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
                span = re.escape(span)  # escape special characters in the span
                matches = re.finditer(span, sent)
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
                aligned_tags.append(self.covert_tag2id[tag])  # covert original tags to ids of new tags

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

            # 2.1.4 get NP spans by spaCy constituency parsing
            # get constituency parse tree (String) of the sentence
            # refer to https://github.com/nikitakit/self-attentive-parser
            spa_cons_string = list(spa_doc.sents)[0]._.parse_string

            # Convert string to nltk.tree.Tree
            # refer to https://www.nltk.org/api/nltk.tree.html#nltk.tree.Tree.fromstring
            spa_cons_tree = Tree.fromstring(spa_cons_string)

            # filter out all the NP subtrees
            # We can use a filter function to restrict the Tree.subtrees we want,
            # refer to https://www.nltk.org/api/nltk.tree.html#nltk.tree.Tree.subtrees
            spa_subtrees = [subtree.leaves() for subtree in spa_cons_tree.subtrees(lambda t: t.label() == 'NP')]

            # init the spacy spans from Np subtrees
            # We initiate the start character index and end character index with -1.
            spa_subtrees_spans = [(-1, -1, ' '.join(subtree)) for subtree in spa_subtrees]
            spacy_result += self._convert_ch_position(sent, spa_subtrees_spans)

            # 2.2 stanza
            # 2.2.1 get NP spans by stanza
            # refer to https://stanfordnlp.github.io/stanza/constituency.html
            # Here, Constituency parser of Stanza will split compound words formed by hyphen (-) into several words
            # e.g., 'United-States' will be split into 'United', '-' and 'States'
            sta_cons_string = sta_doc.sentences[0].constituency  # constituency parse tree (String) of the sentence
            sta_cons_tree = Tree.fromstring(repr(sta_cons_string))   # convert string to nltk.tree.Tree

            # filter out all the NP subtrees
            # We can use a filter function to restrict the Tree.subtrees we want,
            sta_subtrees = [subtree.leaves() for subtree in sta_cons_tree.subtrees(lambda t: t.label() == 'NP')]

            # However, as mentioned before, the compound words formed by hyphen (-) will be split into several
            # words by stanza. So we need to combine them back into a single word.
            # e.g., ['United', '-', 'States'] -> ['United-States']
            sta_subtrees = self._merge_compound_words(sta_subtrees)

            # get stanza spans
            stanza_spans = [(-1, -1, ' '.join(subtree)) for subtree in sta_subtrees]
            stanza_result = self._convert_ch_position(sent, stanza_spans)

            # 2.3. Select the union of two parsers' recognition results (NP/NE spans)
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
            'spans_labels': res_spans_labels  # store the gold spans and labels of the instances, shape like (start, end, gold_mention_span, gold_label)
        }

    def process(self):
        # 0. init config
        if self.config['gold_span']:
            save_dir = os.path.join(self.config['save_dir'], 'gold_span')
            process_func = self.data_format_gold
            with_rank = False
            continue_dir = os.path.join(self.config['continue_dir'], 'gold_span')
        else:
            save_dir = os.path.join(self.config['save_dir'], 'span')
            process_func = self.data_format_span
            # with_rank is used to determine whether to assign a value to the rank parameter in the map function
            # we use rank number to specify the GPU device to be used by stanza and spaCy in the different processing
            with_rank = True
            # batch_size = self.config['batch_num_per_device'] * self.config['batch_size_per_device']
            continue_dir = os.path.join(self.config['continue_dir'], 'span')

        # set 'spawn' start method in the main process to parallelize computation across several GPUs when using multi-processes in the map function
        # refer to https://huggingface.co/docs/datasets/process#map
        multiprocess.set_start_method('spawn')

        # 1. check and load the cache file
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

        if self.config['continue']:
            try:
                dataset = load_from_disk(continue_dir)
                return dataset
            except FileNotFoundError:
                dataset = None

        if self.config['split'] is not None:
            dataset = formated_dataset[self.config['split']]

        if self.config['shuffle']:
            dataset = dataset.shuffle()

        if self.config['select']:
            dataset = dataset.select(range(self.config['select']))
        dataset.save_to_disk(continue_dir)
        return dataset