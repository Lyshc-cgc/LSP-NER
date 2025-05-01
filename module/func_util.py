import sys
import math
import random
import re
import yaml
import itertools
import logging
import contextvars

import pandas as pd
from collections import Counter
# from scipy.stats import norm
# from statsmodels.stats import inter_rater as irr
from aiologger import Logger
from aiologger.handlers.streams import AsyncStreamHandler
from aiologger.handlers.files import AsyncFileHandler
from aiologger.formatters.base import Formatter
from yaml import SafeLoader

# a context variable used to record coroutine id
coroutine_id_var = contextvars.ContextVar('coroutine_id', default=None)
_async_logger = None
_sync_logger = None

# formatter
class CoroutineIDFormatter(Formatter):
    def format(self, record):
        # Get the coroutine ID from the context variable
        coroutine_id = getattr(record, 'coroutine_id', 'N/A')
        # Add the coroutine ID to the log record
        record.coroutine_id = coroutine_id
        return super().format(record)

def get_async_logger(name='async', level=logging.INFO, log_file='test.log'):
    global _async_logger
    if _async_logger is None:
        logger: Logger = Logger(name=name, level=level)
        # file handler
        asy_file_handler = AsyncFileHandler(
            filename=log_file,
            encoding="utf8"
        )

        # stream handler
        asy_stream_handler = AsyncStreamHandler(stream=sys.stdout)

        # set format
        formatter = CoroutineIDFormatter(
            "[%(asctime)s| %(levelname)s| %(name)s| %(coroutine_id)s| %(filename)s:%(lineno)d] %(message)s"
        )
        asy_file_handler.formatter = formatter
        asy_stream_handler.formatter = formatter

        logger.add_handler(asy_file_handler)
        logger.add_handler(asy_stream_handler)
        _async_logger = logger
    return _async_logger

def get_sync_logger(name='sync', level=logging.INFO, filename='test.log'):
    global _sync_logger
    if _sync_logger is None:
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # remove redundant handler
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # file handler to store log
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(level)

        # console handler to print log
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # format
        formatter = logging.Formatter('[%(asctime)s| %(levelname)s| %(name)s| %(filename)s:%(lineno)d] %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        _sync_logger = logger
    return _sync_logger

def get_config(cfg_file):
    """
    Get the configuration from the configuration file.

    :param cfg_file: str, the path to the configuration file. YAML format is used.
    :return: dict, the configuration.
    """
    with open(cfg_file, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=SafeLoader)
    return config

def batched(iterable, n):
    """
    Yield successive n-sized batches from iterable. It's a generator function in itertools module of python 3.12.
    https://docs.python.org/3.12/library/itertools.html#itertools.batched
    However, it's not available in python 3.10. So, I have to implement it here.

    :param iterable:
    :param n:
    :return:
    """
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch

# def eval_anno_quality(data, metric='fleiss_kappa'):
#     """
#     Evaluate the quality of the annotations.
#
#     :param data: array_like, 2-Dim data containing category assignment with subjects in rows and raters in columns.
#     :param metric: The metric to evaluate the quality of the annotations.
#     :return:
#     """
#     # For calculating fleiss_kappa, we refer to the following link:
#     # https://support.minitab.com/zh-cn/minitab/help-and-how-to/quality-and-process-improvement/measurement-system-analysis/how-to/attribute-agreement-analysis/attribute-agreement-analysis/methods-and-formulas/kappa-statistics/#testing-significance-of-fleiss-kappa-unknown-standard
#
#     if metric == 'fleiss_kappa':
#         # https://www.statsmodels.org/stable/generated/statsmodels.stats.inter_rater.aggregate_raters.html
#         data = irr.aggregate_raters(data)[0]  # returns a tuple (data, categories), we need only data
#
#         # the code below is taken from the following link:
#         # https://github.com/Lucienxhh/Fleiss-Kappa/blob/main/fleiss_kappa.py
#         subjects, categories = data.shape
#         n_rater = np.sum(data[0])
#
#         p_j = np.sum(data, axis=0) / (n_rater * subjects)
#         P_e_bar = np.sum(p_j ** 2)
#
#         P_i = (np.sum(data ** 2, axis=1) - n_rater) / (n_rater * (n_rater - 1))
#         P_bar = np.mean(P_i)
#
#         K = (P_bar - P_e_bar) / (1 - P_e_bar)
#
#         tmp = (1 - P_e_bar) ** 2
#         var = 2 * (tmp - np.sum(p_j * (1 - p_j) * (1 - 2 * p_j))) / (tmp * subjects * n_rater * (n_rater - 1))
#
#         SE = np.sqrt(var)  # standard error
#         Z = K / SE
#         p_value = 2 * (1 - norm.cdf(np.abs(Z)))
#
#         ci_bound = 1.96 * SE / subjects
#         lower_ci_bound = K - ci_bound
#         upper_ci_bound = K + ci_bound
#
#         return {
#             'fleiss_kappa': K,
#             'standard_error': SE,
#             'z': Z,
#             'p_value': p_value,
#             'lower_0.95_ci_bound': lower_ci_bound,
#             'upper_0.95_ci_bound': upper_ci_bound
#         }

def compute_span_f1(gold_spans, pred_spans):
    """
    Compute the confusion matrix, span-level metrics such as precision, recall and F1-micro.
    :param gold_spans: the gold spans.
    :param pred_spans: the spans predicted by the model.
    :return:
    """
    true_positive, false_positive, false_negative = 0, 0, 0
    for span_item in pred_spans:
        if span_item in gold_spans:
            true_positive += 1
            gold_spans.remove(span_item)
        else:
            false_positive += 1

    # these entities are not predicted.
    false_negative += len(gold_spans)

    if true_positive + false_negative == 0:
        recall = 0
    else:
        recall = true_positive / (true_positive + false_negative)
    if true_positive + false_positive == 0:
        precision = 0
    else:
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

def compute_span_f1_by_labels(gold_spans, pred_spans, id2label, res_file):
    """
    Compute the confusion matrix, span-level metrics such as precision, recall and F1-micro for each label.
    :param gold_spans: the gold spans.
    :param pred_spans: the predicted spans.
    :param id2label: a map from label id to the label name.
    :param res_file: the file to save the results.
    :return:
    """
    # one record for one label
    # every record conclude all information we need
    # 1) Label, the label name
    # 2) Gold count, the number of gold spans for this label
    # 3) Gold rate, the proportion of gold spans to the total gold spans for this label
    # 4) Pred count, the number of predicted spans for this label
    # 5) Pred rate, the proportion of predicted spans to the total predicted spans for this label
    # 6) TP, the true positive for this label
    # 7) FP, the false positive for this label
    # 8) FN, the false negative for this label
    # 9) Pre, the precision for this label
    # 10) Rec, the recall for this label
    # 11) F1, the F1-micro for this label

    label_record = {}
    for lb in id2label.values():
        if lb == 'O':  # do not consider the 'O' label
            continue
        label_record[lb] = {"Label": lb, "Gold count": 0, "Gold rate": 0, "Pred count": 0, "Pred rate": 0,
                           "TP": 0, "FP": 0, "FN": 0, "P": 0, "R": 0, "F1": 0}

    for gold_span in gold_spans:
        label_id = int(gold_span[-1])  # shape like (start, end, span, label)
        label = id2label[label_id]
        label_record[label]["Gold count"] += 1

    ood_type_preds = []
    ood_mention_preds = []
    for pred_span in pred_spans:
        mention, label_id = pred_span[-2], pred_span[-1]  # shape like (start, end, mention span, label)
        label_id = int(label_id)  # shape like (start, end, span, label)
        label = id2label[label_id]
        # ood type
        if label not in id2label.values():
            ood_type_preds.append({label: mention})
            continue
        label_record[label]["Pred count"] += 1
        # ood mention,
        # if tmp_mention not in item["sentence"]:
        #     ood_mention_preds.append({tmp_mention: tmp_type})
        #     continue

        if pred_span in gold_spans:
            label_record[label]["TP"] += 1
            gold_spans.remove(pred_span)

    # the total metrics
    n_gold_tot = sum([x["Gold count"] for x in label_record.values()])
    n_pred_tot = sum([x["Pred count"] for x in label_record.values()])
    true_positive_total = sum([x["TP"] for x in label_record.values()])
    false_positive_total = n_pred_tot - true_positive_total
    false_negative_total = n_gold_tot - true_positive_total
    precision = true_positive_total / n_pred_tot if n_pred_tot else 0
    recall = true_positive_total / n_gold_tot if n_gold_tot else 0
    if precision and recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    precision_total = round(precision, 4) * 100
    recall_total = round(recall, 4) * 100
    f1_total = round(f1, 4) * 100

    # metrics for each label
    for l in label_record:
        gold_count = label_record[l]["Gold count"]
        pred_count = label_record[l]["Pred count"]
        true_positive = label_record[l]["TP"]
        false_positive = pred_count - true_positive
        false_negative = gold_count - true_positive

        gold_rate = gold_count / n_gold_tot if n_gold_tot else 0
        pred_rate = pred_count / n_pred_tot if n_pred_tot else 0
        gold_rate = round(gold_rate, 4) * 100
        pred_rate = round(pred_rate, 4) * 100

        pre = true_positive / pred_count if pred_count else 0
        rec = true_positive / gold_count if gold_count else 0
        if pre and rec:
            f1 = 2 * pre * rec / (pre + rec)
        else:
            f1 = 0
        pre = round(pre, 4) * 100
        rec = round(rec, 4) * 100
        f1 = round(f1, 4) * 100

        label_record[l]["Gold rate"] = gold_rate
        label_record[l]["Pred rate"] = pred_rate
        label_record[l]["TP"] = true_positive
        label_record[l]["FP"] = false_positive
        label_record[l]["FN"] = false_negative
        label_record[l]["P"] = pre
        label_record[l]["R"] = rec
        label_record[l]["F1"] = f1

    label_record["Total"] = {"Label": "ToTal", "Gold count": n_gold_tot, "Gold rate": 100, "Pred count": n_pred_tot,
                            "Pred rate": 100, "TP": true_positive_total, "FP": false_positive_total, "FN": false_negative_total,
                             "P": precision_total, "R": recall_total, "F1": f1_total}

    # convert to dataframe
    df_metrics = pd.DataFrame(list(label_record.values()))
    # cache the results
    df_metrics.to_csv(res_file, index=False)
    return df_metrics

def find_span(text: str, span: str, language: str = 'en'):
    """
    Find the span in the text.
    :param text: str, the text.
    :param span: str, the mention.
    :param language: str, the language of the text. Default is 'en'.
    :return: list, the list of spans.
    """
    if not span:
        return []
    res_spans = []
    # Find the start character index and end character index of the first matched span.
    re_span = re.escape(str(span))  # escape special characters in the span
    if language == 'en':
        pattern_0 = r"\b(" + re_span + r")\b"  # match the whole span after escaping special characters
        pattern_1 = r"\s(" + re_span + r")\s"  # match the span surrounded by spaces after escaping special characters
        patterns = [pattern_0, pattern_1]
    else:
        # for chinese, we directly match the span
        patterns = [re_span]
    res_matches = []
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        res_matches += [match for match in matches]

    for match in res_matches:
        if language == 'en':
            start_ch_idx, end_ch_idx = match.span(1)  # get the capture group 1
            # To get the start|end position of the first word of the matched NP span,
            # we just need to count the number of spaces before the start character
            start = text[:start_ch_idx].count(' ')
            end = text[:end_ch_idx].count(' ') + 1  # end position of the NP span, excluded
        else:
            start_ch_idx, end_ch_idx = match.span()  # get the capture group 0
            # for chinese, the start and end position are the same as the character index
            start, end = start_ch_idx, end_ch_idx

        res_spans.append((start, end, span))

    return res_spans

def get_label_subsets(labels, sub_size, repeat_num=1, fixed_subsets=None):
    """
    Get the subsets of the labels.
    :param labels: list, the list of labels.
    :param sub_size: int or float (<1), the size of the label subset.
    :param repeat_num: the number of times to repeat each label.
    :param fixed_subsets: a list of lists or tuples, the fixed subsets. we randomly sample the rest of the labels. e.g., [['PER', 'ORG'], ['LOC', 'GPE'],...]
    :return: list, the list of subsets.
    """
    if 0 < sub_size < 1:
        sub_size = math.floor(len(labels) * sub_size)
        if sub_size < 1:
            sub_size = 1

    label_subsets = []
    for _ in range(repeat_num):
        random.shuffle(labels)
        if fixed_subsets:
            labels = [l for l in labels if l not in fixed_subsets]  # filter out labels in the fixed subsets
            label_subsets += fixed_subsets
        label_subsets += list(batched(labels, sub_size))  # batch method return a generator
    return label_subsets

def get_label_mention_pairs(original_pairs, label_mention_map_portion, id2label):
    """
    Get the label-mention pairs, where the label-mention pairs are partially correct.
    :param original_pairs: the original label-mention pairs.
    :param label_mention_map_portion: the portion of the corrected label-mention pair. Default is 1, which means all the label-mention pairs are correct.
    :param id2label: a map from label id to the label name.
    :return:
    """
    random.shuffle(original_pairs)
    wrong_pairs_num = int(len(original_pairs) * (1 - label_mention_map_portion))
    tmp_wrong_pairs, correct_pairs = original_pairs[:wrong_pairs_num], original_pairs[wrong_pairs_num:]
    wrong_pairs = []
    for start, end, entity_mention, label_id in tmp_wrong_pairs:
        label_ids = list(id2label.keys())
        label_ids.remove(int(label_id))
        wrong_label_id = random.choice(label_ids)
        wrong_pairs.append((start, end, entity_mention, wrong_label_id))
    res_pairs = correct_pairs + wrong_pairs
    return res_pairs

def remove_duplicated_label_sets(label_sets: list):
    """
    Remove the duplicated label sets.
    :param label_sets: the label sets
    :return:
    """
    duplicated_id = []  # store the idx of the duplicated label sets
    for i in range(len(label_sets)):
        if i in duplicated_id:  # skip the idx that has been in the duplicated_id
            continue
        for j in range(i + 1, len(label_sets)):
            if j in duplicated_id:  # skip the idx that has been in the duplicated_id
                continue
            if Counter(label_sets[i]) == Counter(label_sets[j]):
                duplicated_id.append(j)
    label_sets = [label_sets[i] for i in range(len(label_sets)) if i not in duplicated_id]
    return label_sets

def compute_lspi(label_sets: list):
    """
    compute the label space per instance (LSPI). This metric is used to evaluate the diversity of the labels.
    :param label_sets: the label_sets to be evaluated in demonstrations. It is shaped like [label_set1, label_set2, ...].
        the label set i is shaped like [label1, label2,...]
    :return:
    """
    instances_num = len(label_sets)

    # remove the duplicated label sets
    label_counters = remove_duplicated_label_sets(label_sets)

    # compute the label space per instance (LSPI)
    label_space = len(label_counters)
    lspi = label_space/instances_num
    return lspi

def compute_label_coverage(label_sets: list, gold_label_sets: list):
    """
    Compute the label coverage (LC). This metric is used to evaluate the coverage of labels from the demonstrations over golden labels.
    :param label_sets: the label sets to be evaluated in demonstrations. It is shaped like [label_set1, label_set2, ...].
        the label set i is shaped like [label1, label2,...]
    :param gold_label_sets: the golden label sets from the test set. It is shaped like [gold_label_set1, gold_label_set2,...].
        the gold label set i is shaped like [gold_label1, gold_label2,...]
    :return:
    """
    # remove the duplicated label sets
    label_sets = remove_duplicated_label_sets(label_sets)
    gold_label_sets = remove_duplicated_label_sets(gold_label_sets)

    # compute the label cover (LC)
    cover_num = 0
    for label_set in label_sets:
        if label_set in gold_label_sets:
            cover_num += 1
    label_coverage = cover_num/len(gold_label_sets)
    union_label_sets = label_sets + gold_label_sets
    # label_coverage1 = cover_num/len(union_label_sets)
    return label_coverage


def write_metrics_to_excel(worksheet, start_row, res_file, anno_cfg):
    """
    write metrics to excel files

    :param worksheet: the worksheet to write the metrics.
    :param start_row: the row index we start. If the worksheet is new, start_row is 2. elif start_row > 2, it's the last row we add data.
    :param res_file, the file to save the results
    :param anno_cfg: the configuration of the annotation.
    :return:
    """

    with open(res_file, 'r') as f:
        eval_results = f.readlines()

    metric_num = len(eval_results)  # the number of metrics
    if start_row == 2:  # it's a new worksheet
        worksheet.write(0, 0, 'label_mention_map_portion')
        worksheet.write(0, 1, 'rep_num')
        worksheet.write(0, 2, '5-shot')
        worksheet.write(0, 3, 'subset_size')
        worksheet.write(0, metric_num + 4, '1-shot')
        if anno_cfg['k_shot'] == 5:  # 5-shot
            head_row, head_col = 1, 4  # headers start from (1, 4)
            data_row, data_col = 2, 4  # datas start from (2, 4)
        else:  # 1-shot
            head_row, head_col = 1, metric_num + 4  # headers start from (1, metric_num + 4)
            data_row, data_col = 2, metric_num + 4  # datas start from (2, metric_num + 4)
    elif start_row > 2:  # we continue to write data to the older worksheet
        if anno_cfg['k_shot'] == 5:  # 5-shot
            head_row, head_col = 1, 4  # headers start from (1, 4)
            data_row, data_col = start_row, 4  # datas start from (start_row, 4)
        else:  # 1-shot
            head_row, head_col = 1, metric_num + 4  # headers start from (1, metric_num + 4)
            data_row, data_col = start_row, metric_num + 4  # datas start from (start_row, metric_num + 4)

    if 'repeat_num' in anno_cfg.keys():
        rep_num = anno_cfg['repeat_num']
    elif 'demo_times':
        rep_num = anno_cfg['demo_times']
    worksheet.write(data_row, 0, anno_cfg['label_mention_map_portion'])
    worksheet.write(data_row, 1, rep_num)
    worksheet.write(data_row, 2, anno_cfg['annotator_name'])
    worksheet.write(data_row, 3, anno_cfg['subset_size'])
    for line in eval_results:
        line = line.strip()
        line = line.split(' ')
        metric, res = line[0], float(line[1])
        if start_row == 2:  # header
            worksheet.write(head_row, head_col, metric)
            head_col += 1
        worksheet.write(data_row, data_col, res)
        data_col += 1
    data_row += 1
    return  data_row