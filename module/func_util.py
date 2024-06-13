import re
import yaml
import itertools
import numpy as np
from scipy.stats import norm
from statsmodels.stats import inter_rater as irr
from yaml import SafeLoader
def get_config(cfg_file):
    """
    Get the configuration from the configuration file.

    :param cfg_file: str, the path to the configuration file. YAML format is used.
    :return: dict, the configuration.
    """
    with open(cfg_file, 'r') as f:
        config = yaml.load(f, Loader=SafeLoader)
    return config

def batched(iterable, n):
    """
    Yield successive n-sized batches from iterable. It's a generator function in itertools module of python 3.12.
    https://docs.python.org/3.12/library/itertools.html#itertools.batched
    However, it's not available in python 3.10. So, I have implemented it here.

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

def eval_anno_quality(data, metric='fleiss_kappa'):
    """
    Evaluate the quality of the annotations.

    :param data: array_like, 2-Dim data containing category assignment with subjects in rows and raters in columns.
    :param metric: The metric to evaluate the quality of the annotations.
    :return:
    """
    # For calculating fleiss_kappa, we refer to the following link:
    # https://support.minitab.com/zh-cn/minitab/help-and-how-to/quality-and-process-improvement/measurement-system-analysis/how-to/attribute-agreement-analysis/attribute-agreement-analysis/methods-and-formulas/kappa-statistics/#testing-significance-of-fleiss-kappa-unknown-standard

    if metric == 'fleiss_kappa':
        # https://www.statsmodels.org/stable/generated/statsmodels.stats.inter_rater.aggregate_raters.html
        data = irr.aggregate_raters(data)[0]  # returns a tuple (data, categories), we need only data

        # the code below is taken from the following link:
        # https://github.com/Lucienxhh/Fleiss-Kappa/blob/main/fleiss_kappa.py
        subjects, categories = data.shape
        n_rater = np.sum(data[0])

        p_j = np.sum(data, axis=0) / (n_rater * subjects)
        P_e_bar = np.sum(p_j ** 2)

        P_i = (np.sum(data ** 2, axis=1) - n_rater) / (n_rater * (n_rater - 1))
        P_bar = np.mean(P_i)

        K = (P_bar - P_e_bar) / (1 - P_e_bar)

        tmp = (1 - P_e_bar) ** 2
        var = 2 * (tmp - np.sum(p_j * (1 - p_j) * (1 - 2 * p_j))) / (tmp * subjects * n_rater * (n_rater - 1))

        SE = np.sqrt(var)  # standard error
        Z = K / SE
        p_value = 2 * (1 - norm.cdf(np.abs(Z)))

        ci_bound = 1.96 * SE / subjects
        lower_ci_bound = K - ci_bound
        upper_ci_bound = K + ci_bound

        return {
            'fleiss_kappa': K,
            'standard_error': SE,
            'z': Z,
            'p_value': p_value,
            'lower_0.95_ci_bound': lower_ci_bound,
            'upper_0.95_ci_bound': upper_ci_bound
        }

def compute_span_f1(gold_spans, pred_spans):
    """
    Compute the confusion matrix, span-level metrics such as precision, recall and F1-micro.
    :param gold_spans: the gold spans in a batch.
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

def find_span(text, span):
    """
    Find the span in the text.
    :param text: str, the text.
    :param span: the mention.
    :return: list, the list of spans.
    """
    res_spans = []
    # Find the start character index and end character index of the first matched span.
    re_span = re.escape(span)  # escape special characters in the span
    pattern_0 = r"\b(" + re_span + r")\b"  # match the whole span after escaping special characters
    pattern_1 = r"\s(" + re_span + r")\s"  # match the span surrounded by spaces after escaping special characters
    patterns = [pattern_0, pattern_1]
    res_matches = []
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        res_matches += [match for match in matches]

    for match in res_matches:
        start_ch_idx, end_ch_idx = match.span(1)  # get the capture group 1
        # To get the start position of the first word of the matched NP span,
        # we just need to count the number of spaces before the start character
        start = text[:start_ch_idx].count(' ')

        # To get the end position of the last word of the matched NP span,
        # we just need to count the number of spaces before the end character
        end = text[:end_ch_idx].count(' ') + 1  # end position of the NP span, excluded
        res_spans.append((start, end, span))

    return res_spans
