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