
class Label:
    """
    The Label class is used to store the labels, label ids.
    """
    def __init__(self, labels_cfg):
        self.labels = labels_cfg['labels']
        self.bio_labels = labels_cfg['bio']
        self.label2id = dict()
        self.id2label = dict()
        self.covert_tag2id = dict()  # covert the original label (tag) id to the new label id. e.g., {2:2,3:2} means 2 (B-DATE) and 3 (I-DATE) are the same (DATE, id=2).
        self.init_labels()

    def init_labels(self):
        idx = 0
        for k, v in self.bio_labels.items():
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