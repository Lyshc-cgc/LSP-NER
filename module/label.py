
class Label:
    """
    The Label class is used to store the labels, label ids.
    """
    def __init__(self, labels_cfg):
        self.labels = labels_cfg['labels']
        self.bio_labels = labels_cfg['bio']
        self.label2id = dict()
        self.id2label = dict()
        self.covert_tag2id = dict()  # covert the original BIO label (tag) id to the new label id. e.g., {2:2,3:2} means 2 (B-DATE) and 3 (I-DATE) are the same (DATE, id=2).
        if labels_cfg['natural']:
            self.init_natural_labels()
        else:
            self.init_bio_labels()

    def init_bio_labels(self):
        """
        init label2id, id2label, covert_tag2id form bio-format labels.
        All labels are simplified format. e.g., person -> PER, location -> LOC, organization -> ORG.
        :return:
        """
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

    def init_natural_labels(self):
        """
        init label2id, id2label, covert_tag2id form natural labels.
        ALL labels are natural format. e.g., person, location, organization.
        :return:
        """
        self.label2id['O'] = 0
        self.id2label[0] = 'O'
        for index, (k, v) in enumerate(self.labels.items()):
            label = v['natural']
            id = index + 1
            self.label2id[label] = id
            self.id2label[id] = label

        self.covert_tag2id[0] = 0  # 'O' -> 'O'
        for k, v in self.bio_labels.items():
            if k.startswith('B-') or k.startswith('I-'):
                label = k.replace('B-', '').replace('I-', '')
            else:
                label = k
            if label not in self.labels.keys():  # skip 'O'
                continue
            natural_label = self.labels[label]['natural']
            self.covert_tag2id[v] = self.label2id[natural_label]