
class Label:
    """
    The Label class is used to store the label_cfgs, label ids.
    """
    def __init__(self, labels_cfg, natural_form=False):
        """
        Initialize the Label class.
        :param labels_cfg: the label_cfgs from the config file.
        :param natural_form: whether the labels are in natural language form.
        """
        self.labels = labels_cfg['labels']
        self.raw_label2id = labels_cfg['raw_label2id']  # the raw label2id mapping from the raw dataset.
        self.raw_bio = labels_cfg['raw_bio']  # a flag to indicate whether the labels are in BIO format in the raw dataset.
        self.label2id = dict()
        self.id2label = dict()

        # covert the original BIO label (tag) id to the new label id. e.g., {2:2,3:2} means 2 (B-DATE) and 3 (I-DATE) are the same (DATE, id=2).
        # if the labels are not in BIO format in the raw datasets, there is no need to covert the original BIO label (tag) id to the new label id.
        # key-value are same with to the label2id and id2label.
        self.covert_tag2id = dict()
        if natural_form:  # use natural-language-form labels
            self.init_natural_labels()
        else:  # use simple-form labels
            self.init_simp_labels()

    def init_simp_labels(self):
        """
        init label2id, id2label, covert_tag2id from label_cfgs using simple-form.
        All label_cfgs are simplified format. e.g., person -> PER, location -> LOC, organization -> ORG.
        :return:
        """
        idx = 0
        for k, v in self.raw_label2id.items():
            if k.startswith('B-') or k.startswith('I-'):
                label = k.replace('B-', '').replace('I-', '')
            else:
                label = k
            if label not in self.label2id.keys():
                self.label2id[label] = idx
                self.id2label[idx] = label
                idx += 1

            # if the labels are in BIO format in the raw datasets, we need to covert the
            # original BIO label (tag) id to the new label id.
            # e.g., {2:2,3:2} means 2 (B-DATE) and 3 (I-DATE) are the same (DATE, id=2).
            self.covert_tag2id[v] = self.label2id[label]

    def init_natural_labels(self):
        """
        init label2id, id2label, covert_tag2id from label_cfgs using natural language form.
        ALL label_cfgs are natural format. e.g., person, location, organization.
        :return:
        """
        self.label2id['O'] = 0
        self.id2label[0] = 'O'
        for index, (k, v) in enumerate(self.labels.items()):
            label = v['natural']
            id = index + 1
            self.label2id[label] = id
            self.id2label[id] = label

        # if the labels are not in BIO format in the raw datasets, there is no need to covert the
        # original BIO label (tag) id to the new label id.
        if not self.raw_bio:
            return

        # if the labels are in BIO format in the raw datasets, covert the original BIO label (tag) id to the new label id.
        self.covert_tag2id[0] = 0  # 'O' -> 'O'
        for k, v in self.raw_label2id.items():
            if k.startswith('B-') or k.startswith('I-'):
                label = k.replace('B-', '').replace('I-', '')
            else:
                label = k
            if label not in self.labels.keys():  # skip 'O'
                continue
            natural_label = self.labels[label]['natural']
            self.covert_tag2id[v] = self.label2id[natural_label]