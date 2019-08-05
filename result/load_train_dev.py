dev_data_path = r"./data/diffText2.txt"
# dev_data_path = r"./data/dev_con_final.txt"
# dev_data_path = r"./data/dev_con_final1.txt"
# dev_data_path = r"./data/test_data1.txt"
# dev_data_path = r"./data/test_data2.txt"
train_data_path = r"./data/train_con_final.txt"


def get_train_dev(train_path=train_data_path, dev_path=dev_data_path):
    train_data_list = []
    with open(train_path) as train_file:
        line = train_file.readline()
        while line:
            train_data = dict()
            lines = line.split("\t")
            train_data["seq1"] = lines[1].split(" ")
            train_data["seq2"] = lines[2].split(" ")
            train_data["tag"] = lines[3]
            train_data_list.append(train_data)
            line = train_file.readline()
            pass
        pass
    dev_data_list = []
    with open(dev_path) as dev_file:
        line = dev_file.readline()
        while line:
            dev_data = dict()
            lines = line.split("\t")
            dev_data["seq1"] = lines[1].split(" ")
            dev_data["seq2"] = lines[2].split(" ")
            dev_data["tag"] = lines[3]
            dev_data_list.append(dev_data)
            line = dev_file.readline()
            pass
        pass
    return train_data_list, dev_data_list
    pass


def pad_sequences(sequences, pad_token='。'):
    """将没有对齐的句子进行对齐
    将一个batch中所有的句子进行对齐
    因为embedding里面没有PAD编码，所以默认编码就是"。"
    """
    max_length = max(map(lambda x: len(x), sequences))
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_token] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]
        pass
    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    """
    在data里面产生一个由(sentence0, sentence1, tag)组成的列表，数目为minibatch_size
    :param data:有{"seq1":sentence0, "seq2":sentence1, "tag":tag}组成的list
    :param minibatch_size:batch的大小
    :return x1_batch:由sequence0组成的list
    :return x2_batch:由sequence1组成的list
    :return y_batch:由tags组成的list
    """
    x1_batch, x2_batch, y_batch = [], [], []
    for d in data:
        if len(x1_batch) == minibatch_size:
            yield x1_batch, x2_batch, y_batch
            x1_batch, x2_batch, y_batch = [], [], []

        # if type(x[0]) == tuple:
        #     x = zip(*x)
        x1_batch += [d["seq1"]]
        x2_batch += [d["seq2"]]
        y_batch += [int(d["tag"])]

    if len(x1_batch) != 0:
        yield x1_batch, x2_batch, y_batch
