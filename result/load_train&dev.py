dev_data_path = r"./data/dev_con_final.txt"
train_data_path = r"./data/train_con_final.txt"


def get_train_dev(train_path=train_data_path, dev_path=dev_data_path):
    train_data = {}
    with open(train_path) as train_file:
        line = train_file.readline()
        while line:
            lines = line.split("\t")
            train_data["seq1"] = lines[1].split(" ")
            train_data["seq2"] = lines[2].split(" ")
            train_data["tag"] = lines[3]
            line = train_file.readline()
            pass
        pass
    dev_data = {}
    with open(dev_path) as dev_file:
        line = dev_file.readline()
        while line:
            lines = line.split("\t")
            dev_data["seq1"] = lines[1].split(" ")
            dev_data["seq2"] = lines[2].split(" ")
            dev_data["tag"] = lines[3]
            line = dev_file.readline()
            pass
        pass
    return train_data, dev_data
    pass
