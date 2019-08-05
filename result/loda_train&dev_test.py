# 首先读取dev
dev_data_path = r"./data/dev_con_final.txt"
train_data_path = r"./data/train_con_final.txt"

with open(dev_data_path) as dev_file:
    line = dev_file.readline()
    print(line)
    lines = line.split("\t")
    for l in lines:
        print(l)

