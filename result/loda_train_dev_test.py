from load_train_dev import get_train_dev

# 首先读取dev
dev_data_path = r"./data/dev_con_final.txt"
train_data_path = r"./data/train_con_final.txt"

with open(dev_data_path) as dev_file:
    line = dev_file.readline()
    print(line)
    lines = line.split("\t")
    for l in lines:
        print(l)

train_data, dev_data = get_train_dev()
print(len(train_data))
print(len(dev_data))
print(train_data[0])

max_tag = -1
min_tag = 100
for t_data in train_data:
    if int(t_data["tag"]) < min_tag:
        min_tag = int(t_data["tag"])
    if int(t_data["tag"]) > max_tag:
        max_tag = int(t_data["tag"])
    pass
print(min_tag, max_tag)
