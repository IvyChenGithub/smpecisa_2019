# 这个脚本的作用是训练
from load_train_dev import get_train_dev
import random
from BiLSTM import SentenceRelationModel
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
train_ratio = 0.8  # 选取出train_ratio比例的数据作为训练数据
train_data, dev_data = get_train_dev()
print("有效训练数据有", len(train_data), "条")
# 接下来打乱rawdata
random.shuffle(train_data)
train = []
dev = []
train_num = int(len(train_data)*train_ratio)
train = train_data[:train_num]
dev = train_data[train_num:]
# 接下来是训练了
model = SentenceRelationModel()
model.build()
model.train(train, dev)
