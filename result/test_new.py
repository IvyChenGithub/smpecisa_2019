from BiLSTM import SentenceRelationModel
from load_embedding import get_embedding_data
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
embedding = get_embedding_data()
model = SentenceRelationModel()
model.build()

test_data_path = r"./data/test_final4.txt"
result_path = r"./data/result.txt"

# if not os.path.exists(result_path):
#     print("创建result.txt")
#     os.makedirs(result_path)
#     pass

with open(test_data_path) as test_file:
    with open(result_path, "a+") as result_file:
        line = test_file.readline()
        while line:
            lines = line.split("\t")
            print(lines[0])
            if len(lines) >= 4:
                seq1 = lines[2].split(" ")
                seq2 = lines[3].split(" ")
                if len(seq1) != 0 and len(seq2) != 0:
                    logit = model.predict_batch([seq1], [seq2])
                    result = str(np.argmax(logit[0]))
                    line += "\t" + result
                else:
                    line += "\t0"
                    pass
            else:
                line += "\t1"
                pass
            line += "\n"
            result_file.write(line)
            line = test_file.readline()
        pass
    pass
