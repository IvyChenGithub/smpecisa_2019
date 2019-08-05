import tensorflow as tf
import numpy as np

from BiLSTM import SentenceRelationModel
from general_utils import Progbar
from load_embedding import get_embedding_data
from load_train_dev import get_train_dev, pad_sequences

batch_size = 50  # 每次训练的大小


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

def run_evaluate(self, test):
    """
    在test数据集上进行验证,输出准确率和召回率
    :param test:由(sequence, tags)组成的list
    :return:
    """
    nbatches = (len(test) + batch_size - 1) // batch_size  # 训练batch的次数
    prog = Progbar(target=nbatches)
    accs = []
    i = 0
    for words1, words2, labels in minibatches(test, batch_size):
        logits = self.predict_batch(words1, words2)
        for logit, label in zip(logits[0], labels):
            accs += [np.argmax(logit) == label]
            pass
        prog.update(i + 1, [("evaluate acc", 100*np.mean(accs))])
        i += 1
        pass
    acc = np.mean(accs)
    return {"acc": 100 * acc}

def load_model(data):
    embeddings = get_embedding_data()
    nbatches = (len(data) + batch_size - 1) // batch_size  # 训练batch的次数
    prog = Progbar(target=nbatches)
    accs = []
    i = 0
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('save/SentenceRelationModel1/SentenceRelationModel1_nl_3_hsl_196__.cpkt.meta')
        saver.restore(sess, tf.train.latest_checkpoint("save/SentenceRelationModel1/"))
        print(tf.get_collection('logit'))
        y = tf.get_collection('logit')
        graph = tf.get_default_graph()
        seq1_word_embeddings = graph.get_operation_by_name('seq1_word_embeddings').outputs[0]
        seq2_word_embeddings = graph.get_operation_by_name('seq2_word_embeddings').outputs[0]
        sequence1_lengths1 = graph.get_operation_by_name('sequence1_lengths').outputs[0]
        sequence2_lengths1 = graph.get_operation_by_name('sequence2_lengths').outputs[0]
        for words1, words2, labels in minibatches(data, batch_size):
            words1, sequence1_lengths = pad_sequences(words1)
            words2, sequence2_lengths = pad_sequences(words2)
            words1_embeddings = [[embeddings[w1] if w1 in embeddings.keys() else embeddings[","]
                                  for w1 in ws1] if len(ws1)>0 else [embeddings["。"]] for ws1 in words1]
            words2_embeddings = [[embeddings[w2] if w2 in embeddings.keys() else embeddings[","]
                                  for w2 in ws2] if len(ws2)>0 else [embeddings["。"]] for ws2 in words2]
            sess.run(y, feed_dict={seq1_word_embeddings: words1_embeddings,
                                   seq2_word_embeddings: words2_embeddings,
                                   sequence1_lengths: sequence1_lengths1,
                                   sequence2_lengths: sequence2_lengths1})
            for logit, label in zip(y[0], labels):
                accs += [np.argmax(logit) == label]
                pass
            prog.update(i + 1, [("evaluate acc", 100*np.mean(accs))])
            i += 1
            pass
        acc = np.mean(accs)
        return {"acc": 100 * acc}



if __name__ == '__main__':
    model = SentenceRelationModel()
    train_data, dev_data = get_train_dev()
    # load_model(dev_data)
    #
    model.build()
    print('*'*20, model.run_evaluate(dev_data)["acc"])
