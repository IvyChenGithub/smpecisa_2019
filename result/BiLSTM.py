# 模型文件
# 静态和动态的都有,这次用动态的
import os
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn

from TfUtils import linear
from attn_cell import AttnCell
from load_train_dev import pad_sequences, minibatches
from load_embedding import get_embedding_data
from general_utils import Progbar
import numpy as np
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.rnn_cell_impl import *
import core_rnn_cell_impl as rnn_cell

# 参数集
# vocabulary_size = 5000  # 整个词表的大小
embedding_size = 200  # 词向量的维度的大小
hidden_size_lstm = 196  # 隐藏层大小
ntags = 3  # 0,1,2
num_layers = 3  # 多层BiLSTM的层数
lr_method = 'adam'  # 训练的方法
lr = 0.0001  # learning rate
clip = -1  # 将损失裁切，-1为不进行裁切
batch_size = 50  # 每次训练的大小
dropout = 0.6
nepoches = 3000  # 迭代次数
lr_decay = 0.9  # learning rate损失率
nepoche_to_change = 10  # nepoche_to_change次后精度依旧没有提高的话就减小损失
# word2vec_type = 'SG'  # 采取word2vec_type方法训练的词向量（SG:skip gram ）
# embedding_save_dir = 'save/word2vec_' + word2vec_type
attention_size = 50     # attention的维数

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


def attention(inputs, attention_size, time_major=False, return_alphas=False):

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas


class SentenceRelationModel:
    """判断两个句子关系的模型"""

    def __init__(self, model_name='SentenceRelationModel3'):
        # 首先需要修改默认图
        tf.reset_default_graph()
        # save_dir需要存储的信息有：
        # epoch、num_layers、hidden_size_lstm
        self.save_dir = 'save/' + model_name + '/' + model_name + '_nl_' + str(num_layers) +\
                        '_hsl_' + str(hidden_size_lstm) + '__.cpkt'
        self.save_os = 'save/' + model_name
        self.seq1_words = None
        self.seq2_words = None
        self.sequence1_lengths = None
        self.sequence2_lengths = None
        self.labels = None
        self.dropout = None
        self.lr = None
        self.embeddings = get_embedding_data()  # 所有词的词向量
        self.seq1_word_embeddings = None  # seq1的batch要用到的词的词向量
        self.seq2_word_embeddings = None  # seq2的batch要用到的词的词向量
        self.logits = None
        self.loss = None
        self.train_op = None
        self.sess = None
        self.saver = None
        pass

    def add_placeholders(self):
        """添加所有的placehonder"""
        # 训练数据ph
        self.seq1_word_embeddings = tf.placeholder(tf.float32, shape=[None, None, embedding_size],
                                                   name='seq1_word_embeddings')  # [batch_size, max_seq_length, embedding_size]
        self.seq2_word_embeddings = tf.placeholder(tf.float32, shape=[None, None, embedding_size],
                                                   name='seq2_word_embeddings')  # [batch_size, max_seq_length, embedding_size]
        # 句子长度
        self.sequence1_lengths = tf.placeholder(tf.int32, shape=[None],
                                                name='sequence1_lengths')  # [batch_size]
        self.sequence2_lengths = tf.placeholder(tf.int32, shape=[None],
                                                name='sequence2_lengths')  # [batch_size]
        # 标签
        self.labels = tf.placeholder(tf.int32, shape=[None],
                                     name='labels')  # [batch_size, ntags]
        # 超参
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name='dropout')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                                 name='lr')
        pass

    def get_feed_dict(self, words1, words2, labels=None, lr=None, dropout=None):
        """
        获取feed_dict
        Args:
            words1:尚未对齐的句子的batch，list形式，由string组成
            words2:尚未对齐的句子的batch，list形式，由string组成
            labels:标签
            lr:学习率
            dropout:dropout率
        """
        words1, sequence1_lengths = pad_sequences(words1)
        words2, sequence2_lengths = pad_sequences(words2)
        # print('words1=', words1, '\n', len(words1), '\n', sequence1_lengths, '\n', len(sequence1_lengths))
        # print('words2=', words2, '\n', len(words2), '\n', sequence2_lengths, '\n', len(sequence2_lengths))
        # 接下来将words1和words2转换为embeddings
        # words1/words2: [batch_size, max_seq_length1/max_seq_length2]
        words1_embeddings = [[self.embeddings[w1] if w1 in self.embeddings.keys() else self.embeddings[","]
                              for w1 in ws1] if len(ws1)>0 else [self.embeddings["。"]] for ws1 in words1]
        words2_embeddings = [[self.embeddings[w2] if w2 in self.embeddings.keys() else self.embeddings[","]
                              for w2 in ws2] if len(ws2)>0 else [self.embeddings["。"]] for ws2 in words2]
        # print(len(words1_embeddings), len(words2_embeddings))
        feed = {
            # [batch_size, max_seq_length1/max_seq_length2, embedding_size]
            self.seq1_word_embeddings: words1_embeddings,
            self.seq2_word_embeddings: words2_embeddings,
            self.sequence1_lengths: sequence1_lengths,
            self.sequence2_lengths: sequence2_lengths
        }
        if labels is not None:
            # labels, _ = pad_sequences(labels, 0)
            # feed[self.labels] = tf.one_hot(labels, depth=ntags, on_value=1.0, off_value=0.0, axis=-1)
            feed[self.labels] = labels
            pass
        if lr is not None:
            feed[self.lr] = lr
            pass
        if dropout is not None:
            feed[self.dropout] = dropout
            pass
        return feed, sequence1_lengths, sequence2_lengths

    def add_word_embeddings_op(self):
        """获取word embeddings"""
        # self.embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1, 1),
        #                               name='embeddings', trainable=False)
        # self.embeddings = get_embedding_data()
        # self.word_embeddings = tf.nn.embedding_lookup(self.embeddings, self.word_ids, name="word_embeddings")
        pass

    def add_logits_op(self):
        """利用BLSTM生成结果，batch中每个句子的每个单词都有一个结果，一个结果是n维的变量，n大小为类别的数目"""
        stacked_rnn_fw1 = []
        print('self.sequence1_lengths=', np.shape(self.sequence1_lengths))
        print('self.sequence2_lengths=', np.shape(self.sequence2_lengths))
        for _ in range(num_layers):
            fw_cell1 = BasicLSTMCell(hidden_size_lstm, forget_bias=1.0, state_is_tuple=True
                                     , name="fw_cell1"+str(num_layers))
            stacked_rnn_fw1.append(fw_cell1)
        lstm_fw_cell_m1 = MultiRNNCell(cells=stacked_rnn_fw1, state_is_tuple=True)
        stacked_rnn_bw1 = []
        for _ in range(num_layers):
            bw_cell1 = BasicLSTMCell(hidden_size_lstm, forget_bias=1.0, state_is_tuple=True
                                     , name="bw_cell1"+str(num_layers))
            stacked_rnn_bw1.append(bw_cell1)
        lstm_bw_cell_m1 = MultiRNNCell(cells=stacked_rnn_bw1, state_is_tuple=True)
        (output_fw1, output_bw1), _ = bidirectional_dynamic_rnn(lstm_fw_cell_m1, lstm_bw_cell_m1,
                                                                      self.seq1_word_embeddings,
                                                                      sequence_length=self.sequence1_lengths,
                                                                      dtype=tf.float32)
        stacked_rnn_fw2 = []
        for _ in range(num_layers):
            fw_cell2 = BasicLSTMCell(hidden_size_lstm, forget_bias=1.0, state_is_tuple=True
                                     , name="fw_cell2" + str(num_layers))
            stacked_rnn_fw2.append(fw_cell2)
        lstm_fw_cell_m2 = MultiRNNCell(cells=stacked_rnn_fw2, state_is_tuple=True)
        stacked_rnn_bw2 = []
        for _ in range(num_layers):
            bw_cell2 = BasicLSTMCell(hidden_size_lstm, forget_bias=1.0, state_is_tuple=True
                                     , name="bw_cell2" + str(num_layers))
            stacked_rnn_bw2.append(bw_cell2)
        lstm_bw_cell_m2 = MultiRNNCell(cells=stacked_rnn_bw2, state_is_tuple=True)
        (output_fw2, output_bw2), _ = bidirectional_dynamic_rnn(lstm_fw_cell_m2, lstm_bw_cell_m2,
                                                                      self.seq2_word_embeddings,
                                                                      sequence_length=self.sequence2_lengths,
                                                                      dtype=tf.float32)
        # 接下来是利用attention构造句子级的表示
        # output_fw1/2/output_bw1/2: [batch_size, max_seq_length1/2, embedding_size]
        # output_fw12 = tf.concat([output_fw1, output_fw2], axis=1)
        # output_bw12 = tf.concat([output_bw1, output_bw2], axis=1)
        output_fw12 = output_fw1
        output_bw12 = output_bw1
        output1 = attention((output_fw12, output_bw12),
                            attention_size, return_alphas=False)
        # output_fw21 = tf.concat([output_fw2, output_fw1], axis=1)
        # output_bw21 = tf.concat([output_bw2, output_bw1], axis=1)
        output_fw21 = output_fw2
        output_bw21 = output_bw2
        output2 = attention((output_fw21, output_bw21),
                            attention_size, return_alphas=False)
        # output = output1 + output2
        print('output1=', np.shape(output1))
        print('output2=', np.shape(output2))
        output = tf.concat([output1, output2], axis=1)
        output = tf.nn.dropout(output, self.dropout)  # dropout
        print('shape of output=', np.shape(output))
        # 接下来构造映射层
        # W = tf.get_variable('W', dtype=tf.float32,
        #                     shape=[hidden_size_lstm*2, ntags])
        W = tf.get_variable('W', dtype=tf.float32,
                            shape=[hidden_size_lstm * 4, ntags])
        b = tf.get_variable('b', dtype=tf.float32, shape=[ntags],
                            initializer=tf.zeros_initializer())
        pred = tf.matmul(output, W) + b
        logits = tf.reshape(pred, [-1, ntags])
        logits = tf.nn.softmax(logits)
        self.logits = logits
        pass

    def add_logits_op1(self):
        """利用BLSTM生成结果，batch中每个句子的每个单词都有一个结果，一个结果是n维的变量，n大小为类别的数目"""
        stacked_rnn_fw1 = []
        for _ in range(num_layers):
            fw_cell1 = BasicLSTMCell(hidden_size_lstm, forget_bias=1.0, state_is_tuple=True
                                     , name="fw_cell1"+str(num_layers))
            stacked_rnn_fw1.append(fw_cell1)
        lstm_fw_cell_m1 = MultiRNNCell(cells=stacked_rnn_fw1, state_is_tuple=True)
        stacked_rnn_bw1 = []
        for _ in range(num_layers):
            bw_cell1 = BasicLSTMCell(hidden_size_lstm, forget_bias=1.0, state_is_tuple=True
                                     , name="bw_cell1"+str(num_layers))
            stacked_rnn_bw1.append(bw_cell1)
        lstm_bw_cell_m1 = MultiRNNCell(cells=stacked_rnn_bw1, state_is_tuple=True)
        print('self.sequence1_lengths=', np.shape(self.sequence1_lengths))
        print('self.sequence2_lengths=', np.shape(self.sequence2_lengths))
        # sequence_length = tf.cond(tf.greater(self.sequence1_lengths, self.sequence2_lengths),
        #                           lambda: self.sequence1_lengths, lambda: self.sequence2_lengths)
        (output1_a, output1_b), _ = bidirectional_dynamic_rnn(lstm_fw_cell_m1, lstm_bw_cell_m1,
                                            self.seq1_word_embeddings,
                                            # sequence_length=sequence_length,
                                            dtype=tf.float32)
        stacked_rnn_fw2 = []
        for _ in range(num_layers):
            fw_cell2 = BasicLSTMCell(hidden_size_lstm, forget_bias=1.0, state_is_tuple=True
                                     , name="fw_cell2" + str(num_layers))
            stacked_rnn_fw2.append(fw_cell2)
        lstm_fw_cell_m2 = MultiRNNCell(cells=stacked_rnn_fw2, state_is_tuple=True)
        stacked_rnn_bw2 = []
        for _ in range(num_layers):
            bw_cell2 = BasicLSTMCell(hidden_size_lstm, forget_bias=1.0, state_is_tuple=True
                                     , name="bw_cell2" + str(num_layers))
            stacked_rnn_bw2.append(bw_cell2)
        lstm_bw_cell_m2 = MultiRNNCell(cells=stacked_rnn_bw2, state_is_tuple=True)
        (output2_a, output2_b), _ = bidirectional_dynamic_rnn(lstm_fw_cell_m2, lstm_bw_cell_m2,
                                            self.seq2_word_embeddings,
                                            # sequence_length=sequence_length,
                                            dtype=tf.float32)
        output1 = tf.concat((output1_a, output1_b), 2)
        output2 = tf.concat((output2_a, output2_b), 2)
        print('output1=', output1)
        print('output2=', output2)
        # print('output1[-1]=', np.shape(output1[1]))
        # print('output2[-1]=', np.shape(output2[1]))
        output = tf.add(output1, output2)
        output = tf.nn.dropout(output, self.dropout)  # dropout
        # 接下来构造映射层
        # W = tf.get_variable('W', dtype=tf.float32,
        #                     shape=[hidden_size_lstm*2, ntags])
        W = tf.get_variable('W', dtype=tf.float32,
                            shape=[hidden_size_lstm * 2, ntags])
        b = tf.get_variable('b', dtype=tf.float32, shape=[ntags],
                            initializer=tf.zeros_initializer())
        print('shape of output=', np.shape(output))
        print('shape of W=', np.shape(W))
        pred = tf.matmul(output, W) + b
        logits = tf.reshape(pred, [-1, ntags])
        logits = tf.nn.softmax(logits)
        print('shape of logits=', np.shape(logits))
        self.logits = logits
        pass

    def add_logits_op2(self):
        """利用BLSTM生成结果，batch中每个句子的每个单词都有一个结果，一个结果是n维的变量，n大小为类别的数目"""
        with tf.variable_scope('Premise_encoder'):
            lstm_cell = rnn_cell.BasicLSTMCell(hidden_size_lstm)
            lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=self.dropout,
                                                output_keep_prob=self.dropout)
            Premise_out, Premise_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell,cell_bw=lstm_cell,
                                                                         inputs=self.seq1_word_embeddings,
                                                                         sequence_length=self.sequence1_lengths,
                                                                         dtype=tf.float32,
                                                                         swap_memory=True)
            Premise_output_fw, Premise_output_bw = Premise_out
            Premise_states_fw, Premise_states_bw = Premise_state
            Premise_out = tf.concat(Premise_out, 2)
            Premise_state = tf.concat(Premise_state, 2)
        with tf.variable_scope('Hypothesis_encoder'):
            lstm_cell = rnn_cell.BasicLSTMCell(hidden_size_lstm)
            lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=self.dropout,
                                                output_keep_prob=self.dropout)
            Hypo_out, Hypo_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell,cell_bw=lstm_cell,
                                                                   inputs=self.seq2_word_embeddings,
                                                                   sequence_length=self.sequence2_lengths,
                                                                   # initial_state_fw=Premise_states_fw,
                                                                   # initial_state_bw=Premise_states_bw,
                                                                   dtype=tf.float32,
                                                                   swap_memory=True)
            print('before=', np.shape(Hypo_state[1]))
            Hypo_out = tf.concat(Hypo_out, 2)
            Hypo_state = tf.concat(Hypo_state, 2)

        def w2w_attn(Premise_out, Hypo_out, seqLen_Premise, seqLen_Hypo, scope=None):
            with tf.variable_scope(scope or 'Attn_layer'):
                attn_cell = AttnCell(196*2, Premise_out, seqLen_Premise)
                attn_cell = rnn_cell.DropoutWrapper(attn_cell, input_keep_prob=self.dropout,
                                                    output_keep_prob=self.dropout)

                _, r_state = tf.nn.dynamic_rnn(attn_cell, Hypo_out, seqLen_Hypo,
                                               dtype=Hypo_out.dtype, swap_memory=True)
            return r_state

        r_L = w2w_attn(Premise_out, Hypo_out, self.sequence1_lengths, self.sequence2_lengths, scope='w2w_attention')

        hypo_state1 = tf.reshape(Hypo_state[1], [-1, 392])
        hypo_state1 = tf.nn.dropout(hypo_state1, 0.5)
        print('***********', np.shape(r_L))
        print('***********', np.shape(hypo_state1))

        h_star = tf.tanh(linear([r_L, hypo_state1],       # shape (b_sz, h_sz)
                                392, bias=False,
                                scope='linear_trans'))
        input_fully = h_star
        output = tf.nn.dropout(input_fully, self.dropout)
        W = tf.get_variable('W', dtype=tf.float32,
                            shape=[hidden_size_lstm * 2, ntags])
        b = tf.get_variable('b', dtype=tf.float32, shape=[ntags],
                            initializer=tf.zeros_initializer())
        pred = tf.matmul(output, W) + b
        logits = tf.reshape(pred, [-1, ntags])
        logits = tf.nn.softmax(logits)
        self.logits = logits
        '''
        for i in range(2):
            with tf.variable_scope('fully_connect_'+str(i)):
                logits = tf.contrib.layers.fully_connected(
                    input_fully, 300 * 2, activation_fn=None)
                input_fully = tf.tanh(logits)
        with tf.name_scope('Softmax'):
            logits = tf.contrib.layers.fully_connected(
                input_fully, self.config.class_num, activation_fn=None)
        self.logits = logits
        '''

        '''

        output1 = attention((output_fw12, output_bw12),
                            attention_size, return_alphas=False)
        # output_fw21 = tf.concat([output_fw2, output_fw1], axis=1)
        # output_bw21 = tf.concat([output_bw2, output_bw1], axis=1)
        output_fw21 = output_fw2
        output_bw21 = output_bw2
        output2 = attention((output_fw21, output_bw21),
                            attention_size, return_alphas=False)
        # output = output1 + output2
        print('output1=', np.shape(output1))
        print('output2=', np.shape(output2))
        output = tf.concat([output1, output2], axis=1)
        output = tf.nn.dropout(output, self.dropout)  # dropout
        print('shape of output=', np.shape(output))
        # 接下来构造映射层
        # W = tf.get_variable('W', dtype=tf.float32,
        #                     shape=[hidden_size_lstm*2, ntags])
        W = tf.get_variable('W', dtype=tf.float32,
                            shape=[hidden_size_lstm * 4, ntags])
        b = tf.get_variable('b', dtype=tf.float32, shape=[ntags],
                            initializer=tf.zeros_initializer())
        pred = tf.matmul(output, W) + b
        logits = tf.reshape(pred, [-1, ntags])
        logits = tf.nn.softmax(logits)
        self.logits = logits
        '''
        pass

    def add_loss_op(self):
        """二次均方误差"""
        labels = tf.one_hot(self.labels, depth=ntags, on_value=1.0, off_value=0.0, axis=-1)
        # 先用上面的loss训练，大踏步到达相对最优
        # 再用下面的loss训练，精细调整
        # self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.logits[0]-tf.cast(labels, tf.float32)), axis=1))
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.logits - tf.cast(labels, tf.float32)), axis=1))
        pass

    def add_train_op(self, lr_method, lr, loss, clip=-1):
        """
        定义了在一个batch上运行的训练步
        Args:
            lr_method:字符串类型，优化方法
            lr:learning rate
            loss:损失option
            clip:损失裁切的阈值
        """
        _lr_m = lr_method.lower()  # 字母小写
        with tf.variable_scope("train_step"):
            if _lr_m == 'adam':
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            if clip > 0:  # gradient clipping if clip is positive
                grads, vs = zip(*optimizer.compute_gradients(loss))
                grads, gnorm = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)
            pass
        pass

    def initialize_session(self):
        """进行session初始化"""
        # 需要设置结果sess
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver(max_to_keep=1)
        # 还有embedding的
        # self.saver_embedding = tf.train.Saver(var_list=[self.embeddings], max_to_keep=1)  # 生成saver
        self.sess.run(tf.global_variables_initializer())
        # 接下来查看目标文件是否存在
        if not os.path.exists(self.save_os):
            os.makedirs(self.save_os)
        pass

    def build(self):
        """创建整个训练模型"""
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op2()
        self.add_loss_op()

        # 训练步骤
        self.add_train_op(lr_method, lr, self.loss, clip)
        self.initialize_session()

        # # 导入词向量模型
        # kpt_embedding = tf.train.latest_checkpoint(embedding_save_dir)
        # print("正在导入词向量===================================================================")
        # self.saver_embedding.restore(self.sess, kpt_embedding)
        # print("词向量导入完毕")
        print("模型构建完毕=====================================================================")

        # 查看之前是否存在，如果存在就导入
        kpt = tf.train.latest_checkpoint(self.save_os)
        if kpt is not None:
            print("正在导入模型===================================================================")
            self.saver.restore(self.sess, kpt)
            pass
        pass

    def predict_batch(self, words1, words2):
        """
        对一个batch进行预测
        :param words: 句子列表，为一个batch，尚未对齐
        :return:labels_pred:每个句子的labels
            sequence_length
        """
        fd, sequence1_lengths, sequence2_lengths = self.get_feed_dict(words1, words2, dropout=1.0)
        # 接下来计算结果
        # viterbi_sequences = []
        # logits:[batch_size, max_time. 2*hidden_size_lstm]
        # trans_params:[ntags, ntags]
        logits = self.sess.run([self.logits], feed_dict=fd)
        tf.add_to_collection('pred_network', logits)
        # logits => [batch_size, 3]
        return logits

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
        cnt_eq = 0
        for words1, words2, labels in minibatches(test, batch_size):
            logits = self.predict_batch(words1, words2)
            for logit, label in zip(logits[0], labels):
                # print(logit, np.argmax(logit), label)
                accs += [np.argmax(logit) == label]
                if np.argmax(logit) == label:
                    cnt_eq += 1
                pass
            prog.update(i + 1, [("evaluate acc", 100*np.mean(accs))])
            i += 1
            print('cnt_eq=', cnt_eq)
            pass
        # print('cnt_eq=', cnt_eq)
        acc = np.mean(accs)
        return {"acc": 100 * acc}

    def run_epoch(self, train, dev, epoch):
        """
        在训练集上和测试集上完整地跑一回
        :param train:(sentences, tags)，其中sentences尚未对齐
        :param dev:类似于train的测试数据集
        :param epoch:当前epoch的编号
        :return f1:准确率
        """
        nbatches = (len(train) + batch_size - 1) // batch_size  # 训练batch的次数
        prog = Progbar(target=nbatches)
        # 首先遍历整个训练集
        for i, (words1, words2, labels) in enumerate(minibatches(train, batch_size)):
            fd, _, _ = self.get_feed_dict(words1, words2, labels, lr, dropout)
            _, train_loss = self.sess.run([self.train_op, self.loss], feed_dict=fd)
            prog.update(i + 1, [("train loss", train_loss)])
            pass
        # 接下来是测试了
        metrics = self.run_evaluate(dev)
        # print('第{}次的准确率为:\t{}\t召回率为:\t{}'.format(epoch, metrics['acc'], metrics['rec']))
        return metrics["acc"]  # 返回准确率

    def predict(self, words_raw1, words_raw2):
        """
        对words_raw进行抽取
        :param words_raw1:经过分词、去除停用词、处理之后的句子
        :return preds:对应的tags
        """
        logit = self.predict_batch([words_raw1], [words_raw2])
        sentenceRelation = np.argmax(logit[0])
        return sentenceRelation

    def save(self, epoch):
        print('正在保存模型===============================')
        self.saver.save(self.sess, self.save_dir)
        print('保存模型结束===============================')
        pass


    def train(self, train, dev):
        """
        进行训练
        :param train:所有的训练数据。由(sequence, tags)组成的list
        :param dev:所有的测试数据。由(sequence, tags)组成的list
        :return:
        """
        global lr
        best_score = 0
        best_recall = 0
        nepoch_no_imprv = 0  # 提前停止设置的变量
        for epoch in range(nepoches):
            score = self.run_epoch(train, dev, epoch)
            # lr = lr_decay*lr
            # print('第{}次的准确率为:\t{}\t召回率为:\t{}'.format(epoch, metrics['acc'], metrics['rec']))
            if score < best_score:
                # 如果都没有上升的话
                nepoch_no_imprv += 1
                # 红底字符
                print("\033[0;30;41m第{}次的准确率为:\t{}\033[0m".format(epoch, score))
            elif score >= best_score:
                # 如果准确率上升的话，需要保存
                nepoch_no_imprv = 0
                best_score = score
                print("\033[0;30;46m第{}次的准确率为:\t{}\t\033[0m".format(epoch, score))
                # 保存
                self.save(epoch)
            # 看情况是否需要减小步长
            if nepoch_no_imprv > nepoche_to_change:
                lr *= lr_decay
                pass
            pass
        pass

    pass  # 类end

