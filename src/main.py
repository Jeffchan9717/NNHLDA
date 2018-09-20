# -*- coding: utf-8 -*-
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score
from sklearn.cross_validation import KFold
from utils import *
from tflearn.activations import relu
from sklearn.cross_validation import train_test_split,StratifiedKFold
from sklearn.model_selection import LeaveOneOut
import sys
from tqdm import tqdm
from optparse import OptionParser
import matplotlib.pyplot as plt
import copy

ori_ns = 1001
ns = 100

parser = OptionParser()
parser.add_option("-d", "--d", default=64, help="The embedding dimension d")
parser.add_option("-k","--k",default=64,help="The dimension of project matrices k")
parser.add_option("-r", "--r", default=10, help='Rate of being tested')
(opts, args) = parser.parse_args()

#load network
network_path = '../data/'
rna_rna = np.loadtxt(network_path+'lncRNA_similarity.txt')
rna_dis = np.loadtxt(network_path+'known_lncRNA_disease_interaction.txt')
dis_dis = np.loadtxt(network_path+'disease_similarity.txt')

rna_rna_normalize = row_normalize(rna_rna,True)
rna_dis_normalize = row_normalize(rna_dis,False)
dis_dis_normalize = row_normalize(dis_dis,True)

#define computation graph
num_rna = len(rna_dis_normalize)
num_dis = len(dis_dis_normalize)

dim_rna = int(opts.d)
dim_dis = int(opts.d)
dim_pred = int(opts.k)
dim_pass = int(opts.d)
rate = int(opts.r)

class Model(object):
    #domain adaptation model.
    def __init__(self):
        self._build_model()
    def _build_model(self):
        #inputs
        self.rna_rna = tf.placeholder(tf.float32, [num_rna, num_rna])
        self.rna_rna_normalize = tf.placeholder(tf.float32, [num_rna, num_rna])
        
        self.dis_dis = tf.placeholder(tf.float32, [num_dis, num_dis])
        self.dis_dis_normalize = tf.placeholder(tf.float32, [num_dis, num_dis])

        self.rna_dis = tf.placeholder(tf.float32, [num_rna, num_dis])
        self.rna_dis_normalize = tf.placeholder(tf.float32, [num_rna, num_dis])

        self.dis_rna = tf.placeholder(tf.float32, [num_dis, num_rna])
        self.dis_rna_normalize = tf.placeholder(tf.float32, [num_dis, num_rna])

        self.rna_dis_mask = tf.placeholder(tf.float32, [num_rna, num_dis])

        #features
        self.rna_embedding = weight_variable([num_rna,dim_rna])
        self.dis_embedding = weight_variable([num_dis,dim_dis])
        
        #feature passing weights (maybe different types of nodes can use different weights)
        W0 = weight_variable([dim_pass+dim_rna, dim_rna])
        b0 = bias_variable([dim_rna])

        rna_vector1 = tf.nn.l2_normalize(relu(tf.matmul(
            tf.concat([\
            tf.matmul(self.rna_rna_normalize, a_layer(self.rna_embedding, dim_pass)) + \
            tf.matmul(self.rna_dis_normalize, a_layer(self.dis_embedding, dim_pass)), \
            self.rna_embedding], axis=1), W0)+b0),dim=1)

        dis_vector1 = tf.nn.l2_normalize(relu(tf.matmul(
            tf.concat([\
            tf.matmul(self.dis_dis_normalize, a_layer(self.dis_embedding, dim_pass)) + \
            tf.matmul(self.dis_rna_normalize, a_layer(self.rna_embedding, dim_pass)) , \
            self.dis_embedding], axis=1), W0)+b0),dim=1)

        self.rna_representation = rna_vector1
        self.dis_representation = dis_vector1

        #reconstructing networks
        self.rna_rna_reconstruct = bi_layer(self.rna_representation,self.rna_representation, sym=True, dim_pred=dim_pred)
        self.rna_rna_reconstruct_loss = tf.reduce_sum(tf.multiply((self.rna_rna_reconstruct-self.rna_rna), (self.rna_rna_reconstruct-self.rna_rna)))

        self.dis_dis_reconstruct = bi_layer(self.dis_representation,self.dis_representation, sym=True, dim_pred=dim_pred)
        self.dis_dis_reconstruct_loss = tf.reduce_sum(tf.multiply((self.dis_dis_reconstruct-self.dis_dis), (self.dis_dis_reconstruct-self.dis_dis)))

        self.rna_dis_reconstruct = bi_layer(self.rna_representation,self.dis_representation, sym=False, dim_pred=dim_pred)
        tmp = tf.multiply(self.rna_dis_mask, (self.rna_dis_reconstruct-self.rna_dis))
        self.rna_dis_reconstruct_loss = tf.reduce_sum(tf.multiply(tmp, tmp))

        self.loss = self.rna_dis_reconstruct_loss + 1.0*(self.rna_rna_reconstruct_loss + \
                                                            self.dis_dis_reconstruct_loss)

def train_and_evaluate(entry, DTItrain, DTIvalid, graph, verbose=True, num_steps = 4000):
    rna_dis = np.zeros((num_rna,num_dis))
    mask = np.zeros((num_rna,num_dis))
    for ele in DTItrain:
        rna_dis[ele[0],ele[1]] = ele[2]
        mask[ele[0],ele[1]] = 1

    # if only one entry at the row(lncRNA) or column(disease), replace with self information
    if np.any(entry):
        if rna_dis[entry[0], :].sum() == 0:
            rna_curr = rna_rna_normalize[entry[0], :]
            rna_curr = np.expand_dims(rna_curr, axis=0)
            rna_dis[entry[0], :] = np.squeeze(np.dot(rna_curr, rna_dis))
        if rna_dis[:, entry[1]].sum() == 0:
            dis_curr = dis_dis_normalize[entry[1], :]
            dis_curr = np.expand_dims(dis_curr, axis=0)
            rna_dis[:, entry[1]] = np.squeeze(np.dot(dis_curr, rna_dis.T).T)
    dis_rna = rna_dis.T
    rna_dis_normalize = row_normalize(rna_dis,False)
    dis_rna_normalize = row_normalize(dis_rna,False)

    lr = 0.001
    best_valid_auc = 0
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        for i in xrange(num_steps):
            _, tloss, dtiloss, results, reconstruction = sess.run([optimizer,total_loss,dti_loss,eval_pred, reconstruct], \
                                        feed_dict={\
                                        model.rna_rna:rna_rna, model.rna_rna_normalize:rna_rna_normalize,\
                                        model.dis_dis:dis_dis, model.dis_dis_normalize:dis_dis_normalize,\
                                        model.rna_dis:rna_dis, model.rna_dis_normalize:rna_dis_normalize,\
                                        model.dis_rna:dis_rna, model.dis_rna_normalize:dis_rna_normalize,\
                                        model.rna_dis_mask:mask,\
                                        learning_rate: lr})
            pred_list = []
            ground_truth = []
            for ele in DTIvalid:
                pred_list.append(results[ele[0],ele[1]])
                ground_truth.append(ele[2])
            valid_auc = roc_auc_score(ground_truth, pred_list)
            if valid_auc >= best_valid_auc:
                result_origin = results
                best_valid_auc = valid_auc

    return result_origin

graph = tf.get_default_graph()
with graph.as_default():
    model = Model()
    learning_rate = tf.placeholder(tf.float32, [])
    total_loss = model.loss
    dti_loss = model.rna_dis_reconstruct_loss
    reconstruct = model.rna_dis_reconstruct

    optimize = tf.train.AdamOptimizer(learning_rate)
    gradients, variables = zip(*optimize.compute_gradients(total_loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1)
    optimizer = optimize.apply_gradients(zip(gradients, variables))

    eval_pred = model.rna_dis_reconstruct

# lda_seq contains all entries
lda_origin = np.loadtxt(network_path+'known_lncRNA_disease_interaction.txt')
ground_truth = lda_origin.flatten()

lda_seq = np.stack(np.where(lda_origin==1), axis=1)
num_entry = len(lda_seq)
print 'total Entry number: ', num_entry
np.random.shuffle(lda_seq)

pos_set = np.stack((np.where(lda_origin==1)), axis=1)
pos_set = np.hstack((pos_set, np.expand_dims(lda_origin[np.where(lda_origin==1)], axis=1)))
pos_set = np.asarray(pos_set, dtype=int)
neg_set = np.stack((np.where(lda_origin==0)), axis=1)
neg_set = np.hstack((neg_set, np.expand_dims(lda_origin[np.where(lda_origin==0)], axis=1)))
neg_set = np.asarray(neg_set, dtype=int)
np.random.shuffle(pos_set)
np.random.shuffle(neg_set)
all_set = np.concatenate((pos_set, neg_set), axis=0)

pos_len = len(pos_set)
neg_len = len(neg_set)
all_len = len(all_set)
print num_rna, num_dis

# split into train set and test set
ptr, ptt = train_test_split(pos_set, test_size=0.05)
ntr, ntt = train_test_split(neg_set, test_size=0.05)
train_set = np.concatenate((ptr, ntr))
test_set = np.concatenate((ptt, ntt))
np.random.shuffle(train_set)
train_set = train_set[:rate*len(test_set)]
DTItrain, DTIvalid = [train_set, test_set]
score_ori_matrix = train_and_evaluate(None, DTItrain=DTItrain, DTIvalid=DTIvalid, graph=graph, num_steps=ori_ns)
score_res_matrix = copy.deepcopy(score_ori_matrix)
print 'score_ori_matrix obtained done.', score_ori_matrix.shape
pred_list = score_ori_matrix.flatten()

r = 0
# point = []
# cnt1 = []
# err = []
for asso_train_index, asso_test_index in tqdm(LeaveOneOut().split(lda_seq)):
    asso_test = lda_seq[asso_test_index]
    # print asso_test
    if r+1 % 10 == 0: print 'sample round %d/%d' % (r+1, len(lda_seq))
    r = r + 1

    cur_elem = np.hstack([asso_test, [[1]]])
    cur_elem = np.squeeze(cur_elem)
    DTItrain = np.delete(train_set, np.where(np.all(train_set==cur_elem, axis=1)), axis=0)
    DTIvalid = np.delete(test_set, np.where(np.all(test_set==cur_elem, axis=1)), axis=0)

    res = train_and_evaluate(asso_test[0], DTItrain=DTItrain, DTIvalid=DTIvalid, graph=graph, num_steps=ns)
    score_res_matrix[tuple(asso_test[0])] = res[tuple(asso_test[0])]
    pred_list = score_res_matrix.flatten()

np.savetxt('score_ori_matrix', score_ori_matrix, fmt='%.5f')
np.savetxt('score_res_matrix', score_res_matrix, fmt='%.5f')
pred_list = score_res_matrix.flatten()
auc = calculate_auc_and_draw(ground_truth, pred_list, draw=True)
with open('result', 'a') as f:
    print >> f, 'pos:neg is: %d, auc is: %.4f' % (rate, auc)
