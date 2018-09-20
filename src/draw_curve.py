import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pdb

ground_truth = np.loadtxt('../data/known_lncRNA_disease_interaction.txt', dtype=int)
pred_list = np.loadtxt('score_res_matrix', dtype='float32')

auc_list = []
for i in xrange(ground_truth.shape[0]):
    for j in xrange(ground_truth.shape[1]):
        gr_tr = []
        pred_li = []
        if ground_truth[i, j] == 1:
            gr_tr.append(ground_truth[i, j])
            pred_li.append(pred_list[i, j])
            for k in xrange(ground_truth.shape[0]):
                if ground_truth[k, j] == 0:
                    gr_tr.append(ground_truth[k, j])
                    pred_li.append(pred_list[k, j])
            if (len(auc_list)+1) % 1 == 0:
                print 'round ', len(auc_list) +1, 'len of gr_tr:', len(gr_tr)
                print ground_truth.shape[0] - len(gr_tr)
                print 'current auc: ', roc_auc_score(gr_tr, pred_li)
                print 'current total auc: ', np.mean(auc_list)
            auc_list.append(roc_auc_score(gr_tr, pred_li))
print 'local_auc: ', np.mean(auc_list)

ground_truth = ground_truth.flatten()
pred_list = pred_list.flatten()
print ground_truth[:10]
print pred_list[:10]

auc = roc_auc_score(ground_truth, pred_list)
fpr, tpr, sh = roc_curve(ground_truth, pred_list)

print 'global_auc: ', auc
plt.plot(fpr, tpr)
plt.title('global_auc: %f, local_auc: %f' % (auc, np.mean(auc_list)))
plt.savefig('roc_curve')
# plt.show()
