## Plot ROC AUC Curves

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib as mpl

mpl.style.use('seaborn')
y_test = np.load('test_labels.npy')

frames_pred = np.load("frames_preds.npy")
frames_prob = np.load("frames_probabs.npy")
# print(frames_pred.shape, frames_prob.shape)
knn_pred = np.load("knn_preds.npy")
knn_probab = np.load("knn_probabs.npy")
# lstm_pred = np.load("lstm_preds.npy")

sgd_pred = np.load("sgd_preds.npy")
sgd_probab = np.load("sgd_probabs.npy")

fprf, tprf, thresholdf = roc_curve(y_test, frames_prob[:, 1])
fprk, tprk, thresholdk = roc_curve(y_test, knn_probab[:, 1])
fprs, tprs, thresholds = roc_curve(y_test, sgd_probab[:, 1])
# fprk, tprk, thresholdk = roc_curve(y_test, knn_probab[:, 1])

fnrf = 1 - tprf
fnrk = 1 - tprk
fnrs = 1 - tprs

eer_threshold = thresholdf[np.nanargmin(np.absolute((fnrf - fprf)))]
EER = fprf[np.nanargmin(np.absolute((fnrf - fprf)))]
print("frames")
print(EER)
roc_aucf = auc(fprf, tprf)
print(roc_aucf)

eer_threshold = thresholdk[np.nanargmin(np.absolute((fnrk - fprk)))]
EER = fprk[np.nanargmin(np.absolute((fnrk - fprk)))]
print("knn")
print(EER)
roc_auck = auc(fprk, tprk)
print(roc_auck)

eer_threshold = thresholds[np.nanargmin(np.absolute((fnrs - fprs)))]
EER = fprs[np.nanargmin(np.absolute((fnrs - fprs)))]
print("SGD")
print(EER)
roc_aucs = auc(fprs, tprs)
print(roc_aucs)

plt.title('Receiver Operating Characteristic')
plt.plot(fprf, tprf, 'm', label = 'AUC (Frames only) = %0.2f' % roc_aucf)
plt.plot(fprk, tprk, 'b', label = 'AUC (Triplets)= %0.2f' % roc_auck)
plt.plot(fprk, tprk, 'y--', label = 'AUC (Triplets)= %0.2f' % roc_aucs)

plt.grid()
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('AUC ROC Curve')
plt.savefig("AUC-ROC Score Curve")

# show the plot
plt.show()
