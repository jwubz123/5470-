import EM as EM
import numpy as np
from sklearn.cluster import KMeans
data = np.genfromtxt('/Users/wujiamin/Desktop/CORTEX data/gene_expression.txt', delimiter=' ')
label_true = np.genfromtxt('/Users/wujiamin/Desktop/CORTEX data/labels.txt', delimiter=' ')
label_true=label_true.astype(int)
n_clus=7
N,D=data.shape
K=50
EZ, params, A, mus, sigmas, decay_coef=EM.fitModel(data, K, singleSigma=False)
mu=np.zeros([D, N])
for i in range(0,N):
    mu[:,i]=list(mus)
fit_data=EZ
clf = KMeans(n_clusters=n_clus)
clf.fit(fit_data)
labels = clf.labels_
accu_count=0
for i in range(0, len(label_true)):
    if label_true[i]==labels[i]:
        accu_count=accu_count+1
accuracy=accu_count/len(label_true)
# print(accuracy)
