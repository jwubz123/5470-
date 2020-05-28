from sklearn.cluster import KMeans
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


data = np.genfromtxt('/Users/wujiamin/Desktop/CORTEX data/gene_expression.txt', delimiter=' ')
label_true = np.genfromtxt('/Users/wujiamin/Desktop/CORTEX data/labels.txt', delimiter=' ')
data = torch.from_numpy(np.array(data)).to('cpu')
data=np.array(data)
label_true = label_true.astype(int)
n_clus = 7
N,D=data.shape
input_dim=D
K=10
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc2_mean = nn.Linear(400, K)
        self.fc2_logvar = nn.Linear(400, K)
        self.fc3 = nn.Linear(K, 400)
        self.fc4 = nn.Linear(400, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mean(h1), self.fc2_logvar(h1)

    def reparametrization(self, mu, logvar):
        # sigma = 0.5*exp(log(sigma^2))= 0.5*exp(log(var))
        std = 0.5 * torch.exp(logvar)
        z = torch.randn(std.size()) * std + mu
        return z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrization(mu, logvar)
        return self.decode(z), mu, logvar
vae = VAE().to('cpu')
optimizer = torch.optim.Adam(vae.parameters(), lr=0.0003)
trainloader = torch.utils.data.DataLoader(data, batch_size=3005, shuffle=True)
def train(epoch):
    vae.train()
    inputs= vae(data)
    real_imgs = torch.flatten(inputs, start_dim=1)

    gen_imgs, mu, logvar = vae(real_imgs)
    return mu

for i in range(20):
    EZ= train(i)
fit_data = EZ
clf = KMeans(n_clusters=n_clus)
clf.fit(fit_data)
labels = clf.labels_
accu_count = 0
for i in range(0, len(label_true)):
    if label_true[i] == labels[i]:
        accu_count = accu_count + 1
accuracy = accu_count / len(label_true)
print(accuracy)