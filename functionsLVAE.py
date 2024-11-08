import numpy as np
import matplotlib.pyplot as plt
import torch as th
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from sklearn.decomposition import PCA

drop=0.2
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

## LABELED VARIATIONAL AUTOENCODER
class LVAE_Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_sizes=[1000,500]):
        super(LVAE_Encoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        layers = []
        for i in range(len(hidden_sizes)):
            layers.append(nn.Linear(self.input_dim if i == 0 else hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(drop))
        self.layers = nn.Sequential(*layers)

        if len(hidden_sizes)!=0 : dim = hidden_sizes[-1]
        else : dim = self.input_dim
        self.embedding = nn.Linear(dim,self.latent_dim)
        self.log_var = nn.Linear(dim,self.latent_dim)

    def forward(self, x:th.Tensor):
        out = self.layers(x)
        return self.embedding(out),self.log_var(out)


class LVAE_Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_sizes=[1000,500]):
        super(LVAE_Decoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        hid = hidden_sizes[::-1]
        layers = []
        for i in range(len(hid)):
            layers.append(nn.Linear(self.latent_dim if i == 0 else hid[i-1], hid[i]))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(drop))
        if len(hid)!=0 : dim = hid[-1]
        else : dim = self.latent_dim
        layers.append(nn.Linear(dim, self.input_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, z:th.Tensor):
        return self.layers(z)



class LabeledVariationalAutoencoder(nn.Module):
    def __init__(self, means, input_dim, latent_dim, n_epochs=10, hidden_sizes=[1000,500], lr=1e-3, opti='Adam', wd=0):
        super(LabeledVariationalAutoencoder, self).__init__()

        # Initialize parameters
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.learning_rate = lr
        self.means = means

        # Initialize networks
        self.encoder = LVAE_Encoder(input_dim, self.latent_dim, hidden_sizes=hidden_sizes).to(device)
        self.decoder = LVAE_Decoder(input_dim, self.latent_dim, hidden_sizes=hidden_sizes).to(device)

        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.loss = {'reconstruction':[],'kl0':[],'kl1':[],'total':[]}

        # Optimizers
        if opti=='Adadelta': self.optimizer = optim.Adadelta(self.parameters(), lr=self.learning_rate, weight_decay=wd)
        elif opti=='Adam': self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=wd)

    def forward(self, x, y):
        mu, log_var = self.encoder(x)
        sigma = th.exp(0.5*log_var).to(device)
        z = mu + sigma * th.distributions.Normal(0, 1).sample(mu.shape).to(device)
        kl0 = -0.5 * th.sum(1 + log_var[y==0] - (mu[y==0]-self.means[0]).pow(2) - log_var[y==0].exp(), dim=-1)
        kl1 = -0.5 * th.sum(1 + log_var[y==1] - (mu[y==1]-self.means[1]).pow(2) - log_var[y==1].exp(), dim=-1)
        return self.decoder(z.to(device)).to(device), kl0.mean(dim=0).to(device), kl1.mean(dim=0).to(device)

    def trainn(self, X, Y):
        self.train()
        for epoch in tqdm(range(1, self.n_epochs + 1)):
            batch_x = th.Tensor(X.values).to(device)
            batch_y = th.Tensor(Y.values).to(device)

            self.optimizer.zero_grad()
            x_hat, KL0_loss, KL1_loss = self.forward(batch_x,batch_y)
            AE_loss = self.reconstruction_loss(batch_x,x_hat)
            Total_loss = AE_loss + 0.5 * (KL0_loss + KL1_loss)
            Total_loss.backward()
            self.optimizer.step()
            self.loss['reconstruction'].append(AE_loss.item())
            self.loss['kl0'].append(KL0_loss.item())
            self.loss['kl1'].append(KL1_loss.item())
            self.loss['total'].append(Total_loss.item())
        self.eval()


def plot_latent_vectors(AE, AE_type, datasets_x, datasets_y, ax):

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    name = ['Train','Test','Val.ext S=75','Val.ext S=50']

    for n,(data_x,data_y) in enumerate(zip(datasets_x,datasets_y)):

        if AE_type == 'initial' : 
            if n==0 :
                pca = PCA(n_components = 2)
                pca.fit(data_x)
            batch_z = th.FloatTensor(pca.transform(data_x))
        elif AE_type == 'LVAE' : 
            batch_z, _ = AE.encoder(th.FloatTensor(data_x.values))
            if batch_z.shape[1]>2 :
                if n==0 :
                    pca = PCA(n_components = 2)
                    pca.fit(batch_z.flatten(1).detach().cpu().numpy())
                batch_z = th.FloatTensor(pca.transform(batch_z.flatten(1).detach().cpu().numpy()))

        plot_data = {}
        for i, z in enumerate(batch_z):
            if data_y.iloc[i] not in plot_data:
                plot_data[data_y.iloc[i]] = {'x':[], 'y':[]}
            plot_data[data_y.iloc[i]]['x'].append(z[0].item())
            plot_data[data_y.iloc[i]]['y'].append(z[1].item())

        for label, data in plot_data.items():
            ax[n].scatter(data['x'], data['y'], c=colors[label], label=label, edgecolors='none',s=2)
            if AE_type == 'initial' : ax[n].set_title(name[n])
            if n==0 : ax[n].set_ylabel(AE_type)


def plot_reconstructed_data(model,obs):

    d = int(obs.shape[0]**0.5)
    recon = model.decoder(model.encoder(th.FloatTensor(obs.values).reshape(1,-1))[0]).detach().cpu().numpy()

    obs_reshaped = obs[:d*d].values.reshape(d,d)
    recon_reshaped = recon[:,:d*d].reshape(d,d)

    fig,axs = plt.subplots(1,2)
    axs[0].imshow(obs_reshaped)
    axs[0].set_title('initial')
    axs[1].imshow(recon_reshaped)
    axs[1].set_title('reconstructed')
    plt.show()






