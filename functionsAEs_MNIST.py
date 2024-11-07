import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.decomposition import PCA
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from pythae.data.datasets import DatasetOutput
from pythae.models import VAE, VAEConfig
from pythae.models import AE, AEConfig
from pythae.trainers import BaseTrainer, BaseTrainerConfig
from pythae.trainers.training_callbacks import TrainingCallback, TrainHistoryCallback
from pythae.models.nn import BaseEncoder, BaseDecoder, BaseMetric
from pythae.models.base.base_utils import ModelOutput


#-----------------------------------------
# TOOLS
#-----------------------------------------

class ConfigData(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index][0]
        return DatasetOutput(data=x)


def plot_latent_vectors(AE, AE_type, datasets_x, datasets_y, ax):

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    name = ['Train','Test','Val.ext S=75','Val.ext S=50']

    for n,(data_x,data_y) in enumerate(zip(datasets_x,datasets_y)):

        if AE_type == 'initial' : 
            if n==0 :
                pca = PCA(n_components = 2)
                pca.fit(data_x.flatten(1).detach().cpu().numpy())
            batch_z = torch.FloatTensor(pca.transform(data_x.flatten(1).detach().cpu().numpy()))
        if AE_type == 'LAAE' : 
            batch_z = AE.encoder(data_x)
        elif AE_type == 'LVAE' : 
            batch_z, _ = AE.encoder(data_x)
        elif AE_type in ['VAE','AE'] : 
            batch_z = AE.encoder(data_x)['embedding']

        # print(batch_z.shape)

        plot_data = {}
        for i, z in enumerate(batch_z):
            if data_y[i].item() not in plot_data:
                plot_data[data_y[i].item()] = {'x':[], 'y':[]}
            # print(plot_data)
            # print(data_y[i].item())
            plot_data[data_y[i].item()]['x'].append(z[0].item())
            plot_data[data_y[i].item()]['y'].append(z[1].item())

        for label, data in plot_data.items():
            ax[n].scatter(data['x'], data['y'], c=colors[label], label=label, edgecolors='none',s=2)
            if AE_type == 'initial' : ax[n].set_title(name[n])
            if n==0 : ax[n].set_ylabel(AE_type)
        # ax.legend()
        # return plot_data

def plot_reconstructed_data(AE, AE_type, datasets_x, sf):

    name = ['Train','Test','Val.ext S=75','Val.ext S=50']

    for n,data_x in enumerate(datasets_x):

        if AE_type == 'initial' : 
            batch_recon = data_x[:9].detach().cpu().numpy()#torch.FloatTensor(pca.transform(data_x.flatten(1).detach().cpu().numpy()))
        if AE_type == 'LAAE' : 
            batch_recon = AE.decoder(AE.encoder(data_x[:9])).detach().cpu().numpy()
        elif AE_type == 'LVAE' : 
            batch_recon = AE.decoder(AE.encoder(data_x[:9])[0]).detach().cpu().numpy()
        elif AE_type in ['VAE','AE'] : 
            batch_recon = AE.decoder(AE.encoder(data_x[:9])['embedding'])['reconstruction'].detach().cpu().numpy()

        if AE_type=='initial' : sf[n].suptitle(name[n])
        if n == 0 : sf[n].supylabel(AE_type)
        axs = sf[n].subplots(3, 3)
        for innerind, ax in enumerate(axs.flat):
            ax.imshow(batch_recon[innerind].reshape(28,28), cmap='Greys',vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])


#-----------------------------------------
# AUTOENCODER
#-----------------------------------------

class AE_Encoder(BaseEncoder):
    def __init__(self, args=None): # Args is a ModelConfig instance
        BaseEncoder.__init__(self)

        self.latent_dim = args.latent_dim

        self.layers = nn.Sequential(*[nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                                       nn.ReLU(),
                                       nn.Dropout(0.2),
                                       nn.Flatten(),
                                       nn.Linear(64 * 7 * 7, 128),
                                       nn.ReLU(),
                                       ])

        self.embedding = nn.Linear(128,self.latent_dim)

    def forward(self, x:torch.Tensor): # -> ModelOutput:
        out = self.layers(x)
        output = ModelOutput(
            embedding=self.embedding(out)
        )
        return output

class AE_Decoder(BaseDecoder):
    def __init__(self, args=None):
        BaseDecoder.__init__(self)

        self.latent_dim = args.latent_dim

        self.layers = nn.Sequential(*[nn.Linear(self.latent_dim, 128),
                                      nn.ReLU(),
                                      nn.Linear(128,64 * 7 * 7),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Unflatten(1,(64,7,7)),
                                      nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
                                      ])

    def forward(self, z:torch.Tensor): # -> ModelOutput:
        output = ModelOutput(reconstruction=self.layers(z)) # Set the output from the decoder in a ModelOutput instance
        return output



#-----------------------------------------
# VARIATIONAL AUTOENCODER
#-----------------------------------------

class VAE_Encoder(BaseEncoder):
    def __init__(self, args=None): # Args is a ModelConfig instance
        BaseEncoder.__init__(self)

        self.latent_dim = args.latent_dim

        self.layers = nn.Sequential(*[nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                                       nn.ReLU(),
                                       nn.Dropout(0.2),
                                       nn.Flatten(),
                                       nn.Linear(64 * 7 * 7, 128),
                                       nn.ReLU(),
                                       ])

        self.embedding = nn.Linear(128,self.latent_dim)
        self.log_var = nn.Linear(128,self.latent_dim)

    def forward(self, x:torch.Tensor): # -> ModelOutput:
        out = self.layers(x)
        output = ModelOutput(
            embedding=self.embedding(out), # Set the output from the encoder in a ModelOutput instance
            log_covariance=self.log_var(out)
        )
        return output

class VAE_Decoder(BaseDecoder):
    def __init__(self, args=None):
        BaseDecoder.__init__(self)

        self.latent_dim = args.latent_dim

        self.layers = nn.Sequential(*[nn.Linear(self.latent_dim, 128),
                                      nn.ReLU(),
                                      nn.Linear(128,64 * 7 * 7),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Unflatten(1,(64,7,7)),
                                      nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
                                      ])

    def forward(self, z:torch.Tensor): # -> ModelOutput:
        output = ModelOutput(reconstruction=self.layers(z)) # Set the output from the decoder in a ModelOutput instance
        return output


#-----------------------------------------
# LABELED ADVERSARIAL AUTOENCODER
#-----------------------------------------


class LabeledAdversarialAutoencoder(nn.Module):
    def __init__(self, dataloader, latent_dim=2, batch_size=64, n_epochs=1000):
        super(LabeledAdversarialAutoencoder, self).__init__()

        self.train_loader = dataloader

        # Parameters
        self.z_dim = latent_dim
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.covariance_matrix = np.diag([1.,1.])
        self.n_labels = 10
        self.learning_rate = 0.001

        # Initialize networks
        self.encoder = nn.Sequential(*[nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Flatten(),
                                        nn.Linear(64 * 7 * 7, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, self.z_dim)
                                        ])

        self.decoder = nn.Sequential(*[nn.Linear(self.z_dim, 128),
                                       nn.ReLU(),
                                       nn.Linear(128,64 * 7 * 7),
                                       nn.ReLU(),
                                       nn.Dropout(0.2),
                                       nn.Unflatten(1,(64,7,7)),
                                       nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                                       nn.ReLU(),
                                       nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
                                       ])

        self.discriminator = nn.Sequential(*[nn.Linear(self.z_dim + 1, 50),
                                             nn.ReLU(),
                                             nn.Linear(50,50),
                                             nn.ReLU(),
                                             nn.Linear(50,1),
                                             ])

        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.discriminator_loss = nn.BCEWithLogitsLoss()
        self.encoder_loss = nn.BCEWithLogitsLoss()
        self.loss = {'reconstruction':[],'discriminator':[],'encoder':[]}

        # Optimizers
        self.autoencoder_optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.learning_rate)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)

    def generate_sample_prior(self, labels):
        mean = {}
        stdev = {}
        for label in np.arange(self.n_labels):
            angle = (2 * np.pi * label) / self.n_labels
            mean[label] = [2 * self.n_labels * np.cos(angle), 2 * self.n_labels * np.sin(angle)]
            stdev[label] = self.covariance_matrix

        priors = None
        for label in labels:
            lab = label.item()
            z = np.random.multivariate_normal(mean[lab], stdev[lab]).reshape(1,-1)

            if priors is None:
                priors = z
            else:
                priors = np.concatenate((priors, z), axis=0)
        return torch.Tensor(priors).to(device)

    def forward(self, x, label, z_real_dist):
        latent_vector = self.encoder(x)
        reconstruction = self.decoder(latent_vector)
        score_real_prior = self.discriminator(torch.cat([z_real_dist, label.view(self.batch_size,1)], dim=1))
        score_fake_prior = self.discriminator(torch.cat([latent_vector, label.view(self.batch_size,1)], dim=1))
        return reconstruction.to(device), latent_vector.to(device), score_real_prior.to(device), score_fake_prior.to(device)

    def train(self):
        for epoch in tqdm(range(1, self.n_epochs + 1)):
            rec,dis,enc=[],[],[]
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(device) #.view(self.batch_size, -1)
                batch_y = batch_y.to(device)
                z_real_dist = self.generate_sample_prior(batch_y).to(device)

                # Train autoencoder
                for i in np.arange(1):
                    self.autoencoder_optimizer.zero_grad()
                    reconstruction, _, _, _ = self.forward(batch_x, batch_y, z_real_dist)
                    autoencoder_loss = self.reconstruction_loss(reconstruction, batch_x)
                    autoencoder_loss.backward()
                    self.autoencoder_optimizer.step()
                    rec.append(autoencoder_loss.item())

                # Train discriminator
                for i in np.arange(1):
                    self.discriminator_optimizer.zero_grad()
                    _, _, score_real_prior, score_fake_prior = self.forward(batch_x, batch_y, z_real_dist)
                    real_labels = torch.ones(self.batch_size, 1).to(device)
                    fake_labels = torch.zeros(self.batch_size, 1).to(device)
                    discriminator_loss = 0.5 * (self.discriminator_loss(score_real_prior, real_labels) +
                                               self.discriminator_loss(score_fake_prior, fake_labels))
                    discriminator_loss.backward()
                    self.discriminator_optimizer.step()
                    dis.append(discriminator_loss.item())

                # Train encoder
                for i in np.arange(5):
                    self.encoder_optimizer.zero_grad()
                    _, _, _, score_fake_prior = self.forward(batch_x, batch_y, z_real_dist)
                    generator_loss = self.encoder_loss(score_fake_prior, real_labels)
                    generator_loss.backward()
                    self.encoder_optimizer.step()
                    enc.append(generator_loss.item())

            self.loss['reconstruction'].append(np.array(rec).mean())
            self.loss['discriminator'].append(np.array(dis).mean())
            self.loss['encoder'].append(np.array(enc).mean())



#-----------------------------------------
# LABELED VARIATIONAL AUTOENCODER
#-----------------------------------------

class LVAE_Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(LVAE_Encoder, self).__init__()

        self.latent_dim = latent_dim

        self.layers = nn.Sequential(*[nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                                       nn.ReLU(),
                                       nn.Dropout(0.2),
                                       nn.Flatten(),
                                       nn.Linear(64 * 7 * 7, 128),
                                       nn.ReLU(),
                                       ])

        self.embedding = nn.Linear(128,self.latent_dim)
        self.log_var = nn.Linear(128,self.latent_dim)

    def forward(self, x:torch.Tensor):
        out = self.layers(x)
        return self.embedding(out),self.log_var(out)


class LVAE_Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(LVAE_Decoder, self).__init__()

        self.latent_dim = latent_dim

        self.layers = nn.Sequential(*[nn.Linear(self.latent_dim, 128),
                                      nn.ReLU(),
                                      nn.Linear(128,64 * 7 * 7),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Unflatten(1,(64,7,7)),
                                      nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
                                      ])

    def forward(self, z:torch.Tensor):
        return self.layers(z)


class LabeledVariationalAutoencoder(nn.Module):
    def __init__(self, dataloader, latent_dim, batch_size=64, n_epochs=10):
        super(LabeledVariationalAutoencoder, self).__init__()

        # Initialize data
        self.train_loader = dataloader

        # Initialize parameters
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = 1e-3
        self.n_labels = 10
        self.variance = np.array([1.,1.])

        # Initialize networks
        self.encoder = LVAE_Encoder(self.latent_dim).to(device)
        self.decoder = LVAE_Decoder(self.latent_dim).to(device)

        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.loss = {'reconstruction':[],'total':[]}
        for lab in range(self.n_labels):
            self.loss[f'kl{lab}'] = []

        # Optimizers
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        # Priors
        self.generate_prior()


    def generate_prior(self):
        means = {}
        var = {}
        for label in np.arange(self.n_labels):
            angle = (2 * np.pi * label) / self.n_labels
            means[label] = 2 * torch.FloatTensor([self.n_labels * np.cos(angle), self.n_labels * np.sin(angle)])
            var[label] = torch.FloatTensor(self.variance)

        self.means = means
        self.var = var


    def forward(self, x, y=None):
        mu, log_var = self.encoder(x)
        sigma = torch.exp(0.5*log_var).to(device)
        z = mu + sigma * torch.distributions.Normal(0, 1).sample(mu.shape).to(device)
        kl = {}
        if y!=None:
            for lab in np.arange(self.n_labels):
                to_mean = - 0.5 * torch.sum(1 + log_var[y==lab] - (mu[y==lab]-self.means[lab]).pow(2) - log_var[y==lab].exp(), dim=-1)
                kl[lab] = to_mean.mean(dim=0).to(device)
        return self.decoder(z.to(device)).to(device), kl


    def trainn(self):
        self.train()
        for epoch in tqdm(range(1, self.n_epochs + 1)):
            rec,tot,kl=[],[],{}
            for lab in range(self.n_labels):
                kl[f'kl{lab}']=[]
            for bat,(batch_x, batch_y) in enumerate(self.train_loader):
                batch_x = batch_x.to(device) #.view(-1,28,28)
                batch_y = batch_y.to(device)

                self.optimizer.zero_grad()
                x_hat, KLs = self.forward(batch_x,batch_y)
                KL_loss = sum(KLs.values())
                AE_loss = self.reconstruction_loss(batch_x,x_hat)
                Total_loss = AE_loss + KL_loss / self.n_labels

                Total_loss.backward()
                self.optimizer.step()
                rec.append(AE_loss.item())
                for lab in range(self.n_labels):
                    kl[f'kl{lab}'].append(KLs[lab].item())
                tot.append(Total_loss.item())
            self.loss['reconstruction'].append(np.array(rec).mean())
            for lab in range(self.n_labels):
                self.loss[f'kl{lab}'].append(np.array(kl[f'kl{lab}']).mean())
            self.loss['total'].append(np.array(tot).mean())
        self.eval()