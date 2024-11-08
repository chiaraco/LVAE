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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from pythae.data.datasets import DatasetOutput
from pythae.models import VAE, VAEConfig
from pythae.models import AE, AEConfig
from pythae.trainers import BaseTrainer, BaseTrainerConfig
from pythae.trainers.training_callbacks import TrainingCallback, TrainHistoryCallback
from pythae.models.nn import BaseEncoder, BaseDecoder, BaseMetric
from pythae.models.base.base_utils import ModelOutput
import logging

from functionsAEs_MNIST import *

class multiply_by:
    def __init__(self, s=None):
        self.s = s

    def __call__(self, x):
        if self.s is None:
            return x * 255
        else:
            return x * (self.s - 1) + 1

# Exemple de collate_fn si tu veux ajuster les batches
def my_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)  # Créer un tensor du batch d'images
    labels = torch.tensor(labels)  # Créer un tensor des labels
    return images, labels

def train():
    ######### create datasets
    # train
    train = datasets.MNIST('./Data', train=True, download=True, 
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Lambda(multiply_by(None))  # Applique la transformation
                                    ]))
    train_dataloader = DataLoader(train,batch_size=len(train),shuffle=False)
    for x,y in train_dataloader: train_x,train_y = x,y
    
    batch_size = 64#512
    epochs = 10
    # train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=my_collate_fn, drop_last=True)

    # test
    test = datasets.MNIST('./Data', train=False, download=True, 
                           transform=transforms.Compose([transforms.ToTensor(),
                                                         transforms.Lambda(multiply_by(None))]))
    test_dataloader = DataLoader(test,batch_size=len(test),shuffle=False)
    for x,y in test_dataloader: test_x,test_y = x,y
    
    # external validation
    s=75
    testS75 = datasets.MNIST('./Data', train=False, download=True, 
                           transform=transforms.Compose([transforms.ToTensor(),
                                                         transforms.Lambda(multiply_by(s))]))
    testS75_dataloader = DataLoader(testS75,batch_size=len(testS75),shuffle=False)
    for x,y in testS75_dataloader: testS75_x,testS75_y = x,y
    
    s=50
    testS50 = datasets.MNIST('./Data', train=False, download=True, 
                           transform=transforms.Compose([transforms.ToTensor(),
                                                         transforms.Lambda(multiply_by(s))]))
    testS50_dataloader = DataLoader(testS50,batch_size=len(testS50),shuffle=False)
    for x,y in testS50_dataloader: testS50_x,testS50_y = x,y
    
    
    
    
    
    # ------------------------------------
    # AE training
    # ------------------------------------
    
    train_data = ConfigData(train)
    test_data = ConfigData(test)
    
    logger = logging.getLogger(__name__)
    console = logging.StreamHandler()
    logger.addHandler(console)
    logger.setLevel(logging.INFO)
    
    ## model configuration
    model_config = AEConfig(latent_dim=2)
    encoder = AE_Encoder(model_config)
    decoder = AE_Decoder(model_config)
    model_AE = AE(model_config=model_config, encoder=encoder, decoder=decoder)
    
    training_config = BaseTrainerConfig(
        num_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=1e-3,
        optimizer_cls="Adam",
        optimizer_params={},
        scheduler_cls="MultiStepLR",
        scheduler_params={"milestones": [10, 20, 30], "gamma": 10**(-1/5)}
        )
    
    callbacks = []
    loss_cb = TrainHistoryCallback()
    callbacks.append(loss_cb)
    
    trainer = BaseTrainer(
        model=model_AE,
        train_dataset=train_data,
        eval_dataset=test_data,
        training_config=training_config,
        callbacks=callbacks)
    
    trainer.train()
    
    
    
    #------------------------------------
    # VAE training
    #------------------------------------
    
    train_data = ConfigData(train)
    test_data = ConfigData(test)
    
    logger = logging.getLogger(__name__)
    console = logging.StreamHandler()
    logger.addHandler(console)
    logger.setLevel(logging.INFO)
    
    ## model configuration
    model_config = VAEConfig(latent_dim=2)
    encoder = VAE_Encoder(model_config)
    decoder = VAE_Decoder(model_config)
    model_VAE = VAE(model_config=model_config, encoder=encoder, decoder=decoder)
    
    training_config = BaseTrainerConfig(
        num_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=1e-3,
        optimizer_cls="Adam",
        optimizer_params={},
        scheduler_cls="MultiStepLR",
        scheduler_params={"milestones": [10, 20, 30], "gamma": 10**(-1/5)}
        )
    
    callbacks = []
    loss_cb = TrainHistoryCallback()
    callbacks.append(loss_cb)
    
    trainer = BaseTrainer(
        model=model_VAE,
        train_dataset=train_data,
        eval_dataset=test_data,
        training_config=training_config,
        callbacks=callbacks)
    
    trainer.train()
    
    
    #------------------------------------
    # LAAE training
    #------------------------------------
    
    model_LAAE = LabeledAdversarialAutoencoder(train_loader,2,batch_size,epochs).to(device)
    model_LAAE.train()
    
    
    
    #------------------------------------
    # LVAE training
    #------------------------------------
    
    model_LVAE = LabeledVariationalAutoencoder(train_loader,2,batch_size,epochs).to(device)
    model_LVAE.trainn()
    
    
    
    #------------------------------------
    # Latent space visualisation
    #------------------------------------
    
    fig,axs = plt.subplots(5,4,figsize=(12,10))
    
    plot_latent_vectors(None, 'initial', [train_x,test_x,testS75_x,testS50_x], [train_y,test_y,testS75_y,testS50_y], axs[0])
    
    plot_latent_vectors(model_AE, 'AE', [train_x,test_x,testS75_x,testS50_x], [train_y,test_y,testS75_y,testS50_y], axs[1])
    
    plot_latent_vectors(model_VAE, 'VAE', [train_x,test_x,testS75_x,testS50_x], [train_y,test_y,testS75_y,testS50_y], axs[2])
    
    plot_latent_vectors(model_LAAE, 'LAAE', [train_x,test_x,testS75_x,testS50_x], [train_y,test_y,testS75_y,testS50_y], axs[3])
    
    plot_latent_vectors(model_LVAE, 'LVAE', [train_x,test_x,testS75_x,testS50_x], [train_y,test_y,testS75_y,testS50_y], axs[4])
    
    plt.title('Latent space visualisation')
    plt.tight_layout()
    plt.show()
    
    
    
    #------------------------------------
    # Reconstruction visualisation
    #------------------------------------
    
    fig = plt.figure(constrained_layout=True,figsize=(12,10))
    subfigs = fig.subfigures(5, 4, hspace=0.1, wspace=0.05)
    
    plot_reconstructed_data(None, 'initial', [train_x,test_x,testS75_x,testS50_x], subfigs[0])
    
    plot_reconstructed_data(model_AE, 'AE', [train_x,test_x,testS75_x,testS50_x], subfigs[1])
    
    plot_reconstructed_data(model_VAE, 'VAE', [train_x,test_x,testS75_x,testS50_x], subfigs[2])
    
    plot_reconstructed_data(model_LAAE, 'LAAE', [train_x,test_x,testS75_x,testS50_x], subfigs[3])
    
    plot_reconstructed_data(model_LVAE, 'LVAE', [train_x,test_x,testS75_x,testS50_x], subfigs[4])
    
    plt.title('Reconstruction visualisation')
    plt.tight_layout()
    plt.show()
    
    
    
    #------------------------------------
    # Prediction performances
    #------------------------------------
    
    res_rf = pd.DataFrame(index=['initial','AE','VAE','LAAE','LVAE'],columns=pd.MultiIndex.from_product([[f'CV{i+1}' for i in range(5)],['Train','Test','S75','S50']]))
    
    for AE_type,model in zip(['initial','LVAE','AE','VAE','LAAE'],[None,model_LVAE,model_AE,model_VAE,model_LAAE]):
        for cv in range(5):
            print('\n\n',AE_type,cv,'\n')
    
            Xdata = {}
            Ydata = {}
            for name,data_x,data_y in zip(['Train','Test','S75','S50'],[train_x,test_x,testS75_x,testS50_x],[train_y,test_y,testS75_y,testS50_y]):
                if AE_type=='initial':
                    Xdata[name] = data_x.flatten(1).detach().cpu().numpy()
                    Ydata[name] = data_y.detach().cpu().numpy()
                if AE_type == 'LAAE' : 
                    Xdata[name] = model.encoder(data_x).detach().cpu().numpy()
                    Ydata[name] = data_y.detach().cpu().numpy()
                elif AE_type == 'LVAE' : 
                    Xdata[name] = model.encoder(data_x)[0].detach().cpu().numpy()
                    Ydata[name] = data_y.detach().cpu().numpy()
                elif AE_type in ['VAE','AE'] : 
                    Xdata[name] = model.encoder(data_x)['embedding'].detach().cpu().numpy()
                    Ydata[name] = data_y.detach().cpu().numpy()
    
            if cv==0:
                # param_grid_rf = {'n_estimators': [100, 500],'max_depth' : [2,10,None],'criterion' :['gini', 'entropy'],'min_samples_leaf': [2,5]}
                param_grid_rf = {'n_estimators': [10],'max_depth' : [2],'criterion' :['gini', 'entropy'],'min_samples_leaf': [2]}
                rf = GridSearchCV(RFC(n_jobs=-1), param_grid_rf,n_jobs=-1)
                rf.fit(Xdata['Train'],Ydata['Train'])
    
            for name in ['Train','Test','S75','S50']:
                res_rf.loc[AE_type,(f'CV{cv+1}',name)] = MCC(Ydata[name],rf.predict(Xdata[name]))
    
    
    res_rf.groupby(level=1, axis=1).mean().plot(kind='bar',yerr=res_rf.groupby(level=1, axis=1).std())
    plt.ylabel('MCC (mean of 5 CV)')
    plt.title('Prediction performances (Random Forest)')





if __name__ == "__main__":
    train()  # Appelle la fonction de formation dans le bloc "main"






