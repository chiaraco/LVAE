# Implementation of LVAE

## Installation
Before using LVAE, it is necessary to install requirements presented in [`requirements.yml`](requirements.yml) file.

If you are using `conda`, you can install an environment called `AEenv`. First, clone this repository ([Cloning a repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)), or download ([Downloading a repository](https://docs.github.com/en/get-started/start-your-journey/downloading-files-from-github)) and unpack it. 

Then, open an Anaconda prompt and run the following command :
```bash
> conda env create -f path\to\the\folder\requirements.yml  # path to the folder where the requirements.yml file is
```
( The installation can take more than 15 min)

The IDE is not included in the environment, so you can install one if necessary.
```bash
> conda activate AEenv
#if needed:
> conda install spyder  
> spyder 
```


## Usage 

The algorithm can be used as follows:

```python
#######################################################
##### import required functions

# You have to be in the folder where functionsLVAE.py is.

import pandas
import torch as th
import matplotlib.pyplot as plt
from functionsLVAE import *

#######################################################
##### load your data

# With your own datasets
#------------------------

# Train dataset
X = pandas.read_csv(...) # pandas DataFrame with observations in row, genes in column
Y = pandas.read_csv(...) # pandas Series with class to predict

# External dataset(s) - Y for external dataset can be loaded if available, to realize performance tests.
Xext1 = pandas.read_csv(...) # pandas DataFrame with observations in row, genes in column for external dataset 1
Xext2 = pandas.read_csv(...) # obs/gene dataframe for external dataset 2
# Optional:
Yext1 = pandas.read_csv(...) # pandas Series with class to predict for external dataset 1
Yext2 = pandas.read_csv(...) # class to predict for external dataset 2

# Categorical variables can be included with OneHotEncoder
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html


# With the included datasets (in the same folder than the one with functionsLVAE.py file)
#-------------------------------------------------------------------------------------------------

to_load = 'data'

data_train = pandas.read_csv(f'{to_load}/train.csv',header=0,index_col=0)
X = data_train.drop('Y',axis=1)
Y = data_train['Y'].copy()

data_valid = pandas.read_csv(f'{to_load}/valid.csv',header=0,index_col=0)
Xval1 = data_valid.drop('Y',axis=1)
Yval1 = data_valid['Y'].copy() # Y for validation is not required, but can be loaded in order to allow perfomance evaluation


#######################################################
##### run LVAE (choose parameters)

# With your own datasets
#------------------------

###### important note
# functionsLVAE.py is suitable for binary classification for tabular data whose priors are normal distributions and are different between classes only by their mean
# you can adapt this to your own data
# there is an example with MNIST data: suitable for multiclass, image and 2D reduction
######


# first, create a LabeledVariationalAutoencoder class with the chosen parameters and train the algorithm with your own data
model = LabeledVariationalAutoencoder(means=[-1,1], input_dim=X.shape[1], latent_dim=2, n_epochs=100, hidden_sizes=[1000,500], lr=1e-3, opti='Adam', wd=0)
model.trainn(X,Y)


# then, observe the latent space (if it is not 2D, a PCA step is added), and the quality of the reconstruction
fig,axs = plt.subplots(2,number_of_datasets,figsize=(12,10)) # number_of_datasets = 3
plot_latent_vectors(None, 'initial', datasets_x, datasets_y, axs[0]) # datasets_x = [X,Xext1,Xext2] / datasets_y = [Y,Yext1,Yext2]
plot_latent_vectors(model, 'LVAE', datasets_x, datasets_y, axs[1]) # datasets_x = [X,Xext1,Xext2] / datasets_y = [Y,Yext1,Yext2]
plt.tight_layout()
plt.show()

obs = X.iloc[0,]
plot_reconstructed_data(model, obs) # observe an initial observation and its reconstruction


# finally, you can add a classification step, trained on initial and reduced data, and compare their performances 
Xred = model.encoder(th.FloatTensor(X.values))[0] # the reduced data is obtained with this command


# With the included datasets (loaded above)
#-------------------------------------------------------------------------------------------------

model = LabeledVariationalAutoencoder(means=[-1,1], input_dim=X.shape[1], latent_dim=2, n_epochs=100, hidden_sizes=[1000,500], lr=1e-3, opti='Adam', wd=0)
model.trainn(X,Y)

fig,axs = plt.subplots(2,2,figsize=(12,10))
plot_latent_vectors(None, 'initial', [X,Xval1], [Y,Yval1], axs[0])
plot_latent_vectors(model, 'LVAE', [X,Xval1], [Y,Yval1], axs[1])
plt.tight_layout()
plt.show()

obs = X.iloc[0,]
plot_reconstructed_data(model, obs)


```

## Run an example with MNIST data testing several autoencoders (standard AE, variational AE, labeled adversarial AE and our labeled variational AE) 

For a complete running example on MNIST dataset, please see [scriptAEs_MNIST.py](scriptAEs_MNIST.py).
The code generates three plots two DataFrames with latent and reconstruction visualisation, and prediction performances (mean and standard deviation). 

To run the example code, activate the conda environment and execute the code from the root of the project:
```bash
> conda activate AEenv
> python scriptAEs_MNIST.py
```


## Transcriptomics data preprocessing
When using transcriptomics data, validation datasets are prealably homogenised with the train dataset using MatchMixeR algorithm (https://doi.org/10.1093/bioinformatics/btz974).
The script for homogenisation is available in [preprocessing.R](preprocessing.R) (with MatchMixer, the reference dataset (here, train dataset) is not modified, only the validation one).
Train and external validation datasets after homogenisation (including class to predict and standard prognostic variables) are available on Zenodo: [10.5281/zenodo.15210652](https://doi.org/10.5281/zenodo.15210652)

METABRIC dataset was initially downloaded from https://www.cbioportal.org/study/summary?id=brca_metabric (data_expression_median.txt).
Hatzis dataset was initally downloaded from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE25055  (.CEL download then mas5 normalization).
Saal dataset was initially downloaded from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96058.


# License

   Copyright 2024 INSTITUT DE CANCEROLOGIE DE L'OUEST (ICO) and UNIVERSITE ANGERS

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
