# THEx Background
Source code contributing to the research of the Transient Host Exchange project (THEx) project at the University of Arizona. THEx aims to predict astronomical transients for the Large Synoptic Survey Telescope (LSST) before they occur by using host-galaxy photometric data, in expedite their follow up once detected by LSST.


# Set-Up
1. Data: Make sure to have the correct data file installed at the location pointed to in thex_data.data_consts. Specifically, in your home directory /data/catalogs. 


2. Use the following commands to clone this repository and run the install script. Note, additional directories will be created in the current directory. See install.sh for more details. The Jupyter Notebook interfaces will be automatically loaded.
```
git clone https://github.com/marinakiseleva/thex_model.git
cd thex_model
sh install.sh
```


3. The install script sets up a virtual environment. After the script finishes, ensure that virtualenv has  been activated ((thex_env) should be prepended to your shell prompt) and the structure of the project looks like this:
```
thex_project
└───thex_code
     └───libraries
     └───environments
     └───thex_model

```
4. When you are done developing/running models you may exit virtualenv with the following command.
```
deactivate
```

# Running
Use the Jupyter Notebook (notebooks/Models.ipynb) located in the notebooks directory to help you get started with running the models. Be sure to use the correct environment with the notebook.

# Dependencies
This module requires you have the following versions of Python and virtualenv installed. We cannot guarantee it will work with older versions.
- Python 3.8.59
- virtualenv (be sure virtualenv uses 3.8.5 by default)

Note: The install script automatically downloads another dependency for you, hmc. This is a forked and edited version, and only this version will work with our project.

# Project Structure
This module is broken up into smaller modules that each provide different utilities and are described below.

## classifiers
Contains the different classifiers used in the project. 

## mainmodel
Abstract class which all models are built on. 

## models
Directory that contains existing models - which differ based on the underlying computation of probabilities. Each uses kernel density estimation per class.

### binary_model
Binary Classifiers: Treats each class as a separate binary classification problem and reports probability of each class versus all other classes. For each class, a multivariate Kernel Density Estimate (KDE) is estimated for the samples in that class, and a separate KDE for all the other samples not in the class. 

### ind_model
Ensemble Classifier: Same as binary model except resulting probabilities are normalized together to get multiclass probabilities. 

### multi_model
KDE Multiclass Classifier: Multiclass classifier like the previous one, except now KDEs are created for each class, and probabilities are computed over those positive-class KDEs. The negative class space is not fitted.


## thex_data
Data pulling, cleansing, normalizing, preparation, and plotting.


# Acknowledgments
This project uses a [fork](https://github.com/marinakiseleva/hmc) of [HMC](https://github.com/davidwarshaw/hmc), available by the BSD 3-Clause "New" or "Revised" License.  
