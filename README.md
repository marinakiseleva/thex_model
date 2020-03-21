# THEx Background
Source code contributing to the research of the Transient Host Exchange project (THEx) project at the University of Arizona. THEx aims to predict astronomical transients for the Large Synoptic Survey Telescope (LSST) before they occur, in order to expedite their follow up once detected by LSST, by predicting transients using host-galaxy information alone.


# Set-Up
1. Set up the following directory structure (with the dataset as a FITs file in the data directory):
```
thex_project
└───thex_code
│
└───data
│   │   THEx-dataset.fits
```
2. Use the following commands to clone this repository and run the install script. The Jupyter Notebook interfaces will be automatically loaded.
```
cd thex_code
git clone https://github.com/marinakiseleva/thex_model.git
cd thex_model
sh install.sh
```


3. After installation, ensure that virtualenv has  been activated ((thex_env) should be prepended to your shell prompt) and the structure of the project looks like this:
```
thex_project
└───thex_code
|    └───libraries
|    └───environments
|    └───thex_model
│
└───data
│   │   THEx-dataset.fits
```
4. When you are done developing/running models you may exit virtualenv with the following command.
```
deactivate
```

# Running
Use the Jupyter Notebook [THEx Model Intro](notebooks/THEx%20Model%20Intro.ipynb) located in the notebooks directory to help you get started with running the models. Be sure to use the correct environment with the notebook.

# Dependencies
This module requires you have the following already installed:
- Python 3.6
- virtualenv (be sure virtualenv uses 3.6 by default)
<!-- Listed in requirements.txt and the following that needs to be separately installed in another directory. -->
- hmc fork available here: (https://github.com/marinakiseleva/hmc) 

Note: Do not pip install hmc. Download it from the link above and install it using setup.py. This is a forked and edited version, and only this version will work with our project.

# Project Structure
This module is broken up into smaller modules that each provide different utilities and are described below.

## classifiers
Contains the different classifiers explored/used in the project. 

## mainmodel
Abstract class which all models are built on. 

## models
Directory that contains existing models - which differ based on the underlying computation of probabilities. Each uses kernel density estimation per class.

### binary_model
Binary Classifiers: Treats each class as a separate binary classification problem and reports probability of each class versus all other classes. A multivariate Kernel Density Estimate (KDE) is estimated for each class, and for all the samples not with that class. 

### ind_model
Ensemble Classifier: Same as binary model except resulting probabilities are normalized together to get multiclass probabilities. 

### multi_model
KDE Multiclass Classifier: Multiclass classifier like the previous one, except now KDEs are created for each class, and probabilities are computed over those positive-class KDEs. The negative class space is not fitted.


## thex_data
Data pulling, cleansing, normalizing, preparation, and plotting.


# Acknowledgments
This project uses a [fork](https://github.com/marinakiseleva/hmc) of [HMC](https://github.com/davidwarshaw/hmc), available by the BSD 3-Clause "New" or "Revised" License.  
