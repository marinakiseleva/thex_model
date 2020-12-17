# THEx Background
Source code contributing to the research of the Transient Host Exchange (THEx) at the University of Arizona. We classify astronomical transients using host-galaxy photometric data, so that transients detected by the Large Synoptic Survey Telescope (LSST) may be immediately followed-up on after their initial discovery based on their classified type.


# Set-Up

```
pip install -i https://test.pypi.org/simple/ thex-model==3.0
```

And install this version of hmc in the same environment, like so
```
git clone https://github.com/marinakiseleva/hmc.git
cd hmc
python setup.py install
```

Data: Make sure to have the correct data file installed at the location pointed to in thex_data.data_consts. Specifically, in your home directory /data/catalogs. 



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
One-vs-All (OVA) Classifier: Same as binary model except resulting probabilities are normalized together to get multiclass probabilities. 

### multi_model
KDE Multiclass Classifier: Multiclass classifier like the previous one, except now KDEs are created for each class, and probabilities are computed over those positive-class KDEs. The negative class space is not fitted.


## thex_data
Data pulling, cleansing, normalizing, preparation, and plotting.


# Acknowledgments
This project uses a [fork](https://github.com/marinakiseleva/hmc) of [HMC](https://github.com/davidwarshaw/hmc), available by the BSD 3-Clause "New" or "Revised" License.  
