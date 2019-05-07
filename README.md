# THEx Background
Source code contributing to the research of the Transient Host Exchange project (THEx) project at the University of Arizona. THEx aims to predict astronomical transients for the Large Synoptic Survey Telescope (LSST) before they occur, in order to expedite their follow up once detected by LSST, by predicting transients using host-galaxy information alone.


# Set-Up
1. Set up the following directory structure (assuming you have the data in the corresponding FITs file):
```
thex_project
└───thex_code
│
└───data
│   │   THEx-training-set.v0_0_1.fits
```
2. Use the following commands to clone this repository and run the install script. The Jupyter Notebook interfaces will be automatically loaded.
```
cd thex_code
git clone https://github.com/marinakiseleva/thex_model.git
sh install.sh
```
<!-- Set up thex_model using Python virtualenv with the following commands (ensure that you have [HMC](https://github.com/marinakiseleva/hmc) downloaded into a neighboring directory called 'libraries'.)

```
mkdir libraries
cd libraries
git clone https://github.com/marinakiseleva/hmc.git
python setup.py install
cd ..
git clone https://github.com/marinakiseleva/thex_model.git
mkdir environments
virtualenv environments/thex_env
source environments/thex_env/bin/activate
cd libraries/hmc
python setup.py install
cd ../../thex_model
pip install -r requirements.txt
python setup.py develop
python -m ipykernel install --user --name thexenv --display-name "THEx env (py3env)"

``` -->

<!-- Update LOCAL_DATA_PATH in [thex_data/data_consts.py](thex_data/data_consts.py) with the path to data FITS file  (relative to thex_model root dir). It is best to follow this structure: -->
3. After installation, ensure the structure of the project looks like this:
```
thex_project
└───thex_code
|    └───libraries
|    └───environments
|    └───thex_model
│
└───data
│   │   THEx-training-set.v0_0_1.fits
```
4. When you are done developing/running models you may exit virtualenv with the following command.
```
deactivate
```

# Running
Please use the Jupyter Notebook [THEx Model Intro](notebooks/THEx%20Model%20Intro.ipynb) located in the notebooks directory to help you get started with running the models.

# Dependencies
This requires you have the following already installed:
- Python 3
- virtualenv
<!-- Listed in requirements.txt and the following that needs to be separately installed in another directory.
- [hmc](https://github.com/marinakiseleva/hmc) -- see above

Do not pip install hmc. Download it from the link above and install it using setup.py. This is a forked and edited version, and only this version will work with our project. -->

# Project Structure
This module is broken up into smaller modules that each provide different utilities and are described below.

## models
Contains the different classifiers explored/used in the project.

### mc_kde_model
Multiclass Kernel Density Estimate (KDE) model. Can be run either 'naively' (assuming feature independence and creating distinct distributions per feature per class) or non-naively, which creates a single distribution over all features. The 'mc' implies Multiclass, which functions by creating a class vector of 0s and 1s for each sample, and creating separate KDEs for each class.
<!-- ### hmc_model
Decisioning tree using the Hierarchical Multi-label Decisioing Tree from Vens, et al. 2008. -->

### clus_hmc_ens_model
Decisioning tree using the Hierarchical Multi-label Decisioing Tree using bagging and variance based on class vector (CLUS-HMC-ENS) from Schietgat, Vens, Struyf, et al. 2010. Uses class vectors similar to MC KDE.

### ktrees_model
Multiclass model with an ensemble of decision trees. Creates decision tree for each class, minimizing the Brier score. A probability is assigned for each class, for each sample independently.

## thex_data
Data pulling, cleansing, normalizing, preparation, and plotting.


# Acknowledgments
This project uses a [fork](https://github.com/marinakiseleva/hmc) of [HMC](https://github.com/davidwarshaw/hmc), available by the BSD 3-Clause "New" or "Revised" License.  
