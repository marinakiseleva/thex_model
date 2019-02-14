# THEx Background
Source code contributing to the research of the Transient Host Exchange project (THEx) project at the University of Arizona. THEx aims to predict astronomical transients for the Large Synoptic Survey Telescope (LSST) before they occur, in order to expedite their follow up once detected by LSST, by predicting transients using host-galaxy information alone. 


# Set-Up
Set up thex_model using Python virtualenv with the following commands (ensure that you have [HMC](https://github.com/marinakiseleva/hmc) downloaded beforehand.)

```
mkdir environments
virtualenv environments/thex_env
source environments/thex_env/bin/activate
cd libraries/hmc
python setup.py install
cd ../../thex_model
pip install -r requirements.txt 
python setup.py develop
```
Update LOCAL_DATA_PATH in [thex_data/data_consts.py](thex_data/data_consts.py) with the path to data FITS file  (relative to thex_model root dir)

When you are finished, you may exit virtualenv with:
```
deactivate
```

<!-- The LOCAL_DATA_PATH and LOCAL_LIBS_PATH paths in thex_data/data_consts.py need to be updated with the data path and directory containing hmc, respectively. Acquire FITS data file from THEx. Download and install [HMC](https://github.com/marinakiseleva/hmc).
Update the following values in thex_data/data_consts.py:
- DATA_PATH : Path to FITS file  (relative to thex_model root dir)
- LIB_PATH : Path to HMC root  (relative to thex_model root dir) -->


# Running
Please use the Jupyter Notebook 'THEx Model Intro' located in the notebooks directory to help you get started with running the models.

# Dependencies
Listed in requirements.txt and the following that needs to be separately installed in another directory. 
- [hmc](https://github.com/marinakiseleva/hmc) -- see above

Do not pip install hmc. Download it from the link above and install it using setup.py. This is a forked and edited version, and only this version will work with our project.

# Project Structure
This module is broken up into smaller modules that each provide different utilities and are described below.

## thex_data 
Data pulling, cleansing, normalizing, preparation, and plotting. 

## models
Contains the different classifiers explored/used in the project. Contains the Hierarchical Decision Tree model and Naive Bayes model.

### tree_model
Decisioning tree using the Hierarchical Multi-label Decisioing Tree from Vens, et al. 2008. 

### nb_model
Naive Bayes Model using kernel density estimation to determine the feature distributions.

## model_performance
Evaluates the existing models in this project.


# Acknowledgments
This project uses a [fork](https://github.com/marinakiseleva/hmc) of [HMC](https://github.com/davidwarshaw/hmc), available by the BSD 3-Clause "New" or "Revised" License.  

