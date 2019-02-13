# THEx Background
Source code contributing to the research of the Transient Host Exchange project (THEx) project at the University of Arizona. THEx aims to predict astronomical transients for the Large Synoptic Survey Telescope (LSST) before they occur, in order to expedite their follow up once detected by LSST, by predicting transients using host-galaxy information alone. 

# thex_model
This model is broken up into smaller modules that each provide different utilities and are described below.

## thex_data 
Data pulling, cleansing, normalizing, preparation, and plotting. 

## models
Contains the different classifiers explored/used in the project.

### tree_model
Decisioning tree using the Hierarchical Multi-label Decisioing Tree from Vens, et al. 2008. 

### nb_model
The Gaussian (or on-demand best-fitting distribution) Naive Bayes Model that runs on the code. 

## model_performance
Compares the existing models in this project.

# Running

The Naive Bayes or Tree programs can be run using specific column names:

> python run_classifier.py -cols PS1_gKmag PS1_rKmag PS1_iKmag PS1_zKmag PS1_yKmag

Or using a generic column name that will match on all relevant columns:

> python run_classifier.py -col_names PS1


# Set-Up
Acquire FITS data file from THEx. Download and install [HMC](https://github.com/marinakiseleva/hmc).
Update the following values in thex_data/data_consts.py:
- DATA_PATH : Path to FITS file  (relative to thex_model root dir)
- LIB_PATH : Path to HMC root  (relative to thex_model root dir)


# Dependencies
This module requires you to have the following libraries already installed:
- numpy
- pandas
- sklearn
- scipy
- astropy
- [hmc](https://github.com/marinakiseleva/hmc) -- see above



# Acknowledgments
This project uses a [fork](https://github.com/marinakiseleva/hmc) of [HMC](https://github.com/davidwarshaw/hmc), available by the BSD 3-Clause "New" or "Revised" License.  

