# thex_model
Source code for THEx transient prediction using host-galaxy information. Includes data pulling, cleansing, and preparation. Also has an implementation of a Gaussian Naive Bayes Classifier designed to predict astronomical transient events using galaxy data. 


## thex_data 
Contains all setup code for pulling in, normalizing, and cleansing data


## nb_model
The Gaussian Naive Bayes Model that runs on the code. Run like so:

python run_classifier.py -cols PS1_gKmag PS1_rKmag PS1_iKmag PS1_zKmag PS1_yKmag
