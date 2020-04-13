# PsNano
A Predictor for RNA Pseudouridine sites in Oxford Nanopore RNA Sequencing reads

# About PsNano
PsNano is a predictor to identify Pseudouridine sites presented in direct RNA sequencing reads. 
PsNano will extract a set of features from the raw signal extracted from Oxford Nanopore RNA Sequencing reads and the corresponding basecalled kmer, which will be in turn used to predict whether the signal is modified by the presence of pseudouridine sites or not. 
Features extracted include:
- k-mer one hot encoding
- signal current intensity
- signal length
- signal statistical features including max, mini, mean, median and standard deviation of the signal.
The predictor has been trained and tested upon a set of 'unmodified' and 'modified' sequences containing Pseudouridine at known sites or uridine. 
The predictor can be adopted to detect other RNA modifications which has not yet been tested.

# Considerations when using this predictor:
Current trained machine learning models of PsNano are Support Vector Machine (SVM), Neural Network (NN), and Random Forest (RF). 
Those models will only be accurate if the data is for hela cell line and has been base-called with Albacore 2.1.0.
Training new models for data of different cell lines that is base-called using other basecallers has not been yet tested. 

# What's included?

- data folder that include two datasets. The first is the control hela that represents the unmodified signal extracted from Nanopore RNA sequence reads, while the second is the post_epi_hela that represents the modified signal. Both datasets should be feed to the predictors in this order.
- SVM.py, NN.py, and RF.py  are python scripts for SVM, NN, and RF predictors respectively. Each script extracts features from modified/unmodified signals and use those features to train the classifier and test it to predict RNA Pseudouridine modifications. 
- pseudoExtractor.py is a python script for exploring the data folder and pass the datasets to the predictor. 
- signalExtractor.py is python script that returns the modified/unmodified signal into a padded form in order to ensure that all signals in the dataset have the same length. This is important if we are going to use signal intensity as a feature for training the predictor.
- plot_learning_curves.py is a python script for plotting the learning curve for each predictor.

# Getting Started and pre-requisites

The following softwares and modules were used by PsNano

Software			      Version   

python				      3.7.4

numpy				        1.18.1

pandas				      1.0.2

sklearn				      0.22.2.post1

tensorflow			    2.0.0

tensorflow.keras		2.2.4-tf

Anaconda3 			    2019.10

# Running the predictor
- To train SVM and perform predictions:
This step includes SVM training, prediction and performance assessment using the features that lead to best performance.

$ python SVM.py 

- Similarly, to train NN and perform predictions:

$ python NN.py 

- Finally to train RF and perform predictions:

$ python RF.py

# Authors:
Daniel Acevedo and Doaa Hassan

