# Concurrent_Neural_Network

## Purpose & general Presentation

This code represent supplementary material for the paper Concurrent neural network: a model of competition between times series ( http://link.springer.com/article/10.1007/s10479-021-04253-3 ), published in Annals of Operations Research. 
To sum up, it present a embedding of a neural network f. This embedding rescale its output to a given neural network to match a given sum. The i-th component of its output is then :

```math 
y_i = \frac{f_i(x)}{\sum_{j} f_j(x)} 
``` 
This allows us to train this neural network for predicting market share of competing assets.

## Requirements
This code was tested using Python 3.7. The following packages were used for the core package 
- numpy==1.18.5
- torch==1.8.1
- pandas

sklearn was used to provide benchmark.

## Organisation of the code

- Concurrent_Neural_Network : contains the main source code for the project
  -   models.py : definition of the main Concurrent_Module module presented in the article
  -   submodel.py : definition of the subneural network Multi_layer_feed_forward_model
  -   preprocessing.py : various tools used to preprocessed data
- example_notebook.ipynb :  Jupyter notebook showing a simple example of use of the module
- example : outdated example of use of preprocessing tools
- data : A small sample of data used in example
