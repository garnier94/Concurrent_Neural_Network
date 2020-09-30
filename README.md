# Concurrent_Neural_Network

## Purpose & general Presentation

This code represent supplementary material for the paper Modelisation of competition between times series currently under review. 
To sum up, it present a embedding of a neural network $f$. This embedding rescale its output to a given neural network to match a given sum. The i-th component of its output is then 

$$ y_i = \frac{f_i(x)}{\sum_{j} f_j(x)} $$ 

## Requirements
