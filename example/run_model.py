import pandas as pd
import sys, configparser
from Concurrent_Neural_Network.preprocessing import filter_index_from_dataframe, compute_proportion,add_temporal_features
from path import Path

data = pd.read_csv( 'data_example.csv', sep=';')
CONFIG_CONTAINER = configparser.RawConfigParser()
CONFIG_CONTAINER.read(Path(__file__).parent / '../config.cfg')

resize = CONFIG_CONTAINER.getfloat('data', 'resizing')
min_number_product = CONFIG_CONTAINER.getint('data', 'min_number_product')
min_length = CONFIG_CONTAINER.getint('data', 'min_length')
print(min_length+min_number_product)

data,_ = filter_index_from_dataframe(data,'Vente lisse' , minimal_sum_target=min_number_product, minimal_positive_length=20)
data_restrained, sum_product, product_list = compute_proportion(data,'Vente lisse', resize=resize)
data_restrained = add_temporal_features(data_restrained)
