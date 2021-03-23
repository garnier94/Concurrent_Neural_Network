import pandas as pd
import numpy as np
import torch

def filter_index_from_dataframe(df, target_column, minimal_sum_target=100, minimal_positive_length=20):
    """
    Filter the dataframe
    :param df: pd.DataFrame indexed by [spatial_index, temporal_index]
    :param target_column: The target value of interest
    :param minimal_sum_target: A product must have at least minimal_sum to be considered
    :param minimal_positive_length: A product must have been sold during at least minimal_positive_length to be considered
    :return:
    """
    final_index_list = []
    final_sud_dataframe_list = []
    index_set = set(df.index.get_level_values(0))

    for index in index_set:
        sub_series = df.loc[index].fillna(0)
        if sum(sub_series[target_column]) > minimal_sum_target:
            min_date = min(sub_series[sub_series[target_column] > 0].index)
            max_date = max(sub_series[sub_series[target_column] > 0].index)
            sub_series = sub_series[sub_series.index >= min_date]
            sub_series = sub_series[sub_series.index <= max_date]
            if sub_series.shape[0] > minimal_positive_length:
                final_index_list.append(index)
                final_sud_dataframe_list.append(sub_series)
    return pd.concat(final_sud_dataframe_list, keys=final_index_list, names=['product']), final_index_list


def compute_proportion(data, target_column, resize=None):
    """

    :param data:  pd.DataFrame indexed by [spatial_index, temporal_index]
    :param target_column:  The target value of interest
    :param resize: if None, there is no resizing, and the proportion sum to one , otherwise to resize to sum the proportion
    :return:
    """
    product_list = set(data.index.get_level_values(0))
    sum_product = data.groupby(level=1)[target_column].sum()
    data = data.join(sum_product, rsuffix='_somme')
    alpha = 1
    if not resize is None:
        alpha = resize 
    data['proportion'] = alpha * data[target_column] / data[target_column + '_somme']
    return data, sum_product, product_list 


def add_temporal_features(data, target='proportion', horizon=6, shift_list=[0, 1]):
    """
    Add shifted data with shift in [horizon + c for c in shift_list]
    :param data: pd.DataFrame indexed by [spatial_index, temporal_index]
    :param target: The target value of interest to be shifted
    :param horizon: prediction horizon
    :param shift_list:
    :return:
    """
    for shift_i in shift_list:
        data[target + '_shift_' + str(shift_i + horizon)] = data.groupby(level=0)[target].shift(shift_i + horizon)
    return data[data[target + '_shift_' + str(shift_i + horizon)].notna()]

def dataframe_to_data_loader(data, features,target):
    """
    Tranform a pandas DataFrame into a data_loader suited for 
    data: pd.DataFrame indexed by [spatial_index, temporal_index]
    target : Target columns name in the dataframe
    features: Features used for learning
    """
    date_index = list(set(data.index.get_level_values(1)))#List of the date existing in the dataset
    date_index.sort()
    data_loader= []
    for dt in date_index:
        feat_dt = data[data.index.get_level_values(1)==dt ][features].values.astype(np.float32)
        target_dt = data[data.index.get_level_values(1)==dt ][target].values.astype(np.float32)
        data_loader.append([torch.tensor(feat_dt), torch.tensor(target_dt)])
    return data_loader