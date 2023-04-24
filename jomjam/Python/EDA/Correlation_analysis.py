import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import xlrd
import graphviz
import seaborn as sns

############################################################################ Data import
df = pd.read_excel('data_200416.xlsx')
print(df.shape)
inputs = df.copy()
inputs[cat_col] = inputs[cat_col].fillna('nov')
inputs[con_col] = inputs[con_col].fillna(0)
inputs[cat_col] = inputs[cat_col].astype('str')
inputs[con_col] = inputs[con_col].astype('float64')


############################################################################ Def. Collection
import scipy.stats as ss
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

import math
from collections import Counter

def replace_nan_with_value(x, y, value):
    x = np.array([v if v == v and v is not None else value for v in x])  # NaN != NaN
    y = np.array([v if v == v and v is not None else value for v in y])
    return x, y


def conditional_entropy(x,
                        y,
                        nan_strategy="replace",
                        nan_replace_value=0.0,
                        log_base: float = math.e):
    if nan_strategy == "replace":
        x, y = replace_nan_with_value(x, y, nan_replace_value)
    elif nan_strategy == _DROP:
        x, y = remove_incomplete_samples(x, y)
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y / p_xy, log_base)
    return entropy

def theils_u(x, y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x
        
def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta
    
############################################################################ Cramers_V

col_list = list(inputs.columns)
tot_list=[]
for col_1 in col_list:
    raw_list=[]
    for col_2 in col_list:
        x_val = inputs[col_1].values
        y_val = inputs[col_2].values
        if (col_1 in cat_col)&(col_2 in cat_col):
            raw_list.append(cramers_v(x_val,y_val))
        elif (col_1 in con_col)&(col_2 in con_col):
            raw_list.append(np.correlate(x_val,y_val))
        elif (col_1 in cat_col)&(col_2 in con_col):
            raw_list.append(correlation_ratio(x_val,y_val))
        elif (col_1 in con_col)&(col_2 in cat_col):
            raw_list.append(correlation_ratio(y_val,x_val))
    tot_list.append(raw_list)

tot_arr = np.array(tot_list, dtype=np.float32)
cramers_df = pd.DataFrame(tot_arr, columns=col_list, index=col_list)

fig, ax = plt.subplots( figsize=(25,25) )

mask = np.zeros_like(cramers_df, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

############################################################################ Vis

# 히트맵을 그린다
sns.heatmap(cramers_df, 
            cmap = 'RdYlBu_r', 
            annot = True,
            mask=mask,
            linewidths=.5,
            cbar_kws={"shrink": .5},
            vmin = -1,vmax = 1
           )  
plt.show()

############################################################################ Theils_u

col_list = list(inputs.columns)
tot_list=[]
for col_1 in col_list:
    raw_list=[]
    for col_2 in col_list:
        x_val = inputs[col_1].values
        y_val = inputs[col_2].values
        if (col_1 in cat_col)&(col_2 in cat_col):
            raw_list.append(theils_u(x_val,y_val))
        elif (col_1 in con_col)&(col_2 in con_col):
            raw_list.append(np.correlate(x_val,y_val))
        elif (col_1 in cat_col)&(col_2 in con_col):
            raw_list.append(correlation_ratio(x_val,y_val))
        elif (col_1 in con_col)&(col_2 in cat_col):
            raw_list.append(correlation_ratio(y_val,x_val))
    tot_list.append(raw_list)

tot_arr = np.array(tot_list, dtype=np.float32)
theils_u_df = pd.DataFrame(tot_arr, columns=col_list, index=col_list)

############################################################################ Vis

fig, ax = plt.subplots( figsize=(25,25) )

mask = np.zeros_like(theils_u_df, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# 히트맵을 그린다
sns.heatmap(theils_u_df, 
            cmap = 'RdYlBu_r', 
            annot = True,
            mask=mask,
            linewidths=.5,
            cbar_kws={"shrink": .5},
            vmin = -1,vmax = 1
           )  
plt.show()