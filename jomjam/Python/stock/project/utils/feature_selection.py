import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.inspection import permutation_importance
import shap
from sklearn.metrics import mean_squared_error, accuracy_score
from utils import config, feature_selection, preprocessing
import warnings
warnings.filterwarnings("ignore")

def rmse_expm1(pred, true):
    return -np.sqrt(np.mean((np.expm1(pred)-np.expm1(true))**2))

def evaluate(x_data, y_data, train_size):
    model = LGBMRegressor(objective='regression', num_iterations=10**3)
    cut_index = int(len(y_data)*train_size)
    x_train = x_data.iloc[:cut_index]
    y_train = y_data[:cut_index]
    x_val = x_data.iloc[cut_index:]
    y_val = y_data[cut_index:]
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=100, verbose=False)
    val_pred = model.predict(x_val)
    score = mean_squared_error(val_pred, y_val)
    score = np.sqrt(score)
    return score


def rfe(x_data, y_data, method, train_size, ratio=0.9, min_feats=40):
    feats = x_data.columns.tolist()
    archive = pd.DataFrame(columns=['model', 'n_feats', 'feats', 'score'])
    while True:
        model = LGBMRegressor(objective='regression', num_iterations=10**3)
        cut_index = int(len(y_data)*train_size)
        x_train = x_data[feats].iloc[:cut_index]
        y_train = y_data[:cut_index]
        x_val = x_data[feats].iloc[cut_index:]
        y_val = y_data[cut_index:]
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=100, verbose=False)
        val_pred = model.predict(x_val)
        score = mean_squared_error(val_pred, y_val)
        score = np.sqrt(score)
        n_feats = len(feats)
        #print(n_feats, score)
        archive = archive.append({'model': model, 'n_feats': n_feats, 'feats': feats, 'score': score}, ignore_index=True)
        if method == 'basic':
            feat_imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
        elif method == 'perm':
            perm = permutation_importance(model, x_val, y_val, random_state=0)
            feat_imp = pd.Series(perm.importances_mean, index=feats).sort_values(ascending=False)
        elif method == 'shap':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x_data[feats])
            feat_imp = pd.Series(np.abs(shap_values).mean(axis=0), index=feats).sort_values(ascending=False)
        next_n_feats = int(n_feats * ratio)
        if next_n_feats < min_feats:
            break
        else:
            feats = feat_imp.iloc[:next_n_feats].index.tolist()
    return archive


def return_filtered_feature_df(train_input, train_target, feature_cate_quantile_num=0):
    if (feature_cate_quantile_num!=0):
        train_input = train_input.astype('category')

    model = LGBMRegressor(objective='regression', num_iterations=10**3)
    
    cut_index = int(len(train_target)*(0.7))
    x_train = train_input.iloc[:cut_index]
    y_train = train_target[:cut_index]
    x_val = train_input.iloc[cut_index:]
    y_val = train_target[cut_index:]
    if (feature_cate_quantile_num!=0):
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=100, verbose=False, categorical_feature=preprocessing.tot_col_names)
    else:
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=100, verbose=False)

    feat_imp = pd.Series(model.feature_importances_, index=preprocessing.tot_col_names).sort_values(ascending=False)
    
    perm = permutation_importance(model, x_val, y_val, random_state=0)
    perm_feat_imp = pd.Series(perm.importances_mean, index=preprocessing.tot_col_names).sort_values(ascending=False)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(train_input)
    shap_feat_imp = pd.Series(np.abs(shap_values).mean(axis=0), index=preprocessing.tot_col_names).sort_values(ascending=False)
    
    feat_imp_archive = pd.DataFrame(index=preprocessing.tot_col_names, columns=['basic', 'perm', 'shap', 'mean'])
    feat_imp_archive['basic'] = feat_imp.rank(ascending=False)
    feat_imp_archive['perm'] = perm_feat_imp.rank(ascending=False)
    feat_imp_archive['shap'] = shap_feat_imp.rank(ascending=False)
    feat_imp_archive['mean'] = feat_imp_archive[['basic', 'perm', 'shap']].mean(axis=1)
    feat_imp_archive = feat_imp_archive.sort_values('mean')
    score = 100
    number_of_feature = 0
    for idx_ in range(10, len(preprocessing.tot_col_names)+1, 10):
        tmp_score = feature_selection.evaluate(train_input[feat_imp_archive.iloc[:idx_].index], train_target, train_size=0.7)
        if (score>tmp_score):
            score = tmp_score
            number_of_feature = idx_
    
    filtered_columns_split_inputs = train_input[feat_imp_archive.iloc[:number_of_feature].index]
    return filtered_columns_split_inputs