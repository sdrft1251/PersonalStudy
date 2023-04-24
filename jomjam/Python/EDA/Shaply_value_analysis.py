import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import xlrd
import graphviz

########################################################################## Data Import
df = pd.read_excel('data_200416.xlsx')
print(df.shape)

######Preprocessing................
inputs = df.drop(columns=['Fraud'])
target = df.Fraud.values
inputs[cat_col] = inputs[cat_col].fillna('nov')
inputs[con_col] = inputs[con_col].fillna(0)
inputs[cat_col] = inputs[cat_col].astype('str')
inputs[con_col] = inputs[con_col].astype('float64')


#####Make Dummy variable (For Categorical features)
start=True
for col in cat_col:
    tmp_dummy = pd.get_dummies(df[col], prefix=col+"@@")
    if start:
        tot_dummy = tmp_dummy
        start=False
    else:
        tot_dummy = pd.merge(tot_dummy,tmp_dummy, left_index=True, right_index=True)
inputs_df = pd.merge(inputs[con_col], tot_dummy, left_index=True, right_index=True)
labeled_target = np.where(target=='N',0,1)
inputs_df = inputs_df.astype('float64')


########################################################################## Modeling
import lightgbm as lgb
lgb_train = lgb.Dataset(inputs_df, labeled_target)
evals_result = {} 
params = {
    'boosting_type': 'gbdt',
    'max_depth' : 4,
    'objective': 'cross_entropy',
    #'min_child_samples':2000,
    #'subsample': 0.5,
    #'colsample_bytree': 0.01,
    #'reg_alpha': 0.01,
    #'reg_lambda': 0.01,
    'num_leaves': 15,
    #'min_child_weight': 0.0001,
    #'min_split_gain' : 0.001,
    #'colsample_bytree': 0.9338761403075675, 'min_child_samples': 128,
    #'min_child_weight': 0.01, 'num_leaves': 32, 'reg_alpha': 0,
    #'reg_lambda': 0.01, 'subsample': 0.8337835096797013, 'subsample_for_bin': 191,
    
    'learning_rate': 0.05,
    'metric' : ['accucary','auc'],
    #multi_logloss
    #multi_error
    'random_state': 501,
    #'categorical_feature':'auto',
    #'class_weight':'balanced'
    
}

########################################################################## Visualizing Model

print('Starting training...')
#mdl.fit(X=train_tot,y=y_train, eval_set=(valid_tot, y_val), eval_metric=['multi_error','multi_logloss'])
gbm = lgb.train(params,lgb_train,num_boost_round=2000,valid_sets=lgb_train,early_stopping_rounds=500,evals_result=evals_result)


print('Plotting metrics recorded during training...')
ax = lgb.plot_metric(evals_result, metric='auc')
plt.show()

print('Plotting feature importances...')
ax = lgb.plot_importance(gbm, max_num_features=10)
plt.show()

print('Plotting tree...')
ax = lgb.plot_tree(gbm, figsize=(30, 30), show_info=['split_gain'])
plt.show()

print('Plotting tree with graphviz...')
graph = lgb.create_tree_digraph(gbm, name='tt')
graph.render(view=True)

########################################################################## Sharply Values
import shap
explainer = shap.TreeExplainer(gbm)
shap_values = explainer.shap_values(inputs_df)

########################################################################## Summary Plot
shap.summary_plot(shap_values, inputs_df)
########################################################################## Summary Plot(for feature importance)
shap.summary_plot(shap_values, inputs_df, plot_type='bar')
########################################################################## Dependence plot for each
for col in inputs_df.columns:
    shap.dependence_plot(col, shap_values, inputs_df)
########################################################################## Force Plot by record
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[10])
########################################################################## Force Plot record that if label is "1"
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[labeled_target==1])