from statsmodels.stats.outliers_influence import variance_inflation_factor


####################################################################### Data Import and Factorize
inputs_cop = inputs.drop(columns=["Fraud","KWHRate3","KWHRate2","KWHRate1","enroll_month","start_year","sectional_area","CreditScore","EnergyCharge"])
inputs_cop2 = inputs_cop.copy()
for col in inputs_cop.columns:
    if col in cat_col:
        factored,_ = pd.factorize(inputs_cop[col].values)
        inputs_cop2[col] = factored
    else:
        pass

####################################################################### Make Table
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(
    inputs_cop2.values, i) for i in range(inputs_cop2.shape[1])]
vif["features"] = inputs_cop2.columns

vif = vif.sort_values(by=['VIF Factor'], ascending=False).reset_index(drop=True)
vif