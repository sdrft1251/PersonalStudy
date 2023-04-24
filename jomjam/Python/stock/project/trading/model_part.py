import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import tensorflow as tf
from utils import preprocessing
from lightgbm import LGBMRegressor

def return_dnn_model():
    input_tens = tf.keras.Input(shape=(66,))
    x = tf.keras.layers.Dense(32)(input_tens)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=input_tens, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss = 'mean_squared_error', metrics = ['mae'])
    return model


def get_result_from_model(train_input, train_target, running_opt, val_input=0, val_target=0, train_size=0.7, model_opt=False, feature_cate_quantile_num=0, selected_col=False):
    if (running_opt=='score'):
        train_cut_index = int(len(train_target)*train_size)

        train_parts_input = train_input.iloc[:train_cut_index]
        train_parts_target = train_target[:train_cut_index]
        val_parts_input = train_input.iloc[train_cut_index:]
        val_parts_target = train_target[train_cut_index:]

        if (feature_cate_quantile_num==0):
            scaler = RobustScaler()
            scaler.fit(train_parts_input)
            train_parts_input = scaler.transform(train_parts_input)
            val_parts_input = scaler.transform(val_parts_input)
            
        
        if (model_opt=='lgbm'):
            model_ob = LGBMRegressor(objective='regression', num_iterations=10**3)
            if (feature_cate_quantile_num!=0):
                model_ob.fit(train_parts_input, train_parts_target, eval_set=[(train_parts_input, train_parts_target)], early_stopping_rounds=100, verbose=False, categorical_feature=selected_col)
            else:
                model_ob.fit(train_parts_input, train_parts_target, eval_set=[(train_parts_input, train_parts_target)], early_stopping_rounds=100, verbose=False)

        elif(model_opt=='linear'):
            model_ob = LinearRegression()
            model_ob.fit(train_parts_input, train_parts_target)
        elif(model_opt=='svr'):
            model_ob = SVR()
            model_ob.fit(train_parts_input, train_parts_target)
        elif(model_opt=='dnn'):
            model_ob = return_dnn_model()
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
            model_ob.fit(train_parts_input, train_parts_target, epochs=3000, batch_size=64, validation_data=(val_parts_input, val_parts_target), callbacks=[early_stopping], verbose=0)
        else:
            print("Wrong model opt")
            return -1

        train_pred = model_ob.predict(train_parts_input)
        val_pred = model_ob.predict(val_parts_input)

        train_mse = mean_squared_error(train_parts_target, train_pred)
        val_mse = mean_squared_error(val_parts_target, val_pred)

        train_rmse = np.sqrt(train_mse)
        val_rmse = np.sqrt(val_mse)

        return train_rmse, val_rmse

    elif (running_opt=='simulate'):
        if (feature_cate_quantile_num==0):
            scaler = RobustScaler()
            scaler.fit(train_input)
            train_input = scaler.transform(train_input)
            val_input = scaler.transform(val_input)

        if (model_opt=='lgbm'):
            model_ob = LGBMRegressor(objective='regression', num_iterations=10**3)
            if (feature_cate_quantile_num!=0):
                model_ob.fit(train_input, train_target, eval_set=[(train_input, train_target)], early_stopping_rounds=100, verbose=False, categorical_feature=selected_col)
            else:
                model_ob.fit(train_input, train_target, eval_set=[(train_input, train_target)], early_stopping_rounds=100, verbose=False)

        elif(model_opt=='linear'):
            model_ob = LinearRegression()
            model_ob.fit(train_input, train_target)
        elif(model_opt=='svr'):
            model_ob = SVR()
            model_ob.fit(train_input, train_target)
        else:
            print("Wrong model opt")
            return -1

        for_score_pred = model_ob.predict(train_input)
        mse_ = mean_squared_error(train_target, for_score_pred)
        rmse_ = np.sqrt(mse_)

        fin_result = model_ob.predict(val_input)
        fin_rate = fin_result[-1]
        real_target = val_target
        return fin_rate, rmse_, real_target


    elif (running_opt=='trade'):
        if (feature_cate_quantile_num==0):
            scaler = RobustScaler()
            scaler.fit(train_input)
            train_input = scaler.transform(train_input)
            val_input = scaler.transform(val_input)

        if (model_opt=='lgbm'):
            model_ob = LGBMRegressor(objective='regression', num_iterations=10**3)
            if (feature_cate_quantile_num!=0):
                model_ob.fit(train_input, train_target, eval_set=[(train_input, train_target)], early_stopping_rounds=100, verbose=False, categorical_feature=selected_col)
            else:
                model_ob.fit(train_input, train_target, eval_set=[(train_input, train_target)], early_stopping_rounds=100, verbose=False)

        elif(model_opt=='linear'):
            model_ob = LinearRegression()
            model_ob.fit(train_input, train_target)
        elif(model_opt=='svr'):
            model_ob = SVR()
            model_ob.fit(train_input, train_target)
        else:
            print("Wrong model opt")
            return -1

        for_score_pred = model_ob.predict(train_input)
        mse_ = mean_squared_error(train_target, for_score_pred)
        rmse_ = np.sqrt(mse_)

        fin_result = model_ob.predict(val_input)
        fin_rate = fin_result[-1]
        return fin_rate, rmse_

    else:
        if (feature_cate_quantile_num==0):
            scaler = RobustScaler()
            scaler.fit(train_input)
            train_input = scaler.transform(train_input)
            val_input = scaler.transform(val_input)
        
        if (model_opt=='lgbm'):
            model_ob = LGBMRegressor(objective='regression', num_iterations=10**3)
            if (feature_cate_quantile_num!=0):
                model_ob.fit(train_input, train_target, eval_set=[(train_input, train_target)], early_stopping_rounds=100, verbose=False, categorical_feature=selected_col)
            else:
                model_ob.fit(train_input, train_target, eval_set=[(train_input, train_target)], early_stopping_rounds=100, verbose=False)

        elif(model_opt=='linear'):
            model_ob = LinearRegression()
            model_ob.fit(train_input, train_target)
        elif(model_opt=='svr'):
            model_ob = SVR()
            model_ob.fit(train_input, train_target)
        else:
            print("Wrong model opt")
            return -1

        for_score_pred = model_ob.predict(train_input)
        mse_ = mean_squared_error(train_target, for_score_pred)
        rmse_ = np.sqrt(mse_)

        fin_result = model_ob.predict(val_input)
        fin_rate = fin_result[-1]
        fin_rate_scale = fin_rate - rmse_
        return fin_rate_scale


