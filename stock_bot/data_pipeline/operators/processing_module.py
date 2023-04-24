import numpy as np

############################################# Base #############################################
class processing:
    def __init__(self, df):
        self.df = df
        self.none_data_len = None
        self.none_label_len = None

    def set_none_data_len(self, window_size):
        max_window_size = int(np.array(window_size).max())
        if self.none_data_len is None:
            self.none_data_len = max_window_size
        else:
            self.none_data_len = max(self.none_data_len, max_window_size)

    def mean_windows(self, target_col, window_size):
        self.set_none_data_len(window_size)
        return self.df[target_col].rolling(window_size).mean() / self.df[target_col]

    def max_windows(self, target_col, window_size):
        self.set_none_data_len(window_size)
        return self.df[target_col].rolling(window_size).max() / self.df[target_col]

    def min_windows(self, target_col, window_size):
        self.set_none_data_len(window_size)
        return self.df[target_col].rolling(window_size).min() / self.df[target_col]

    def shifting(self, target_col, periods):
        return self.df[target_col].shift(periods=periods)


    def make_target_cols(self, periods_list):
        tmp_col_list = []
        for periods in periods_list:
            col_name = f"After_{periods}_Close"
            self.df[col_name] = self.df["Close"] / self.shifting("Close", -periods)   # Reverse Periods -> For Reverse shifting
            tmp_col_list.append(col_name)

        self.df["Target"] = self.df[tmp_col_list].mean(axis=1)
        self.df = self.df.drop(columns=tmp_col_list, axis=1)

        self.none_label_len = int(np.array(periods_list).max())

    def get_df_with_drop_na(self):
        return self.df.dropna(axis=0)


############################################# Modules #############################################
class process_1(processing):
    def __init__(self, df):
        super(process_1, self).__init__(df)
        self.window_list = [5,10,20,60,120]
        self.after_periods = [1,2,3,4,5]
        self.processed_columns = ["Date"]

    def append_ma(self, target):
        for win in self.window_list:
            col_name = f"MA_{target}_{win}"
            self.df[col_name] = self.mean_windows(target, win)
            self.processed_columns.append(col_name)

    def append_highest(self, target):
        for win in self.window_list:
            col_name = f"Highest_{target}_{win}"
            self.df[col_name] = self.max_windows(target, win)
            self.processed_columns.append(col_name)

    def append_lowest(self, target):
        for win in self.window_list:
            col_name = f"Lowest_{target}_{win}"
            self.df[col_name] = self.min_windows(target, win)
            self.processed_columns.append(col_name)

    def prcossing_df(self):
        self.append_ma("High")
        self.append_ma("Low")
        self.append_ma("Close")
        self.append_ma("Volume")

        self.append_highest("High")
        self.append_highest("Low")
        self.append_highest("Close")
        self.append_highest("Volume")

        self.append_lowest("High")
        self.append_lowest("Low")
        self.append_lowest("Close")
        self.append_lowest("Volume")

        self.make_target_cols(self.after_periods)

        train_part = self.df[self.processed_columns + ["Target"]].iloc[self.none_data_len-1:-self.none_label_len]
        predict_part = self.df[self.processed_columns].iloc[-self.none_label_len:]
        
        return train_part, predict_part