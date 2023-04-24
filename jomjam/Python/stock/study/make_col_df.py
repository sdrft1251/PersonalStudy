from utils import get_data_from_bt
import pandas as pd


def return_csv(df, bt_index):
    code = []
    close_mean = []
    volume_mean = []
    kind = []
    kospi_kind = []
    eps = []
    per = []
    bps = []
    pbr = []
    for cd in df.code:
        cd_for_naver = cd[1:]
        result_df = get_data_from_bt.return_balancetable(corp_code=cd_for_naver, year_quarter="quarter")
        
        try:
            eps_val = result_df.loc[bt_index]["EPS(원)"]
            per_val = result_df.loc[bt_index]["PER(배)"]
            bps_val = result_df.loc[bt_index]["BPS(원)"]
            pbr_val = result_df.loc[bt_index]["PBR(배)"]

            code.append(cd)
            close_mean.append(df[df["code"==cd]]["close_mean"])
            volume_mean.append(df[df["code"==cd]]["volume_mean"])
            kind.append(df[df["code"==cd]]["kind"])
            kospi_kind.append(df[df["code"==cd]]["kospi_kind"])
            eps.append(eps_val)
            per.append(per_val)
            bps.append(bps_val)
            pbr.append(pbr_val)
        
        except:
            continue
    
    new_df = {"code":code, "close_mean":close_mean, "volume_mean":volume_mean,\
        "kind":kind, "kospi_kind":kospi_kind, "eps":eps, "per":per, "bps":bps, "pbr":pbr}

    return new_df

# "2020/06"

df_200601_200901 = pd.read_csv("./data/20200601_20200901.csv")
print(df_200601_200901)

new_df = return_csv(df=df_200601_200901, bt_index="2020/06")
print(new_df)
new_df.to_csv("./result/200601_200901_col_df.csv", index=False)