import numpy as np
import pandas as pd

def return_int_target(val):
  if val == "N":
    return 0
  elif val == "S":
    return 1
  elif val == "V":
    return 2
  elif val == "F":
    return 3
  else :
    return 4
    
df = pd.read_csv("/home/jaemincho/works/hb/heartbeat.csv")
df["target"] = df.Category.apply(return_int_target)

train = df[df.columns[2:-1]]
target = df[df.columns[-1]]
train = np.array(train, dtype=np.float32)
target = np.array(target, dtype=np.float32)

def return_diff_array_table(array, dur):
  for idx in range(array.shape[1]-dur):
    before_col = array[:,idx]
    after_col = array[:,idx+dur]
    new_col = ((after_col - before_col)+1)/2
    new_col = new_col.reshape(-1,1)
    if idx == 0:
      new_table = new_col
    else :
      new_table = np.concatenate((new_table, new_col), axis=1)
  padding_array = np.zeros(shape=(array.shape[0],dur))
  new_table = np.concatenate((padding_array, new_table), axis=1)
  return new_table

def return_ma_array_table(array, dur):
  for idx in range(array.shape[1]-dur):
    base_array = array[:,idx]
    for idx2 in range(dur):
      base_array += array[:,idx+idx2]
    base_array = base_array/dur
    base_array = base_array.reshape(-1,1)
    if idx == 0:
      new_table = base_array
    else :
      new_table = np.concatenate((new_table, base_array), axis=1)
  padding_array = np.zeros(shape=(array.shape[0],dur))
  new_table = np.concatenate((padding_array, new_table), axis=1)
  return new_table

def return_ar_array_table(array, dur):
  for idx in range(array.shape[1]-dur):
    before_col = array[:,idx]
    after_col = array[:,idx+dur]
    new_col = ( (after_col+1)/(before_col+1) )-1
    new_col = new_col.reshape(-1,1)
    if idx == 0:
      new_table = new_col
    else :
      new_table = np.concatenate((new_table, new_col), axis=1)
  padding_array = np.zeros(shape=(array.shape[0],dur))
  new_table = np.concatenate((padding_array, new_table), axis=1)
  return new_table

def return_merge_diff_ma_ar_table(df, diff_dur, ar_dur):
  fin_table = df.reshape(-1,187,1,1)
  for dur in diff_dur:
    temp_table = return_diff_array_table(df, dur)
    fin_table = np.concatenate((fin_table, temp_table.reshape(-1,187,1,1)), axis=2)
  for dur in ar_dur:
    temp_table = return_ar_array_table(df, dur)
    fin_table = np.concatenate((fin_table, temp_table.reshape(-1,187,1,1)), axis=2)
  return fin_table
  

x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2, stratify=target)

x_train = return_merge_diff_ma_ar_table(df=x_train, diff_dur=[1], ar_dur=[1])
x_test = return_merge_diff_ma_ar_table(df=x_test, diff_dur=[1], ar_dur=[1])

print("Shape is =============================================")
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# Need more code if you want save