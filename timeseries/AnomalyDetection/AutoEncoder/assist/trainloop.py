import os
import sys
import json
# Selecting GPU
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Keras memory 에러 방지
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("GPU List : {}".format(physical_devices))
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)


# Get Config file
json_file_dir = sys.argv[1]
with open(json_file_dir) as json_file:
    config_json = json.load(json_file)

# Append Path
sys.path.append(config_json['model_path'])
sys.path.append("/works/ai2/train_manager")
# 특정 모델만 가져오기
import model, train, utils

# 모델 객체 생성
model_ob = model.ModelPipe(**config_json['model_params'])

# model_ob.build(input_shape=(config_json["batch_size"],config_json["time_size"],1))
# model_ob.summary()

# Get Data
preprocessed_arr = utils.preprocessing_data(**config_json['data_params'])

# Train Start
train.train(model=model_ob, data_col=preprocessed_arr, **config_json['train_params'])


print("END")