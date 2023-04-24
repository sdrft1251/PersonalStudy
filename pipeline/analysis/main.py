import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


import math
import time
import numpy as np

from model.tflite_maker import transform
from dataloader.base import DataLoader
from model.class3 import Model
from inference.process import multi_run
from analysis.eval_metrics import TopKPrediction
from analysis.plot_utils import PlotUtils



if __name__ == "__main__":
    print(f"--- Inference Start ---")
    start_time = time.time()

    ###################################################################### Setting for inference
    model_base_dir = ""
    data_base_dir = ""
    batch_size = 256
    inputs_shape = (1250, 1)
    num_of_cpu = 10   # For Multiprocessing
    # Get ids of Validation set
    ids = np.load("")
    answer_arr = []
    for cur_id in ids:
        answer_arr.append(int(cur_id.split("_")[-1]))
    answer_arr = np.array(answer_arr)
    ######################################################################

    ###################################################################### Setting for Analysis
    top_k = 50
    target_beats = ["N", "S", "V", "Q"]
    artifact_path = os.path.join(model_base_dir, "artifacts")
    plot_path = os.path.join(model_base_dir, "plots")
    model_name = "model"
    # Paht existence
    os.makedirs(artifact_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)    
    ######################################################################

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ### Create TFLite model
    q_tflite_model_path, v_tflite_model_path, s_tflite_model_path = transform(model_base_dir, batch_size, inputs_shape)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ### Prepare Inference
    data_loader_dumps = []
    model_obj_dumps = []
    len_ids = len(ids)
    split_data_count = len_ids / num_of_cpu
    for thread_index in range(int(num_of_cpu)):
        sliced_ids = ids[int(math.floor(split_data_count * thread_index)) : int(math.floor(split_data_count * (thread_index + 1)))].copy()
        data_loader_dumps.append(DataLoader(sliced_ids, data_base_dir, batch_size, inputs_shape))
        model_obj_dumps.append(Model(q_tflite_model_path, v_tflite_model_path, s_tflite_model_path))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ### Inference start with multiprocessing
    results_all = multi_run(data_loader_dumps, model_obj_dumps, num_of_cpu)
    inference_end_time = time.time()
    print(f"Done --- Result Shape: {results_all.shape} | Number of All beats: {len(ids)} | Processing time: {inference_end_time-start_time}")
    np.save(os.path.join(model_base_dir, "pred_reuslt"), results_all)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ### Analysis start
    eval_cls = TopKPrediction(top_k, target_beats, ids, model_name=model_name)
    if results_all.shape[0] != len(answer_arr):
        cut_idx = min(results_all.shape[0], len(answer_arr))
        eval_df = {
            "prediction": results_all[:cut_idx],
            "target": answer_arr[:cut_idx]
        }
    else:
        eval_df = {
            "prediction": results_all,
            "target": answer_arr
        }
    metrics, artifacts = eval_cls.eval_func(eval_df, artifacts_dir=artifact_path)
    plot_util = PlotUtils(data_base_dir, plot_path)
    for npz_path in artifacts["top_k_list_path"]:
        plot_util.plot_top_k_signals(npz_path)
    plot_util.plot_confusion_matrix(artifacts["confusion_matrix_path"])
    end_time = time.time()
    print(f"check your ID list(.npz) at...\n{artifact_path}")
    print(f"check your Beat images(.png) at...\n{plot_path}")
    print(f"All process is finished. Process time: {end_time-start_time}")


