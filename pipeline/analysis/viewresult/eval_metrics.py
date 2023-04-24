"""
Based on MLflow custom evaluation function, below arguments are needed.

:param eval_df:
    A Pandas or Spark DataFrame containing ``prediction`` and ``target`` column.
    The ``prediction`` column contains the predictions made by the model.
    The ``target`` column contains the corresponding labels to the predictions made
    on that row.
        - ISSUE: custom evaluation function에 직접적으로 numpy input을 받을 수 있을지 알 수 없어서,
            간단히 호환될 수 있도록 우선 dictionary of numpy로 가정.
:param builtin_metrics:
    A dictionary containing the metrics calculated by the default evaluator.
    The keys are the names of the metrics and the values are the scalar values of
    the metrics. Refer to the DefaultEvaluator behavior section for what metrics
    will be returned based on the type of model (i.e. classifier or regressor).
:param artifacts_dir: (optional)
    root path for saved artifact files, such as PNG, npy or else.
"""

from pathlib import Path
import numpy as np

from sklearn.metrics import confusion_matrix

# get Top N signals of softmax result score
# TODO: set arguments - k number, adding lowest k, target beat class, etc.
# TODO: make new pipeline for dataloader input inference.

class EvaluationAnalysis():
    beat_list = ["N","S","V","Q"]
    code_to_index = {code: idx for idx, code in enumerate(beat_list)}
    index_to_code = {idx: code for idx, code in enumerate(beat_list)}

    def eval_func(self, eval_df, builtin_metrics=dict(), artifacts_dir=""):
        pass

class TopKPrediction(EvaluationAnalysis):
    def __init__(self, k, target_beats, id_list, model_name="model", is_top=True):
        self.top_k = k # number of samples to be extracted
        self.target_beats = np.array([self.code_to_index[beat_code] for beat_code in target_beats])
        self.id_list = id_list # input data(beat segmented) file's name list
        self.model_name = model_name # for saved file name
        self.is_top = is_top # whether extract from highest (if False, lowest)

    def eval_func(self, eval_df, builtin_metrics=dict(), artifacts_dir=""):
        y_pred = np.array(eval_df["prediction"]) # model output
        y_test = np.array(eval_df["target"]) # label
        class_num = y_pred.shape[1] # class number(6 beats)

        metrics = {}

        # save confusion matrix as npy
        y_preds = np.argmax(y_pred, axis=1)
        beat_codes = list(range(len(self.beat_list)))
        conf_matrix = confusion_matrix(y_test,y_preds, labels=beat_codes)
        conf_matrix_path = Path(artifacts_dir) / f"{self.model_name}_confusion_matrix.npy"
        np.save(conf_matrix_path, conf_matrix)

        # extract top K softmax result of predictions
        org_ind = np.indices(y_pred.shape)[0]
        path_list = []
        for target_beat in self.target_beats:
            target_ind = np.where(y_test == target_beat)[0]
            filtered_ind = org_ind[target_ind]
            filtered_pred = y_pred[target_ind]

            # get top K indice
            if len(filtered_pred) >= self.top_k:
                top_ind = np.argpartition(filtered_pred, -self.top_k, axis=0)[-self.top_k:][::-1]
            else: 
                top_ind = np.argsort(filtered_pred, axis=0)[::-1]

            result_ids = []
            result_pred = []
            # result_match = np.zeros((class_num, top_ind.shape[0]), dtype=bool)
            result_res = np.zeros((class_num, top_ind.shape[0]), dtype=bool)
            for n in range(class_num):

                # re-ordering inside of top K : np.argpartition doesn't care about order.
                cur_ind = top_ind[:,n]
                cur_pred = filtered_pred[cur_ind, n]
                reordered = np.argsort(cur_pred)[::-1]
                cur_ind = cur_ind[reordered]
                cur_pred = cur_pred[reordered]

                # convert indice into file names
                id_ind = filtered_ind[cur_ind,n] # extract original index
                id_list = self.id_list[id_ind] # shape: (k,)

                result_ids.append(id_list)
                result_pred.append(cur_pred)

                # # record the model's prediction result
                # match_ind = np.where((y_test == target_beat) & (y_preds == n))[0]
                # result_match[n] = np.in1d(id_ind, match_ind)

                # record the model's final prediction result(=highest scored class)
                result_res[n] = y_preds[id_ind]

            result_ids = np.array(result_ids) # shape: (6, k)
            result_pred = np.array(result_pred)

            # simple metrics for mlflow eval result
            for code, row in enumerate(result_pred):
                metrics[f"{self.model_name}_{self.index_to_code[code]}_top_{self.top_k}_pred_mean_on_{target_beat}"] = row.mean()

            # save to .npz file
            list_name = f"{self.index_to_code[target_beat]}_top_{self.top_k}_list.npz"
            file_path = str(Path(artifacts_dir) / list_name)
            path_list.append(file_path)
            np.savez(file_path, index=result_ids, prediction=result_pred, result=result_res)

        # set artifacts
        artifacts = {
            "top_k_list_path": path_list,
            "confusion_matrix_path": conf_matrix_path,
        }

        return metrics, artifacts

