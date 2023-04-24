"""
임시 plotting 유틸리티: 이후 시각화는 Grafana로 대체될 것
"""

import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

beat_list = ["N","S","V","Q"]
code_to_index = {code: idx for idx, code in enumerate(beat_list)}
index_to_code = {idx: code for idx, code in enumerate(beat_list)}


class PlotUtils():
    color_table = ["cornflowerblue", "darkorange", ]

    def __init__(self, raw_data_path, save_path=None):
        self.raw_data_path = raw_data_path
        self.save_path = save_path
        self.is_save = True if save_path else False
        

    def _init_plot(self):
        plt.clf()
        plt.close("all")

    def _load_raw_data(self, id_list_tmp):
        x_ = []
        for idx, val in enumerate(id_list_tmp):
            loaded_data = np.load(f"{self.raw_data_path}/{val}.npy")
            x_.append(loaded_data)
        return x_

    def _apply_padding(self, arrays, padding_value=0):
        """
        apply padding with fixed legnth for all array.
        fixed legnth = max length of given arrays.
        """
        rows = []
        total_length = 0
        for arr in arrays:
            cur_len = len(arr)
            total_length = cur_len if cur_len > total_length else total_length
        for arr in arrays:
            rows.append(np.pad(arr, (0, total_length), "constant", constant_values=padding_value)[:total_length])
        return np.concatenate(rows, axis=0).reshape(-1, total_length)

    def _plot_all_signal(self, signals, predictions, ids_list=None, pred_results=None, prefix=""):
        self._init_plot()

        code_len = predictions.shape[0]
        k = predictions.shape[1]

        # draw summary plot
        fig = plt.figure(figsize=(20,24))
        fig.tight_layout()

        cols = 5
        for i in range(code_len * cols):
            ax = fig.add_subplot(code_len,cols,i+1)
            row = i // cols
            col = i % cols
            ax.plot(signals[row][col])
            ax.set_title(f"{index_to_code[row]}_({col+1}) {predictions[row,col]:.4f}")
        if self.is_save:
            plt.savefig(f"{self.save_path}/{prefix}_top_{cols}_summary.png")

        # if k is small enough, end
        if cols > k:
            return

        # prepare for total plot
        total_path = Path(self.save_path) / "detail"
        total_path.mkdir(parents=True, exist_ok=True)

        if ids_list is None:
            ids_list = [["" for _ in range(k)] for _ in range(code_len)]
        # if matches is None:
        #     matches = np.zeros((code_len, k), dtype=bool)

        text_box_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # draw total plot
        rows = 5
        cols = 5
        plot_size = rows * cols
        iter_len = int(np.ceil(k / plot_size))
        for code_idx in range(code_len):
            for idx in range(iter_len):
                self._init_plot()
                fig = plt.figure(figsize=(20,20))
                fig.tight_layout()

                idx_stack = idx * plot_size
                temp = k - idx_stack
                max_len = plot_size if temp > plot_size else temp
                for i in range(max_len):
                    row = i // cols
                    col = i % cols
                    pred_result = pred_results[code_idx][idx_stack + i]
                    match = 1 if pred_result == code_idx else 0
                    # match = 1 if matches[code_idx][idx_stack + i] else 0
                    ax = fig.add_subplot(rows,cols,i+1)
                    ax.plot(signals[code_idx][idx_stack + i], color=self.color_table[match])
                    plot_text = f"Score:{predictions[code_idx][idx_stack + i]:.4f}\nPredicted as:{index_to_code[pred_result]}"
                    ax.text(0.01, 0.99, plot_text, ha="left", va="top", transform=ax.transAxes, bbox=text_box_props)
                    ax.set_title(f"({idx_stack+i+1}) {ids_list[code_idx][idx_stack + i]}")
                fig.suptitle(f'{prefix} Predicted as "{index_to_code[code_idx]}" (rank: {idx_stack+1} ~ {idx_stack+plot_size})')
                if self.is_save:
                    plt.savefig(total_path / f"{prefix}_{index_to_code[code_idx]}_predicted_top_{k}_{idx}.png")

    def plot_top_k_signals(self, load_path):
        self._init_plot()

        loaded_npz = np.load(load_path)
        prefix = Path(load_path).name.split("_")[0]
        result_ind = loaded_npz["index"]
        result_pred = loaded_npz["prediction"]
        result_res = loaded_npz["result"]
        # result_match = loaded_npz["match"]
        k = result_pred.shape[1]

        total_sig = []
        total_avg_sig = []
        for top_list in result_ind:
            sig = self._load_raw_data(top_list)
            avg_sig = np.mean(self._apply_padding(sig), axis=0)
            total_sig.append(sig)
            total_avg_sig.append(avg_sig)
        
        self._init_plot()
        plt.plot(total_avg_sig[2])
        if self.is_save:
            plt.savefig(f"{self.save_path}/{prefix}_top_{k}_avg.png")
        else:
            plt.show()

        self._plot_all_signal(total_sig, result_pred, result_ind, result_res, prefix)

    def plot_confusion_matrix(self, load_path):
        self._init_plot()

        conf_matrix = np.load(load_path)
        tmp_classes = np.array(beat_list)
        # tmp_classes = tmp_classes[np.unique(y_test)] ### TODO: error when 0 samples class exist(! 비트 등)
        group_counts = [f"{value:.0f}" for value in conf_matrix.flatten()]
        rows_percentages = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
        group_percentages = [f"{value:.2%}" for value in rows_percentages.flatten()]
        annotations = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]
        annotations = np.asarray(annotations).reshape(tmp_classes.size, tmp_classes.size)

        plt.figure(figsize=(10,6))
        fx=sns.heatmap(rows_percentages, annot=annotations, fmt="",cmap="GnBu")
        for i in range(len(tmp_classes)):
            fx.add_patch(Rectangle((i,i), 1, 1, fill=False, edgecolor="crimson", lw=1, clip_on=False))
        fx.set_title('Confusion Matrix \n')
        fx.set_xlabel('\n Predicted Values\n')
        fx.set_ylabel('Actual Values\n')
        fx.xaxis.set_ticklabels(tmp_classes)
        fx.yaxis.set_ticklabels(tmp_classes)
        if self.is_save:
            plt.savefig(f"{self.save_path}/confusion_matrix.png")
        else:
            plt.show()