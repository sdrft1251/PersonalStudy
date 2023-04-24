import tensorflow as tf
import numpy as np


class Model():
    def __init__(self, q_path, v_path, s_path):
        self.q_path = q_path
        self.v_path = v_path
        self.s_path = s_path
        self.build()
        self.setting()


    def build(self):
        self.q_class = tf.lite.Interpreter(model_path=self.q_path)
        self.q_class.allocate_tensors()

        self.v_class = tf.lite.Interpreter(model_path=self.v_path)
        self.v_class.allocate_tensors()

        self.s_class = tf.lite.Interpreter(model_path=self.s_path)
        self.s_class.allocate_tensors()

    def setting(self):
        self.input_details_q = self.q_class.get_input_details()
        self.input_details_v = self.v_class.get_input_details()
        self.input_details_s = self.s_class.get_input_details()

        self.output_details_q = self.q_class.get_output_details()
        self.output_details_v = self.v_class.get_output_details()
        self.output_details_s = self.s_class.get_output_details()


    def inference(self, input_data):
        # Q Classify
        self.q_class.set_tensor(self.input_details_q[0]["index"], input_data)
        self.q_class.invoke()
        q_results = self.q_class.get_tensor(self.output_details_q[0]["index"])

        # V Classify
        self.v_class.set_tensor(self.input_details_v[0]["index"], input_data)
        self.v_class.invoke()
        v_results = self.v_class.get_tensor(self.output_details_v[0]["index"])

        # s Classify
        self.s_class.set_tensor(self.input_details_s[0]["index"], input_data)
        self.s_class.invoke()
        s_results = self.s_class.get_tensor(self.output_details_s[0]["index"])

        # Reset Var
        self.q_class.reset_all_variables()
        self.v_class.reset_all_variables()
        self.s_class.reset_all_variables()

        # Post Processing
        all_result = self.postprocess_for_result(q_results, v_results, s_results)
        return all_result


    def postprocess_for_result(self, q_results, v_results, s_results):
        all_result = np.zeros((len(q_results), 4), dtype=np.float32)
        # Only for SparseCategorical
        q_results_idx = np.argmax(q_results, axis=1)
        v_results_idx = np.argmax(v_results, axis=1)
        s_results_idx = np.argmax(s_results, axis=1)

        for idx_ in range(len(q_results_idx)):
            pred_q = q_results_idx[idx_]
            if pred_q == 0:   # (N,S,V)
                pred_v = v_results_idx[idx_]
                if pred_v == 0:   # (N,S)
                    pred_s = s_results_idx[idx_]
                    if pred_s == 0:   # N remain
                        all_result[idx_][0] = s_results[idx_][0]
                    else:   # S filtered
                        all_result[idx_][1] = s_results[idx_][1]
                else:   # V filtered
                    all_result[idx_][2] = v_results[idx_][1]
            else:   # Q filtered
                all_result[idx_][3] = q_results[idx_][1]
        return all_result


