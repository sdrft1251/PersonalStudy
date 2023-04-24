import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


switch_num = 16


def return_pca(nums, arr):
    pca = PCA(n_components=nums)
    printcipalComponents_mu = pca.fit_transform(arr)
    return pca

def make_zsample_use_pca(pca_vactor, pca):
    return (np.array(pca_vactor).reshape(-1,1) * pca.components_).sum(axis=0)


def generate_ecg(features, feature_names, default_arr):

    reconstructed_model = tf.keras.models.load_model("/works/GitLab/jomjam/Python/AnomalyDetection/ECG/StreamLit/vae_cnn_gru")
    
    switch_add = np.array(features, dtype=np.float32).reshape(1,switch_num)
    for_input = default_arr + switch_add
    outs = reconstructed_model(for_input).numpy().reshape(-1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(outs)), outs)
    ax.set(ylim=(0.2,0.8))
    
    return fig

def main():
    st.title("ECG Generator")

    st.sidebar.title("Features")

    if st.sidebar.checkbox("Zero Inputs"):
        default_arr = np.zeros((1,switch_num), dtype=np.float32)
    else:
        default_arr =  np.load("/works/GitLab/jomjam/Python/AnomalyDetection/ECG/StreamLit/default.npy")

    default_control_features = ["Feature_1", "Feature_2", "Feature_3", "Feature_4","Feature_5","Feature_6","Feature_7","Feature_8",\
    "Feature_9","Feature_10","Feature_11","Feature_12","Feature_13","Feature_14","Feature_15","Feature_16"]

    features = np.zeros(switch_num)

    for idx, name in enumerate(default_control_features):
        features[idx] = st.sidebar.slider(name, -4.0, 4.0, 0.0, 0.1)
    
    fig = generate_ecg(features, default_control_features, default_arr)
    st.pyplot(fig)



if __name__ == "__main__":
    main()