##############################################################################################################################
# 결과 Report를 위한 Util 코드
##############################################################################################################################

########################### Import ###########################
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

########################### For Tensor DataSet ###########################
def tensorset(arr, shape, batch_size, drop_remainder=True):
    # type casting & reshaping
    data = arr.astype(np.float32)
    print("Before reshape : {}".format(data.shape))
    data = np.reshape(data, shape)
    print("After reshape : {} | data type : {}".format(data.shape, data.dtype))
    # make to tensor
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    return ds

########################### Reconstruction Check ###########################
def reconstruct_check(model, train_set_arr, outputs_num=3, pred_idx=0, show_num=20, skip_num=10000, time_len=640):
    show_idx = 0
    # Sample 값을 Return 하기 위해 (추가 분석 필요 시 사용)
    sample_arr_list = []
    for arr in train_set_arr[::skip_num]:
        sample_arr_list.append(arr)
        # Tensor 변환
        tens = tf.convert_to_tensor(arr.reshape(1,time_len,1), dtype=tf.float32)
        # Get Model outputs
        if outputs_num == 1:
            outputs = model(tens)
        else:
            outputs = model(tens)
            outputs = outputs[pred_idx]

        # Ploting
        real = arr.reshape(-1)
        pred = outputs.numpy().reshape(-1)
    
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        ax.plot(np.arange(time_size), real, label="Input", color="b")
        ax.plot(np.arange(time_size), pred, label="Output", color="r")
        ax.legend()
        fig.suptitle('Data Set Number {}'.format(show_idx), fontsize=16)
        plt.show()

        # 적당한 개수를 보여주기 위해 사용
        show_idx+=1
        if show_idx == show_num:
            break
        
    sample_arr = np.array(sample_arr_list)
    print(sample_arr.shape)
    return sample_arr

########################### Model Split ###########################
def model_spliter(model, layers_idx_list):
    first_idx = layers_idx_list[0]
    split_input = tf.keras.Input(model.layers[first_idx].input_shape[1:])
    split_outputs = split_input
    for layer_idx in layers_idx_list:
        split_outputs = model.layers[layer_idx](split_outputs)

    split_model = tf.keras.Model(inputs=split_input, outputs=split_outputs)
    split_model.summary()
    return split_model

########################### Get 1 Sample ###########################
# Latent 조작에 따른 값 변경을 보기 위하여, Sample 추출 시 활용
def get_sample(model, train_set_arr, find_idx, outputs_num=3, latent_idx=0, time_len=640):
    now_idx = 0
    sample_latent_arr = 0
    for arr in train_set_arr:
        now_idx+=1
        if now_idx == find_idx:
            # Tensor 변환
            tens = tf.convert_to_tensor(arr.reshape(1,time_len,1), dtype=tf.float32)
            # Get Model outputs
            if outputs_num == 1:
                outputs = model(tens)
            else:
                outputs = model(tens)
                outputs = outputs[latent_idx]
            
            sample_latent_arr = outputs.numpy().reshape(1, len(outputs))

            fig = plt.figure(figsize=(10,5))
            ax = fig.add_subplot(111)
            ax.plot(arr.reshape(-1), label="Input", color="b")
            ax.legend()
            plt.show()

    return sample_latent_arr

########################### Review From Generator input latent vector ###########################
def ploting_generated_signal(model, sample_latent_arr, outputs_num=3, pred_idx=0):
    tens = tf.convert_to_tensor(sample_latent_arr, dtype=tf.float32)

    # Get Model outputs
    if outputs_num == 1:
        outputs = model(tens)
    else:
        outputs = model(tens)
        outputs = outputs[pred_idx]

    outputs = outputs.numpy()
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax.plot(outputs.reshape(-1), label="Outputs", color="b")
    ax2.stem(sample_latent_arr.reshape(-1))
    ax.legend()
    plt.show()

    return outputs


########################### Latent Space 조작하며 Ploting ###########################
def generation_with_diff_latent_vector(model, sample_latent_arr, add_arr, latent_len, outputs_num=3, pred_idx=0):
    for z_idx in range(latent_len):
        fig = plt.figure(figsize=(len(add_arr)*5,4), constrained_layout=True)
        axs = fig.subplots(1, len(add_arr))
        for idx, ax in enumerate(axs):
            for_input_mu = sample_latent_arr.copy()
            for_input_mu[:,z_idx] += add_arr[idx]

            # Get Model outputs
            if outputs_num == 1:
                outputs = model(for_input_mu)
            else:
                outputs = model(for_input_mu)
                outputs = outputs[pred_idx]

            outputs = outputs.numpy().reshape(-1)
            ax.plot(np.arange(len(outs)), outs)
            ax.set_title("Input Value : {:.3f}".format(add_arr[idx]))

        fig.suptitle('Z index is {}'.format(z_idx+1), fontsize=16)
        plt.show()