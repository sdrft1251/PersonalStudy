def return_index(arr, num):
    return np.flip(arr.argsort()[-num:])

def return_static(features_arr):
    #Transform to static
    #Mean values
    mean_arr = features_arr.mean(axis=0)
    #Mag values
    mag_arr = abs(features_arr).mean(axis=0)
    #Std values
    std_arr = features_arr.std(axis=0)
    #Cov values
    cov_xy = np.cov(features_arr[:,0], features_arr[:,1])[0][1]
    cov_yz = np.cov(features_arr[:,1], features_arr[:,2])[0][1]
    cov_zx = np.cov(features_arr[:,2], features_arr[:,0])[0][1]
    cov_gy_xy = np.cov(features_arr[:,3], features_arr[:,4])[0][1]
    cov_gy_yz = np.cov(features_arr[:,4], features_arr[:,5])[0][1]
    cov_gy_zx = np.cov(features_arr[:,5], features_arr[:,3])[0][1]
    #Corr values
    cor_xy = cov_xy / (std_arr[0]*std_arr[1])
    cor_yz = cov_yz / (std_arr[1]*std_arr[2])
    cor_zx = cov_zx / (std_arr[2]*std_arr[0])
    cor_gy_xy = cov_gy_xy / (std_arr[3]*std_arr[4])
    cor_gy_yz = cov_gy_yz / (std_arr[4]*std_arr[5])
    cor_gy_zx = cov_gy_zx / (std_arr[5]*std_arr[3])
    #Energy values
    shift_0_values = np.roll(features_arr, 1, axis=0)[1:]
    shift_1_values = features_arr[1:]
    energys = np.power((shift_0_values-shift_1_values),2)
    #Mean & STD
    energy_mean = energys.mean(axis=0)
    energy_std = energys.std(axis=0)
    #Cov & Cor
    cov_energy_xy = np.cov(energys[:,0], energys[:,1])[0][1]
    cov_energy_yz = np.cov(energys[:,1], energys[:,2])[0][1]
    cov_energy_zx = np.cov(energys[:,2], energys[:,0])[0][1]
    cov_energy_gy_xy = np.cov(energys[:,3], energys[:,4])[0][1]
    cov_energy_gy_yz = np.cov(energys[:,4], energys[:,5])[0][1]
    cov_energy_gy_zx = np.cov(energys[:,5], energys[:,3])[0][1]
    ###
    cor_energy_xy = cov_energy_xy / (energy_std[0]*energy_std[1])
    cor_energy_yz = cov_energy_yz / (energy_std[1]*energy_std[2])
    cor_energy_zx = cov_energy_zx / (energy_std[2]*energy_std[0])
    cor_energy_gy_xy = cov_energy_gy_xy / (energy_std[3]*energy_std[4])
    cor_energy_gy_yz = cov_energy_gy_yz / (energy_std[4]*energy_std[5])
    cor_energy_gy_zx = cov_energy_gy_zx / (energy_std[5]*energy_std[3])
    #Max Min point num
    increase_shift_0 = np.where(shift_0_values-shift_1_values>=0, 1, -1)
    increase_shift_1 = np.roll(increase_shift_0, 1, axis=0)
    max_min_point_num = np.where(increase_shift_0[1:]*increase_shift_1[1:]==-1, 1, 0).sum(axis=0)
    
    #Merge
    merge_list = list(mean_arr) + list(mag_arr) + list(std_arr) + [cor_xy, cor_yz, cor_zx, cor_gy_xy, cor_gy_yz, cor_gy_zx]\
    + list(energy_mean) + list(energy_std)\
    + [cor_energy_xy, cor_energy_yz, cor_energy_zx, cor_energy_gy_xy, cor_energy_gy_yz, cor_energy_gy_zx]\
    + list(max_min_point_num)
    return merge_list
    
def oversampling(features_set, label_set, split_rate, over_count):
    inputs=[]
    labels=[]
    for id_idx in range(features_set.id.min(), features_set.id.max()+1):
        id_sample = features_set[["acc_x","acc_y","acc_z","gy_x","gy_y","gy_z"]][features_set.id==id_idx].values
        label_idx = label_set[label_set.id == id_idx].label.values[0]
        inputs.append(id_sample)
        labels.append(label_idx)
    
    inputs_arr = np.array(inputs)
    labels_arr = np.array(labels)
    x_train, x_val, y_train, y_val = train_test_split(inputs_arr, labels_arr, test_size=split_rate, stratify=labels_arr)
    print("Before upsampling ===")
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    
    #Upsampling part
    unique, counts = np.unique(y_train, return_counts=True)
    label_count = dict(zip(unique, counts))
    add_inputs = []
    add_labels = []
    for key, val in label_count.items():
        if val < over_count:
            this_count = val
            while(this_count<over_count):
                #Extract data from 
                random_idx = np.random.randint(x_train[y_train == key].shape[0])
                rolling_idx = np.random.randint(x_train[0].shape[0])
                extract_input_set = x_train[y_train == key][random_idx]
                rolling_arr = np.roll(extract_input_set, rolling_idx, axis=0)
                
                this_count += 1
                add_inputs.append(rolling_arr)
                add_labels.append(key)
    add_inputs = np.array(add_inputs)
    add_labels = np.array(add_labels)
    
    #Concat
    x_train = np.concatenate((x_train, add_inputs), axis=0)
    y_train = np.concatenate((y_train, add_labels), axis=0)
    
    shuffle_idx = np.random.permutation(len(x_train))
    x_train = x_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    print("After upsampling ===")
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    return x_train, y_train, x_val, y_val
                

def prerpocessing_using_static(features_sets):
    input_set = []
    for sample_features in features_sets:
        #Delete noise & principa component
        sample_features_cum = []
        for idx in range(1, sample_features.shape[0]+1):
            sample_features_cum.append(sample_features[:idx,:].sum(axis=0))
        sample_features_cum = np.array(sample_features_cum)
        principal_eigen_vector = (sample_features_cum[-1,:] - sample_features_cum[0,:]) / len(sample_features_cum)
        
        sample_features_cum_kl = []
        for idx in range(0, sample_features_cum.shape[0]):
            sample_features_cum_kl.append(sample_features_cum[idx,:] - principal_eigen_vector*idx - sample_features_cum[0,:])
        sample_features_cum_kl = np.array(sample_features_cum_kl)
        
        sample_features_cum_kl_max = abs(sample_features_cum_kl).max(axis=0)
        sample_features_cum_kl_max = np.where(sample_features_cum_kl_max==0,1,sample_features_cum_kl_max)
        sample_features_cum_kl_norm = sample_features_cum_kl / sample_features_cum_kl_max
        
        # Get Static List
        static_list = return_static(features_arr=sample_features_cum_kl_norm)
        
        fmax = 50
        dt = 1/fmax
        N  = 600
        t  = np.arange(0,N)*dt
        df = fmax/N
        f = np.arange(0,N)*df
        xf = np.fft.fft(sample_features_cum_kl_norm, axis=0)*dt
        tq_index=f[0:int(N/2+1)]
        tq_abs= np.abs(xf[0:int(N/2+1)])
        
        freq_1 = tq_index[return_index(arr=tq_abs[:,0], num=5)].mean()
        freq_2 = tq_index[return_index(arr=tq_abs[:,1], num=5)].mean()
        freq_3 = tq_index[return_index(arr=tq_abs[:,2], num=5)].mean()
        freq_4 = tq_index[return_index(arr=tq_abs[:,3], num=5)].mean()
        freq_5 = tq_index[return_index(arr=tq_abs[:,4], num=5)].mean()
        freq_6 = tq_index[return_index(arr=tq_abs[:,5], num=5)].mean()
        tot_freq = [freq_1, freq_2, freq_3, freq_4, freq_5, freq_6]
        
        tot_input_list = static_list + tot_freq
        #Append
        input_set.append(tot_input_list)
    
    input_set = np.array(input_set)
    print("Input set : {}".format(input_set.shape))
    return input_set
