from multiprocessing import Process, Manager
import numpy as np

def inference(thread_index, dataloader, model_obj, return_dict):
    results_all = []
    # Inference start
    loop_num = len(dataloader)
    input_len = dataloader.input_len()
    for idx_ in range(loop_num):
        input_datas = dataloader.getitem(idx_)
        results = model_obj.inference(input_datas)
        results_all.append(results)

    results_all = np.concatenate(results_all, axis=0)

    results_all = results_all[:input_len]

    return_dict[thread_index] = {
        "results_all": results_all
    }
    return return_dict


def multi_run(dataloaders, models, num_of_cpu=7):
    manager = Manager()
    return_dict = manager.dict()
    jobs = []
    for thread_index in range(int(num_of_cpu)):
        proc = Process(target=inference, args=(
            thread_index,
            dataloaders[thread_index],
            models[thread_index],
            return_dict
        ))
        jobs.append(proc)
        proc.start()

    for proc in jobs:
        proc.join()

    results_all = []
    for thread_index in range(int(num_of_cpu)):
        data_dumps = return_dict[thread_index]
        results_all.append(data_dumps["results_all"])

    return np.concatenate(results_all, axis=0)




