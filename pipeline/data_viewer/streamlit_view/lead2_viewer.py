import streamlit as st
import sqlite3
import os
import numpy as np
import matplotlib.pyplot as plt


numpy_arr_file_root = "/home/weladmin/Desktop/code/dump/smc_data_view/data"
meta_data_db = "/home/weladmin/Desktop/code/dump/smc_data_view/wavedata_II_v01.db"


# DB Connect
conn = sqlite3.connect(meta_data_db)
cur = conn.cursor()


def main():
    st.title("Viewer (Lead II)")
    st.sidebar.title("Info")

    record_idx = st.sidebar.text_input('Record Index', '')
    print(f"Request Record idx is : {record_idx}")
    
    file_name = get_file_name(record_idx)

    if file_name == -1:   # No Data
        st.text("Sorry... There is no Data!")
    else:   # Exist
        full_path = os.path.join(numpy_arr_file_root, file_name+".npy")
        data = get_data(full_path)
        if type(data) is int:
            st.text("Sorry... Loading Data Failed...\nPlease check record ID\nIf correct, please memo and inform to developer!")
        else:
            st.line_chart(-data, use_container_width=True)
            # fig = make_arr_to_image(data)
            # st.pyplot(fig)


def get_file_name(record_idx):
    cur.execute(f"SELECT filename FROM waveinfo_II WHERE origin_path='{record_idx}';")
    result = cur.fetchall()
    # No Data
    if len(result) == 0:
        print(f"Sorry... There is no Data! -> {record_idx}")
        return -1
    # Exist
    else:
        target = result[0][0]
        return target

def get_data(full_path):
    try:
        return np.load(full_path)
    except Exception as e:
        print(f"Load Failed -> {e}")
        return -1


# def make_arr_to_image(data):
#     fig, ax = plt.subplots(figsize=(15, 3))

#     # For Size
#     qs = np.percentile(data, [1, 99], interpolation='nearest')
#     q_1 = qs[0]
#     q_3 = qs[1]
#     iqr = q_3 - q_1
#     y_min = q_1 - 0.3*iqr
#     y_max = q_3 + 0.3*iqr
#     # For Tick
#     major_ticks_top=np.arange(0,len(data),500)
#     minor_ticks_top=np.arange(0,len(data),500/5)

#     ax.plot(data, 'r-')
#     ax.set_xticks(major_ticks_top)
#     ax.set_xticks(minor_ticks_top, minor=True)
#     ax.grid(which="major",alpha=0.8)
#     ax.grid(which="minor",alpha=0.3)
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_ylim(y_min,y_max)
#     return fig


if __name__ == "__main__":
    main()