from . import TestLink, SampleLink
from . import EcgTest, TestGroup, SampleGroup, PreprocessGroup
from . import db
import os
from config import config

def ecgtest_data_input(save_num=200):
    region_exam = ["KR", "AU", "SG"]
    duration_exam = ["24", "48", "72"]

    edf_file_path = config['development'].EDFFILE_PATH
    edf_file_path_list = []
    for (root, directs, files) in os .walk(edf_file_path):
        for file_ in files:
            file_path = os.path.join(root, file_)
            edf_file_path_list.append(file_path)

    cut_edf_file_path_list = edf_file_path_list[:save_num]

    for idx, edf_path in enumerate(cut_edf_file_path_list):
        tmp_idx = int(idx%3)

        ecg_test = EcgTest(
            region = region_exam[tmp_idx],
            test_id = f"a01_{idx}",
            duration = duration_exam[tmp_idx],
            condition = "Unknown",
            edf_path = edf_path,
            details_path = "None"
        )
        db.session.add(ecg_test)
        db.session.commit()

    return "Success"


def testgroup_data_input(save_num=30):
    for idx in range(save_num):
        test_group = TestGroup(
            group_name= f"a01_testgroup_{idx}",
            group_status= "open"
        )
        db.session.add(test_group)
        db.session.commit()
    return "Success"

def samplegroup_data_input(save_num=30):
    for idx in range(save_num):
        sample_group = SampleGroup(
            group_name= f"a01_samplegroup_{idx}",
            group_status= "open"
        )
        db.session.add(sample_group)
        db.session.commit()
    return "Success"


def preprocess_data_input():
    preprocess_group = PreprocessGroup(group_name="bandwith_process", group_status="None", path="bandwith_process")
    db.session.add(preprocess_group)
    db.session.commit()
    return "Success"


