from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QLineEdit, QTextEdit, QTimeEdit, QPushButton
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QTime
import subprocess
import os
import pickle
import shutil
import datetime
import stat


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.excute_name = 'auto_forward_transportation'
        self.time_term = False
        self.mail_from = False
        self.send_to = False
        self.edit_btn = False
        self.delete_btn = False
        self.init_ui()

    def init_ui(self):
        grid = QGridLayout()
        self.setLayout(grid)

        grid.addWidget(QLabel('메일 전송 시간 간격 (단위 : 분) >'), 0, 0)
        grid.addWidget(QLabel('메일 보낸이 >'), 1, 0)
        grid.addWidget(QLabel('메일 받는이 주소 >'), 2, 0)

        self.time_term = QLineEdit("10", self)

        self.mail_from = QLineEdit("PCTR", self)
        self.send_to = QLineEdit("UO94000@hyundai-partners.com", self)

        self.edit_btn = QPushButton("설정")
        self.edit_btn.clicked.connect(self.edit_func)

        self.delete_btn = QPushButton("포워딩 중지")
        self.delete_btn.clicked.connect(self.delete_func)

        grid.addWidget(self.time_term, 0, 1)
        grid.addWidget(self.mail_from, 1, 1)
        grid.addWidget(self.send_to, 2, 1)
        grid.addWidget(self.edit_btn, 3, 0)
        grid.addWidget(self.delete_btn, 3, 1)

        self.setWindowTitle('설정')
        self.setGeometry(300, 300, 400, 200)
        self.show()

    def delete_func(self):
        delete_code = 'schtasks /delete /tn ' + self.excute_name
        subprocess.call(delete_code)
        log_folder_path = 'C:\\auto_forward_transportation'
        if not os.access(log_folder_path, os.W_OK):
            os.chmod(log_folder_path, stat.S_IWUSR)
        shutil.rmtree(log_folder_path)

    def edit_func(self):

        target_file_list = os.listdir("C:\\")
        log_folder_name = "auto_forward_transportation"
        if log_folder_name in target_file_list:
            find_log_data_pickle_file_list = os.listdir("C:\\auto_forward_transportation")
            if 'log_data.pickle' not in find_log_data_pickle_file_list:
                now_datetime = datetime.datetime.now() - datetime.timedelta(seconds=1)
                log_data = {
                    'term': int(self.time_term.text()),
                    'last_end_time': [now_datetime.year, now_datetime.month, now_datetime.day, now_datetime.hour,
                                      now_datetime.minute, now_datetime.second],
                    'mail_from': self.mail_from.text(),
                    'send_to': self.send_to.text(),
                    'done_list': []
                }
                with open('C:\\auto_forward_transportation\\log_data.pickle', 'wb') as f:
                    pickle.dump(log_data, f, pickle.HIGHEST_PROTOCOL)
        else:
            os.mkdir("C:\\auto_forward_transportation")
            now_datetime = datetime.datetime.now() - datetime.timedelta(seconds=1)
            log_data = {
                'term': int(self.time_term.text()),
                'last_end_time': [now_datetime.year, now_datetime.month, now_datetime.day, now_datetime.hour,
                                  now_datetime.minute, now_datetime.second],
                'mail_from': self.mail_from.text().strip(),
                'send_to': self.send_to.text().strip(),
                'done_list': []
            }
            with open('C:\\auto_forward_transportation\\log_data.pickle', 'wb') as f:
                pickle.dump(log_data, f, pickle.HIGHEST_PROTOCOL)

        current_path = os.getcwd()
        target_file_list = os.listdir("C:\\auto_forward_transportation")
        if "auto_forward_transportation.exe" not in target_file_list:
            shutil.copy(current_path+"\\auto_forward_transportation.exe", "C:\\auto_forward_transportation")

        file_list = os.listdir("C:\\Windows\\System32\\Tasks")
        if self.excute_name in file_list:
            modify_code =\
                'schtasks /change /tn ' + self.excute_name +\
                ' /tr C:\\auto_forward_transportation\\auto_forward_transportation.exe /ri ' +\
                self.time_term.text()
            subprocess.call(modify_code)
        else:
            create_code =\
                'schtasks /create /tn ' + self.excute_name +\
                ' /tr C:\\auto_forward_transportation\\auto_forward_transportation.exe /sc minute /mo ' +\
                self.time_term.text()
            subprocess.call(create_code)
