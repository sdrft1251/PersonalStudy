from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QLineEdit, QTextEdit, QDateTimeEdit, QPushButton, QFileDialog, QMessageBox
from PyQt5.QtCore import QDateTime
from utils import down_file_from_outlook, merge_ppt
import os
import shutil
import stat

class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
        self.FileFolder = self.desktop_path
        self.initUI()

    def initUI(self):
        grid = QGridLayout()
        self.setLayout(grid)

        grid.addWidget(QLabel('필터링 키워드 >'), 0, 0)
        grid.addWidget(QLabel('주의 >'), 1, 0)
        grid.addWidget(QLabel('시작 날짜 :'), 2, 0)
        grid.addWidget(QLabel('종료 날짜 :'), 3, 0)
        self.label_ff = QLabel(self.desktop_path, self)
        grid.addWidget(self.label_ff, 4, 1)
        grid.addWidget(QLabel('Function start'), 5, 0)

        # Start time
        self.start_datetimeedit = QDateTimeEdit(self)
        self.start_datetimeedit.setDateTime(QDateTime.currentDateTime().addDays(-1).addSecs(-1))
        self.start_datetimeedit.setDateTimeRange(QDateTime(1900, 1, 1, 00, 00, 00), QDateTime(2100, 1, 1, 00, 00, 00))
        self.start_datetimeedit.setDisplayFormat('yyyy.MM.dd hh:mm:ss')
        # End time
        self.end_datetimeedit = QDateTimeEdit(self)
        self.end_datetimeedit.setDateTime(QDateTime.currentDateTime().addSecs(60))
        self.end_datetimeedit.setDateTimeRange(QDateTime(1900, 1, 1, 00, 00, 00), QDateTime(2100, 1, 1, 00, 00, 00))
        self.end_datetimeedit.setDisplayFormat('yyyy.MM.dd hh:mm:ss')

        # Filtering keyword
        self.keyword = QTextEdit("사전출입",self)
        self.keyword.append("사전 출입")
        self.keyword.textChanged.connect(self.onChanged)
        self.filter_detail_label = QLabel("첨부 파일명 기준으로 필터링됩니다.\n반드시 한줄에 하나의 키워드를 넣어주세요.\n\n\t<현재 키워드>\n"+self.keyword.toPlainText(), self)

        # Save Path
        self.file_dir_btn = QPushButton("저장 경로")
        self.file_dir_btn.clicked.connect(self.find_folder)

        self.main_func_btn = QPushButton('Start', self)
        self.main_func_btn.setCheckable(True)
        self.main_func_btn.toggle()
        self.main_func_btn.clicked.connect(self.func_start)

        grid.addWidget(self.keyword, 0, 1)
        grid.addWidget(self.filter_detail_label, 1, 1)
        grid.addWidget(self.start_datetimeedit, 2, 1)
        grid.addWidget(self.end_datetimeedit, 3, 1)
        grid.addWidget(self.file_dir_btn, 4, 0)
        grid.addWidget(self.main_func_btn, 5, 1)

        self.setWindowTitle('Down ppt from mail and merge')
        self.setGeometry(300, 300, 300, 200)
        self.show()


    def find_folder(self):
        self.FileFolder = QFileDialog.getExistingDirectory(self, 'Find Folder', self.desktop_path)
        self.label_ff.setText(self.FileFolder)


    def func_start(self) :

        #Check status
        if self.FileFolder and self.keyword.toPlainText():
            self.FileFolder = self.FileFolder.replace("/", "\\")
            start_dt = self.start_datetimeedit.dateTime()
            start_dt_string = start_dt.toString(self.start_datetimeedit.displayFormat())

            end_dt = self.end_datetimeedit.dateTime()
            end_dt_string = end_dt.toString(self.end_datetimeedit.displayFormat())
            
            input_text = str(self.keyword.toPlainText())
            text_list = self.keyword.toPlainText().split("\n")
            keyword_list = self.make_text_list(input_list=text_list)

            messagehead = "Filter keyword -> \n"
            for ke_ in keyword_list:
                messagehead += ("\t" + ke_)
                messagehead += "\n"
            reply = QMessageBox.question(self, "Message", messagehead + \
                "Start datetiime : " + start_dt.toPyDateTime().strftime('%Y/%m/%d %H:%M %p') + "\nEnd datetime : " + end_dt.toPyDateTime().strftime('%Y/%m/%d %H:%M %p') +\
                "\n\nAre you Sure ? ", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

            if reply == QMessageBox.Yes:
                if self.FileFolder[-1] == "\\":
                    tmp_folder_path = self.FileFolder + 'tmp_folder_for_ppt_down'
                else:
                    tmp_folder_path = self.FileFolder+'\\tmp_folder_for_ppt_down'
                os.mkdir(tmp_folder_path)
                save_num = down_file_from_outlook.start_func(keyword_list=keyword_list, start_date=start_dt.toPyDateTime(), end_date=end_dt.toPyDateTime(), save_dir=tmp_folder_path)
                if save_num == 0:
                    QMessageBox.information(self, "Message", "다운로드 된 파일의 개수가 0 입니다!!")
                    if not os.access(tmp_folder_path, os.W_OK):
                        os.chmod(tmp_folder_path, stat.S_IWUSR)
                    shutil.rmtree(tmp_folder_path)

                else:
                    merge_num = merge_ppt.merge_data(file_path=tmp_folder_path, save_path=self.FileFolder)
                    QMessageBox.information(self, "Message", "작업 완료!!\n\n다운 받은 총 파일 개수 : " + str(save_num) +"\n병합한 총 파일 개수 : " + str(merge_num))
                    if not os.access(tmp_folder_path, os.W_OK):
                        os.chmod(tmp_folder_path, stat.S_IWUSR)
                    shutil.rmtree(tmp_folder_path)
        else:
            if self.FileFolder:
                QMessageBox.warning(self, "Warning", "필터링 키워드 필요")
            else:
                QMessageBox.warning(self, "Warning", "저장 경로 필요")


    def onChanged(self):
        self.filter_detail_label.setText("첨부 파일명 기준으로 필터링됩니다.\n반드시 한줄에 하나의 키워드를 넣어주세요.\n\n\t<현재 키워드>\n"+self.keyword.toPlainText())
        self.filter_detail_label.adjustSize()


    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message', 'Are you sure to quit?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def make_text_list(self, input_list):
        new_text_list = []
        for tt in input_list:
            if tt:
                new_text_list.append(tt.strip())
        return new_text_list
