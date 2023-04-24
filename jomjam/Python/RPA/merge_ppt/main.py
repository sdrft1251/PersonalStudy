
from utils import app
from PyQt5.QtWidgets import QApplication
import sys

if __name__ == '__main__':
    app_ = QApplication(sys.argv)
    ex = app.MyApp()
    sys.exit(app_.exec_())
