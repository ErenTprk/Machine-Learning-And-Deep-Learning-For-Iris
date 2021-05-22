import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from kod import window

def main():
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = window()
    mainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()