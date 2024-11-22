## Overloading the entrypoint to cutelog control the flow
import sys
import signal
from cutelog.config import ROOT_LOG, CONFIG, parse_cmdline
from cutelog.main_window import MainWindow
from cutelog.resources import qCleanupResources
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QApplication
from qtpy import PYQT5, PYSIDE2

from installable_image import InstallableImage

class CutelogOverload:
    app: "QApplication"
    window: "MainWindow"
    exec = "QApplication.exec_"

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.exec = self.app.exec_

    def start_log_viewer(self):
        if not PYQT5 and not PYSIDE2:
            if sys.platform == 'linux':
                sys.exit("Error: a compatible Qt library couldn't be imported.\n"
                        "Please install python3-pyqt5 (or just python-pyqt5) from your package manager.")
            else:  # this technically shouldn't ever happen
                sys.exit("Error: a compatible Qt library couldn't be imported.\n"
                        "Please install it by running `pip install pyqt5")
                if sys.platform == 'win32':
                    import ctypes
                    appid = 'busimus.cutelog'
                    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appid)
        image = str(InstallableImage.get_icon_path().absolute())
        self.app.setWindowIcon(QIcon(image))
        overrides, load_logfiles = parse_cmdline(ROOT_LOG)
        CONFIG.set_overrides(overrides)
        self.window = MainWindow(ROOT_LOG, self.app, load_logfiles)
        self.window.setWindowTitle('Letta')
        signal.signal(signal.SIGINT, self.window.signal_handler)

    def stop_log_viewer(self):
        sys.exit(self.exec())
        qCleanupResources()