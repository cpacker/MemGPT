from pathlib import Path
import webbrowser
from pystray import Icon, Menu, MenuItem
from PIL import Image

from installable_image import InstallableImage
from logviewer import LogViewer


class Tray:
    icon_image: "Path"

    def __init__(self):
        self.icon_image = InstallableImage.get_icon_path()
        self.log_viewer = LogViewer()

    def create(self) -> None:
        """creates tray icon in a thread"""

        def discord(icon, item):
            webbrowser.open("https://discord.gg/letta")

        def _on_quit(icon, *args):
            self.log_viewer.stop_log_terminal()
            icon.stop()

        def _start_log_viewer(icon, item):
            self.log_viewer.start_log_terminal()

        icon = Icon("Letta",
                Image.open(self.icon_image),
                menu=Menu(
                    MenuItem(
                       "View Logs",
                        _start_log_viewer
                    ),
                    MenuItem(
                        "Discord",
                        discord
                ),
                MenuItem(
                        "Quit Letta",
                        _on_quit
                )))
        icon.run()
