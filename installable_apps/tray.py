from pathlib import Path
import webbrowser
from pystray import Icon, Menu, MenuItem
from PIL import Image

from installable_image import InstallableImage

class Tray:
    icon_image: "Path"

    def __init__(self):
        self.icon_image = InstallableImage.get_icon_path()

    def create(self) -> None:
        """creates tray icon in a thread"""

        def discord(icon, item):
            webbrowser.open("https://discord.gg/letta")

        def _on_quit(icon, *args):
            icon.stop()

        def _log_viewer(icon, item):
            webbrowser.open("http://localhost:13774")

        icon = Icon("Letta",
                Image.open(self.icon_image),
                menu=Menu(
                    MenuItem(
                       "View Logs",
                        _log_viewer
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
