from typing import TYPE_CHECKING
import darkdetect
import webbrowser
from pystray import Icon, Menu, MenuItem
from PIL import Image

from cutelog_overload import CutelogOverload
from installable_image import InstallableImage

if TYPE_CHECKING:
    from pathlib import Path

class Tray:
    icon_image: "Path"

    def __init__(self):
        self.icon_image = InstallableImage.get_icon_path()

    def create(self) -> None:
        """creates tray icon in a thread"""
        log_viewer = CutelogOverload()

        def discord(icon, item):
            webbrowser.open("https://discord.gg/letta")

        def _on_quit(icon, *args):
            log_viewer.stop_log_viewer()
            icon.stop()

        def _start_log_viewer(icon, item):
            log_viewer.start_log_viewer()

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
