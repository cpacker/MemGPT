import darkdetect
import webbrowser
from pathlib import Path
from pystray import Icon, Menu, MenuItem
from PIL import Image


class Tray:
    icon_image: Path

    def __init__(self):
        image_name = ("dark_" if darkdetect.isDark() else "") +  "tray.png"
        self.icon_image = Path(__file__).parent / image_name

    def create(self) -> None:
        """creates tray icon in a thread"""

        def discord(icon, item):
            webbrowser.open("https://discord.gg/letta")

        def _on_quit(icon, *args):
            icon.stop()

        icon = Icon("Letta",
                Image.open(self.icon_image),
                menu=Menu(
                    MenuItem(
                        "Discord",
                        discord
                )))
                menu=Menu(
                    MenuItem(
                        "Quit Letta",
                        _on_quit
                )))
        icon.run()
