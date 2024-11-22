from pathlib import Path
import darkdetect


class InstallableImage:

        @classmethod
        def get_icon_path(cls) -> Path:
            image_name = ("dark_" if darkdetect.isDark() else "") +  "icon.png"
            return (Path(__file__).parent / "assets") / image_name
