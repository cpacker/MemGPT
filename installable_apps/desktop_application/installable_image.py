from pathlib import Path
import darkdetect

from installable_logger import get_logger

logger = get_logger(__name__)


class InstallableImage:

        @classmethod
        def get_icon_path(cls) -> Path:
            logger.debug("Determining icon path from system settings...")
            image_name = ("dark_" if darkdetect.isDark() else "") +  "icon.png"
            logger.debug(f"Icon path determined to be {image_name} based on system settings.")
            return (Path(__file__).parent.parent / "assets") / image_name
