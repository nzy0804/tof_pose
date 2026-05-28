from pathlib import Path
import sys


def _bundled_root() -> Path | None:
	if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
		return Path(getattr(sys, '_MEIPASS')).resolve()
	return None


_ROOT = _bundled_root() or Path(__file__).resolve().parents[2]

PROJECT_ROOT = _ROOT
ASSETS_DIR = _ROOT / "assets"
MODELS_DIR = ASSETS_DIR / "models"
OUTPUTS_DIR = (Path.cwd() / "outputs") if _bundled_root() else (_ROOT / "outputs")
VIDEOS_DIR = OUTPUTS_DIR / "videos"

DEFAULT_MODEL_PATH = MODELS_DIR / "yolo11l-seg.pt"
DEFAULT_POSE_MODEL_PATH = MODELS_DIR / "yolo11l-pose.pt"
DEFAULT_CAPTURE_VIDEO = VIDEOS_DIR / "tof_capture.mp4"
DEFAULT_POSE_VIDEO = VIDEOS_DIR / "tof_pose_result.mp4"
