from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = ASSETS_DIR / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VIDEOS_DIR = OUTPUTS_DIR / "videos"

DEFAULT_MODEL_PATH = MODELS_DIR / "tof_pose_best.pt"
DEFAULT_CAPTURE_VIDEO = VIDEOS_DIR / "tof_capture.mp4"
DEFAULT_POSE_VIDEO = VIDEOS_DIR / "tof_pose_result.mp4"
