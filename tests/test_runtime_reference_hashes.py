from pathlib import Path
import hashlib


ROOT = Path(__file__).resolve().parents[1]

# These files are checkpoint-structure-critical and must remain exact copies
# of the verified RoboTwin reference during this migration stage.
CHECKPOINT_CRITICAL_FILES = (
    "dataset.py",
    "model.py",
    "model_spatial.py",
    "model_multimodal.py",
    "pointcloud_preprocess.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checkpoint_critical_runtime_matches_verified_reference():
    for name in CHECKPOINT_CRITICAL_FILES:
        public = ROOT / "dapier_act" / name
        reference = ROOT / "reference" / "robotwin_dapier_act" / name

        assert public.exists(), public
        assert reference.exists(), reference
        assert sha256(public) == sha256(reference), name


def test_rgb_contract_files_are_intentionally_public_only_evolutions():
    # dataset_multimodal.py and train_multimodal.py are intentionally allowed
    # to diverge from the frozen reference after the RGB channel mismatch was
    # reproduced on both the v4 wooden-block corpus and marker corpus.
    for name in ("dataset_multimodal.py", "train_multimodal.py"):
        assert (ROOT / "dapier_act" / name).exists()
        assert (ROOT / "reference" / "robotwin_dapier_act" / name).exists()
