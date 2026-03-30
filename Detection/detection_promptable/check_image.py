import hashlib
import shutil
from collections import defaultdict
from pathlib import Path
from utils import collect_image_files, select_or_latest

PARENT_FOLDER = Path(__file__).resolve().parent # Folder containing this script

def get_source_stem(stem: str) -> str:
    """Extract the original image stem from [original]_[class]_[index].
    Ex: 2_l_FIN-Benthic_insect_001 -> 2_l_FIN-Benthic
    """
    parts = stem.rsplit("_", 2)
    return parts[0]

def file_sha256(path: Path) -> str:
    """Return the SHA256 hash of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()

def collect_crop_groups(crop_dir: Path) -> dict[tuple[str, str, str, str], list[Path]]:
    """Collect crop images and group them by comparison key."""
    image_files, _ = collect_image_files(crop_dir, stage="crop comparison")
    crop_groups: dict[tuple[str, str, str, str], list[Path]] = defaultdict(list)
    for image_path in image_files:
        rel_path = image_path.relative_to(crop_dir) # Subfolder/filename
        compare_key = (
            rel_path.parent.as_posix(), # Subfolder name
            get_source_stem(rel_path.stem), # Original image stem
            rel_path.suffix.lower(),    # File extension
            file_sha256(image_path),    # Content hash for uniqueness
        )
        crop_groups[compare_key].append(rel_path)
    return crop_groups

def copy_with_structure(crop_dir: Path, rel_paths: list[Path], output_dir: Path) -> None:
    """Copy images while preserving the directory structure under crop_result."""
    for rel_path in rel_paths:
        target_path = output_dir / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(crop_dir / rel_path, target_path)

def get_unmatched_rel_paths(
    ref_groups: dict[tuple[str, str, str, str], list[Path]],
    cur_groups: dict[tuple[str, str, str, str], list[Path]],
) -> tuple[list[Path], list[Path]]:
    """Return unmatched reference and current relative paths after content-based pairing."""
    extra_rel_paths: list[Path] = []    # Extra images comparing to reference
    missing_rel_paths: list[Path] = []  # Missing images comparing to reference

    # Get union of ref_groups keys + cur_groups keys, sort them and compare in order
    for compare_key in sorted(set(ref_groups) | set(cur_groups)):
        ref_rel_paths = sorted(ref_groups.get(compare_key, []))
        cur_rel_paths = sorted(cur_groups.get(compare_key, []))

        # Minimum common count
        shared_count = min(len(ref_rel_paths), len(cur_rel_paths))

        # Add unmatched paths after the shared count
        extra_rel_paths.extend(cur_rel_paths[shared_count:])
        missing_rel_paths.extend(ref_rel_paths[shared_count:])

    return extra_rel_paths, missing_rel_paths

if __name__ == "__main__":
    # Get the reference and current det_dir paths
    det_dir_ref = select_or_latest(PARENT_FOLDER, "Select REFERENCE result_det folder")
    det_dir = select_or_latest(PARENT_FOLDER, "Select CURRENT result_det folder")

    # Get the crop_result subdirectories
    crop_dir_ref = det_dir_ref / "crop_result"
    crop_dir = det_dir / "crop_result"

    # Collect crop groups for both reference and current
    ref_groups = collect_crop_groups(crop_dir_ref)
    cur_groups = collect_crop_groups(crop_dir)

    # Get unmatched relative paths for extra and missing images
    extra_rel_paths, missing_rel_paths = get_unmatched_rel_paths(ref_groups, cur_groups)

    # Prepare comparison directories and copy unmatched images
    compare_dir = det_dir / "00check_image"
    extra_dir = compare_dir / "ref+"
    missing_dir = compare_dir / "ref-"

    copy_with_structure(crop_dir, extra_rel_paths, extra_dir)
    copy_with_structure(crop_dir_ref, missing_rel_paths, missing_dir)
    summary_lines = [
        f"Reference det_dir: {det_dir_ref}",
        f"Current det_dir: {det_dir}",
        f"Reference crop count: {sum(len(paths) for paths in ref_groups.values())}",
        f"Current crop count: {sum(len(paths) for paths in cur_groups.values())}",
        f"ref+ count (more than reference): {len(extra_rel_paths)}",
        f"ref- count (less than reference): {len(missing_rel_paths)}",
        f"Saved comparison folders in: {compare_dir}",
    ]
    summary_text = "\n".join(summary_lines)

    print(f"\n{summary_text}")
    (compare_dir / "chek_image.txt").write_text(summary_text + "\n", encoding="utf-8")