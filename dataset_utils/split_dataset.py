from __future__ import annotations
import random
import shutil
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# =========================================================
# ⚙️ CONFIG — À MODIFIER
# =========================================================
DATA_DIR = Path("/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/Datasets/AQUA-IA_dataset_mars2026")  # ton dossier actuel: DATA_DIR/classe_x/*.jpg
OUT_DIR = Path("/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/Datasets/AQUA-IA_dataset_mars2026_splited")  # dossier de sortie

GROUP_BY_FILENAME = False

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

SEED = 42
MODE = "copy"  # "copy" ou "move"
DRY_RUN = False  # True = ne copie/déplace rien (affiche juste)

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# =========================================================
# 🧠 GROUPING (nomenclature)
# =========================================================
# Exemple souhaité: "0-Hydropsyche_pel25*" -> groupe = "0-Hydropsyche_pel25"
# Regex:
#  - commence par digits + '-' (ex: 0-)
#  - puis une suite de lettres/chiffres/_ (nom de classe)
#  - puis un suffixe numérique (Num) (ex: 25)
GROUP_RE = re.compile(r"^(\d+-[A-Za-z0-9_]+?\d+)", re.ASCII)


def group_key_from_filename(path: Path) -> str:
    """
    Retourne la clé de groupe pour éviter de répartir des variantes du même item
    dans différents splits.
    """
    stem = path.stem  # nom sans extension
    m = GROUP_RE.match(stem)
    if m:
        return m.group(1)

    # Fallback: avant le premier underscore (souvent "item_001", "item_crop", etc.)
    if "_" in stem:
        return stem.split("_", 1)[0]

    # Dernier fallback: nom complet
    return stem


# =========================================================
# 🔧 UTILITAIRES
# =========================================================
def ensure_dir(p: Path) -> None:
    if not DRY_RUN:
        p.mkdir(parents=True, exist_ok=True)


def transfer(src: Path, dst: Path) -> None:
    if DRY_RUN:
        return
    if MODE == "move":
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def list_class_files(class_dir: Path) -> List[Path]:
    return sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS])


def split_groups(groups: List[str], train_r: float, val_r: float, test_r: float, rng: random.Random) -> Tuple[List[str], List[str], List[str]]:
    """
    Split des GROUPES (pas des images) pour empêcher la fuite.
    Les ratios sont approximatifs; on vise au mieux.
    """
    assert abs((train_r + val_r + test_r) - 1.0) < 1e-6

    n = len(groups)
    if n == 0:
        return [], [], []
    if n == 1:
        return groups, [], []
    if n == 2:
        return [groups[0]], [groups[1]], []
    if n == 3:
        return [groups[0]], [groups[1]], [groups[2]]

    rng.shuffle(groups)

    n_train = int(round(n * train_r))
    n_val = int(round(n * val_r))
    n_test = n - n_train - n_val

    # Ajustements pour éviter des splits vides si possible
    if n_test <= 0:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        else:
            n_val = max(1, n_val - 1)

    if n_val <= 0:
        n_val = 1
        if n_train > 1:
            n_train -= 1
        else:
            n_test = max(1, n_test - 1)

    # Re-garantir somme
    s = n_train + n_val + n_test
    if s != n:
        n_test += n - s

    train_g = groups[:n_train]
    val_g = groups[n_train : n_train + n_val]
    test_g = groups[n_train + n_val : n_train + n_val + n_test]
    return train_g, val_g, test_g


def split_files(files: List[Path], train_ratio: float, val_ratio: float, test_ratio: float, rng):
    files = files[:]
    rng.shuffle(files)

    n_total = len(files)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))

    n_train = min(n_train, n_total)
    n_val = min(n_val, n_total - n_train)

    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val]
    test_files = files[n_train + n_val :]

    return train_files, val_files, test_files


def main():
    assert MODE in {"copy", "move"}, "MODE doit être 'copy' ou 'move'"

    rng = random.Random(SEED)

    classes = sorted([p for p in DATA_DIR.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not classes:
        raise RuntimeError(f"Aucune classe trouvée dans {DATA_DIR}")

    totals = {"train": 0, "val": 0, "test": 0}
    split_checks = []

    for class_dir in classes:
        files = list_class_files(class_dir)
        if not files:
            print(f"[SKIP] {class_dir.name}: 0 fichiers")
            continue

        # Prépare dossiers de sortie
        out_train = OUT_DIR / "train" / class_dir.name
        out_val = OUT_DIR / "val" / class_dir.name
        out_test = OUT_DIR / "test" / class_dir.name
        ensure_dir(out_train)
        ensure_dir(out_val)
        ensure_dir(out_test)

        # -------------------------------------------------
        # CAS 1 : split par groupes (anti-fuite via filename)
        # -------------------------------------------------
        if GROUP_BY_FILENAME:
            grouped: Dict[str, List[Path]] = defaultdict(list)
            for f in files:
                grouped[group_key_from_filename(f)].append(f)

            group_ids = list(grouped.keys())
            train_g, val_g, test_g = split_groups(group_ids, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, rng)

            split_checks.append(
                {
                    "class": class_dir.name,
                    "mode": "grouped",
                    "n_files": len(files),
                    "n_groups": len(group_ids),
                    "train_units": len(train_g),
                    "val_units": len(val_g),
                    "test_units": len(test_g),
                }
            )

            def emit_grouped(split_name: str, g_list: List[str], out_base: Path):
                nonlocal totals
                count = 0
                for g in g_list:
                    for src in grouped[g]:
                        dst = out_base / src.name
                        transfer(src, dst)
                        count += 1
                totals[split_name] += count
                return count

            n_train_imgs = emit_grouped("train", train_g, out_train)
            n_val_imgs = emit_grouped("val", val_g, out_val)
            n_test_imgs = emit_grouped("test", test_g, out_test)

            print(f"[{class_dir.name}] mode=grouped images={len(files):4d} groups={len(group_ids):4d} -> train={n_train_imgs:4d} val={n_val_imgs:4d} test={n_test_imgs:4d}")

        # -------------------------------------------------
        # CAS 2 : split direct fichier par fichier
        # -------------------------------------------------
        else:
            train_files, val_files, test_files = split_files(files, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, rng)

            split_checks.append(
                {
                    "class": class_dir.name,
                    "mode": "filewise",
                    "n_files": len(files),
                    "n_groups": None,
                    "train_units": len(train_files),
                    "val_units": len(val_files),
                    "test_units": len(test_files),
                }
            )

            def emit_files(split_name: str, file_list: List[Path], out_base: Path):
                nonlocal totals
                count = 0
                for src in file_list:
                    dst = out_base / src.name
                    transfer(src, dst)
                    count += 1
                totals[split_name] += count
                return count

            n_train_imgs = emit_files("train", train_files, out_train)
            n_val_imgs = emit_files("val", val_files, out_val)
            n_test_imgs = emit_files("test", test_files, out_test)

            print(f"[{class_dir.name}] mode=filewise images={len(files):4d} -> train={n_train_imgs:4d} val={n_val_imgs:4d} test={n_test_imgs:4d}")

    print("\n=== RÉSUMÉ GLOBAL ===")
    print(f"Total train: {totals['train']}")
    print(f"Total val  : {totals['val']}")
    print(f"Total test : {totals['test']}")

    print("\n=== DÉTAIL SPLIT PAR CLASSE ===")
    for item in split_checks:
        if item["mode"] == "grouped":
            print(
                f"  {item['class']}: mode=grouped "
                f"files={item['n_files']} groups={item['n_groups']} -> "
                f"train_groups={item['train_units']}, "
                f"val_groups={item['val_units']}, "
                f"test_groups={item['test_units']}"
            )
        else:
            print(f"  {item['class']}: mode=filewise files={item['n_files']} -> train_files={item['train_units']}, val_files={item['val_units']}, test_files={item['test_units']}")

    if DRY_RUN:
        print("\n[DRY_RUN] Aucun fichier n'a été copié/déplacé.")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
