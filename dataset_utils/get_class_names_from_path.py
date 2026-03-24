from pathlib import Path

ROOT_DIR = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/FIN Benthic2/IDA/Images_per_class"

for p in sorted(Path(ROOT_DIR).iterdir()):
    if p.is_dir():
        print(p.name)
