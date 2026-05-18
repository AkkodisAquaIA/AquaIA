from collections import Counter
from pathlib import Path
from utils import select_or_latest, load_yaml_from_result, load_label_txt

PARENT_FOLDER = Path(__file__).resolve().parent  # Folder containing this script

if __name__ == "__main__":
	# Select result_det folder or automatically use the latest one
	det_dir = select_or_latest(base_dir=PARENT_FOLDER, title="Select result_det folder (Cancel to use latest)")

	# Read config files
	_, _, dataset_dict = load_yaml_from_result(det_dir)

	# Recursively find all labels folders under the selected result folder
	label_dirs = sorted(label_dir for label_dir in (det_dir / "detection_result").rglob("labels") if label_dir.is_dir())

	class_counts = Counter()

	# For each labels folder, read all txt files
	for label_dir in label_dirs:
		txt_files = sorted(label_dir.glob("*.txt"))
		if not txt_files:
			continue

		# For each txt file
		for txt_file in txt_files:
			labels = load_label_txt(txt_file, with_conf=False, conf_threshold=None)
			if labels.shape[0] == 0:
				continue

			# Count only non-empty label folder, non-empty txt file
			classes = labels[:, 0].astype(int)
			class_counts.update(classes.tolist())

	output_path = det_dir / "docs_run" / "class_count.txt"
	with output_path.open("w", encoding="utf-8") as f:
		f.write("Predicted class counts:\n")
		f.write("(Class ID, Class name, Count)\n")

		total_predictions = 0
		for class_id in sorted(class_counts.keys()):
			class_name = dataset_dict.get(class_id, "UNKNOWN")
			count = class_counts[class_id]
			f.write(f"{class_id}, {class_name}, {count}\n")
			total_predictions += count

		f.write(f"\nTOTAL_PREDICTIONS, {total_predictions}")

	print(f"\nDone! Saved class counts to: {output_path}")
