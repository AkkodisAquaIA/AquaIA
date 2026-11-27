import yaml


def print_train_config(config):
    """Prints config when training."""
    print("\n=== Training Config ===")
    for section_name in ("model", "data", "training", "output"):
        if section_name not in config:
            continue
        print(f"\n[{section_name}]")
        # yaml.safe_dump() transforms dict to YAML str
        # sort_key=False do not reorder alphabetically; default_flow_style=False: use multiline YAML
        print(yaml.safe_dump(config[section_name], sort_keys=False, default_flow_style=False).strip())
    print("=======================\n")
