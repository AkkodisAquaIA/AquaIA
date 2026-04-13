import yaml


def print_train_config(config):
    print("\n=== Training Config ===")
    for section_name in ("model", "data", "training", "output"):
        if section_name not in config:
            continue
        print(f"\n[{section_name}]")
        print(yaml.safe_dump(config[section_name], sort_keys=False, default_flow_style=False).strip())
    print("=======================\n")
