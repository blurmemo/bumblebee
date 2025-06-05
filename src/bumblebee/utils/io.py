import io
import os
import json
from pathlib import Path



def validate_file_path(file_path):
    path = Path(file_path)
    if not path.parent.exists():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise IOError(f"You can not create directory {path.parent}")

    if not os.access(path.parent, os.W_OK):
        raise IOError(f"You can not access {path.parent}.")



def read_json(json_path: str) -> dict:
    open_file = io.open if os.path.exists(json_path) else open
    with open_file(json_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    return config_dict


def write_json(data, output_file: str = None) -> None:
    if output_file is None:
        raise ValueError("`output_file` cannot be None.")
    validate_file_path(output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
