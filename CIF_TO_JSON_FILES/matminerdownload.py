import os
import json
from pathlib import Path
from matminer.datasets import load_dataset
from pymatgen.core import Structure

dataset_names = [
    "boltztrap_mp",
    "brgoch_superhard_training",
    "castelli_perovskites",
    "jarvis_dft_3d",
    "jarvis_ml_dft_training",
    "phonon_dielectric_mp",
    "ricci_boltztrap_mp_tabular",
]

output_dir = Path("Data_process/JSON_files/0205matminer_data")
output_dir.mkdir(parents=True, exist_ok=True)

for name in dataset_names:
    print(f"\nProcessing dataset: {name}")
    try:
        df = load_dataset(name)
    except Exception as e:
        print(f"Failed to load dataset '{name}': {e}")
        continue

    if 'structure' not in df.columns:
        print(f"'structure' column not found in dataset '{name}'. Skipping.")
        continue

    valid_mask = df['structure'].apply(lambda x: isinstance(x, Structure))
    invalid_count = (~valid_mask).sum()

    if invalid_count > 0:
        print(
            f"{invalid_count} invalid entries found in '{name}'. These will be skipped.")

    df = df[valid_mask].reset_index(drop=True)

    if df.empty:
        print(
            f"No valid structures left in '{name}' after filtering. Skipping.")
        continue

    id_column = None
    for candidate in ["material_id", "mp_id", "id", "uid", "filename", "formula"]:
        if candidate in df.columns:
            id_column = candidate
            break

    for idx, row in df.iterrows():
        structure = row['structure']
        identifier = row[id_column] if id_column else f"{idx}"
        safe_identifier = str(identifier).replace(
            "/", "_").replace("\\", "_").replace(" ", "_")
        filename = output_dir / f"{name}_{safe_identifier}.json"

        try:
            with open(filename, "w", encoding="utf-8") as file:
                json.dump(structure.as_dict(), file)
        except Exception as e:
            print(f"Failed to save structure {idx} from dataset '{name}': {e}")
