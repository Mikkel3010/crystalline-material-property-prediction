from kgcnn.crystal.preprocessor import (
    RadiusAsymmetricUnitCell, KNNAsymmetricUnitCell, VoronoiAsymmetricUnitCell,
    RadiusNonPeriodicUnitCell, KNNNonPeriodicUnitCell, VoronoiNonPeriodicUnitCell,
    RadiusUnitCell, KNNUnitCell, VoronoiUnitCell
)
import pytz
from datetime import datetime
from representation_function import MatbenchDataset
import sys
from pathlib import Path
import time
from graphlist import HDFGraphList
import h5py
from multiprocessing import cpu_count


def process_matbench_dataset(dataset_cache_path, crystal_preprocessor, matbench_datasets_subset, timezone="Europe/Copenhagen", batch_size=1000):
    from datetime import datetime
    import pytz
    import time
    from multiprocessing import cpu_count
    from pathlib import Path

    print(f"\nUsing dataset cache: {dataset_cache_path}")
    print(f"Preprocessor: {crystal_preprocessor.__class__.__name__}")
    print(f"Using {cpu_count() - 2} CPU cores\n")

    # Setup timezone
    tz = pytz.timezone(timezone)

    # Initialize MatbenchDataset
    mb_dataset_cache = MatbenchDataset(
        Path(dataset_cache_path),
        task_type=matbench_datasets_subset
    )

    for task in matbench_datasets_subset:
        print(f"Starting task: {task}")
        start_time = time.time()

        preprocessed_crystals_file = mb_dataset_cache.get_dataset_file(
            task,
            crystal_preprocessor,
            processes=cpu_count() - 2,
            batch_size=batch_size
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        end_time_local = datetime.fromtimestamp(
            end_time, tz).strftime('%Y-%m-%d %H:%M:%S')

        print(
            f"✓ Done with {task} in {elapsed_time:.2f} seconds (finished at {end_time_local} {timezone})\n"
        )


# matminer datasets
dataset_all_mat_miner = [
    # (
    #     "/mnt/c/Users/mikke/Desktop/P6_new/Data_process/h5py_files/Matminer_all_KNN_UnitCell/",
    #     KNNUnitCell(24)
    # ),
    # (
    #     "/mnt/c/Users/mikke/Desktop/P6_new/Data_process/h5py_files/Matminer_all_KNN_ASYMMETRIC/",
    #     KNNAsymmetricUnitCell(24)
    # ),
    # (
    #     "/mnt/c/Users/mikke/Desktop/P6_new/Data_process/h5py_files/Matminer_all_KNN_NonPeriodic/",
    #     KNNNonPeriodicUnitCell(24)
    # ),
    # (
    #     "/mnt/c/Users/mikke/Desktop/P6_new/Data_process/h5py_files/Matminer_all_Radi_UnitCell/",
    #     RadiusUnitCell(6)
    # ),
    # (
    #     "/mnt/c/Users/mikke/Desktop/P6_new/Data_process/h5py_files/Matminer_all_Radi_ASYMMETRIC/",
    #     RadiusAsymmetricUnitCell(6)
    # ),
    # (
    #     "/mnt/c/Users/mikke/Desktop/P6_new/Data_process/h5py_files/Matminer_all_Radi_NonPeriodic/",
    #     RadiusNonPeriodicUnitCell(6)
    # ),
    # (
    #     "/mnt/c/Users/mikke/Desktop/P6_new/Data_process/h5py_files/Matminer_all_Voro_UnitCell/",
    #     VoronoiUnitCell(1e-6)
    # ),
    # (
    #     "/mnt/c/Users/mikke/Desktop/P6_new/Data_process/h5py_files/Matminer_all_Voro_ASYMMETRIC/",
    #     VoronoiAsymmetricUnitCell(1e-6)
    # ),
    # (
    #     "/mnt/c/Users/mikke/Desktop/P6_new/Data_process/h5py_files/Matminer_all_Voro_NonPeriodic/",
    #     VoronoiNonPeriodicUnitCell(1e-6)
    # ),
]


# perovskites datasets(finetune dataset)

dataset_matbench_perovskites = [
    (
        "/mnt/c/Users/mikke/Desktop/P6_new/Data_process/h5py_files/Matbench_matbench_perovskites_KNN_UnitCell/",
        KNNUnitCell(24)
    ),
    (
        "/mnt/c/Users/mikke/Desktop/P6_new/Data_process/h5py_files/Matbench_matbench_perovskites_KNN_ASYMMETRIC/",
        KNNAsymmetricUnitCell(24)
    ),
    (
        "/mnt/c/Users/mikke/Desktop/P6_new/Data_process/h5py_files/Matbench_matbench_perovskites_KNN_NonPeriodic/",
        KNNNonPeriodicUnitCell(24)
    ),
    (
        "/mnt/c/Users/mikke/Desktop/P6_new/Data_process/h5py_files/Matbench_matbench_perovskites_Radi_UnitCell/",
        RadiusUnitCell(6)
    ),
    (
        "/mnt/c/Users/mikke/Desktop/P6_new/Data_process/h5py_files/Matbench_matbench_perovskites_Radi_ASYMMETRIC/",
        RadiusAsymmetricUnitCell(6)
    ),
    (
        "/mnt/c/Users/mikke/Desktop/P6_new/Data_process/h5py_files/Matbench_matbench_perovskites_Radi_NonPeriodic/",
        RadiusNonPeriodicUnitCell(6)
    ),
    (
        "/mnt/c/Users/mikke/Desktop/P6_new/Data_process/h5py_files/Matbench_matbench_perovskites_Voro_UnitCell/",
        VoronoiUnitCell(1e-6)
    ),
    (
        "/mnt/c/Users/mikke/Desktop/P6_new/Data_process/h5py_files/Matbench_matbench_perovskites_Voro_ASYMMETRIC/",
        VoronoiAsymmetricUnitCell(1e-6)
    ),
    (
        "/mnt/c/Users/mikke/Desktop/P6_new/Data_process/h5py_files/Matbench_matbench_perovskites_Voro_NonPeriodic/",
        VoronoiNonPeriodicUnitCell(1e-6)
    )
]


dataset_preprocessor_pairs = dataset_matbench_perovskites


for dataset_cache_path, crystal_preprocessor in dataset_preprocessor_pairs:
    process_matbench_dataset(
        dataset_cache_path,
        crystal_preprocessor,
        ["matbench_perovskites"]
    )

# run with
# ["matbench_perovskites"]
# [matbench_jdft2d]
# ["matminer"]
