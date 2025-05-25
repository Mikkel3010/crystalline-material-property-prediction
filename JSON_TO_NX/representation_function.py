from typing import Union, Iterable
from itertools import islice
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import json
import os
from tqdm import tqdm
from multiprocessing import Pool
import gc
from matbench.bench import MatbenchBenchmark
from pymatgen.core.structure import Structure
from networkx import MultiDiGraph
from graphlist import GraphList, HDFGraphList
from kgcnn.crystal.preprocessor import CrystalPreprocessor


class MatbenchDataset:
    def __init__(self, cache_dir: Union[str, Path], task_type=None):
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir
        if not cache_dir.is_dir():
            print(f"Creating cache directory: {cache_dir}")
            cache_dir.mkdir(exist_ok=True)
        if task_type[0] == "pretrain_data":
            self.dataset_names = ["pretrain_data"]
        elif task_type[0] == "matminer":
            print("Loading matminer dataset")
            # self.dataset_names = ["Data_process/JSON_files/0205matminer_data"]
            self.dataset_names = [
                "/mnt/c/Users/mikke/Desktop/P6_new/Data_process/JSON_files/0205matminer_data"]
        else:
            self.matbench = MatbenchBenchmark(autoload=False)
            self.dataset_names = [
                task.dataset_name for task in self.matbench.tasks]

    @staticmethod
    def crystal_iterator(crystal_series: pd.Series):
        for id_, x in zip(crystal_series.index, crystal_series):
            # print(f"ID: {id_}")
            setattr(x, "dataset_id", id_.encode())
            yield x

    def dataset_exists(
        self, dataset_name: str, preprocessor: CrystalPreprocessor
    ) -> bool:
        print(f"Checking if {dataset_name} exists")
        print(f"Dataset names: {self.dataset_names}")
        preprocessor_hash = preprocessor.hash()
        dataset_cache_dir = self.cache_dir / dataset_name
        if not dataset_cache_dir.is_dir():
            return False
        hdf_file = dataset_cache_dir / (preprocessor_hash + ".h5")
        return hdf_file.is_file()

    def load_json_files(self, json_dir):
        json_objects = []
        files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
        ratio = 1
        files_count = int(np.floor(len(files)*ratio))
        print(f"Loading {files_count} JSON files")
        self.files_count = files_count

        files = files[:files_count]

        for filename in tqdm(files[:files_count], desc="Loading JSON files"):
            path = os.path.join(json_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    structure = Structure.from_dict(data)
                    # print(structure)
                    json_objects.append(structure)
            except Exception as e:
                print(f" Failed to load {filename}: {e}")
        # print(json_objects[:10])
        # print("objects in luist: ", [type(x) for x in json_objects[:10]])
        return pd.Series(json_objects, index=files[:])

    def get_dataset_file(
        self,
        dataset_name: str,
        preprocessor: CrystalPreprocessor,
        processes=8,
        batch_size=500,
    ):
        print(
            f"Getting dataset file for {dataset_name} with preprocessor {preprocessor}")
        print(f"Dataset names: {self.dataset_names}")
        # assert dataset_name in self.dataset_names
        preprocessor_hash = preprocessor.hash()
        dataset_cache_dir = self.cache_dir / dataset_name
        if not dataset_cache_dir.is_dir():
            dataset_cache_dir.mkdir(exist_ok=True)
        hdf_file = dataset_cache_dir / (preprocessor_hash + ".h5")
        if hdf_file.is_file():
            return hdf_file

        full_dataset = self._get_full_dataset(dataset_name)
        hdf_file_ = create_graph_dataset(
            self.crystal_iterator(full_dataset),
            preprocessor,
            hdf_file,
            additional_graph_attributes=["dataset_id"],
            processes=processes,
            batch_size=batch_size,
            total_count=self.files_count,
        )
        with open(str(hdf_file) + ".json", "w") as meta_data_file:
            json.dump(preprocessor.get_config(), meta_data_file)
        return hdf_file_

    def _get_full_dataset(self, dataset_name: str):
        if dataset_name == "pretrain_data":
            dataset_dir = self.dataset_names[0]
            all_crystals = self.load_json_files(dataset_dir)
        elif dataset_name == "matminer":
            dataset_dir = self.dataset_names[0]
            all_crystals = self.load_json_files(dataset_dir)
        else:
            task = getattr(self.matbench, dataset_name)
            task.load()
            train_input = task.get_train_and_val_data(0)[0]
            test_input = task.get_test_data(0)
            self.files_count = task.metadata["num_entries"]
            # print(train_input[1])
            # print(type(train_input[1]))
            # print("type(train_input): ", type(train_input))
            all_crystals = pd.concat([train_input, test_input]).sort_index()
            # print(all_crystals[:10])
            # print(type(all_crystals))
            # print([type(x) for x in all_crystals])
        return all_crystals


class PreprocessorWrapper:
    def __init__(self, preprocessor, additional_graph_attributes=[]):
        self.preprocessor = preprocessor
        self.additional_graph_attributes = additional_graph_attributes

    def __call__(self, crystal: Union[MultiDiGraph, Structure]):
        graph = self.preprocessor(crystal)
        for attribute in self.additional_graph_attributes:
            setattr(graph, attribute, getattr(crystal, attribute))
        return graph


def batcher(iterable, batch_size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def create_graph_dataset(
        crystals: Iterable[Structure],
        preprocessor: CrystalPreprocessor,
        out_file: Path,
        additional_graph_attributes=[],
        processes=None,
        batch_size=None,
        total_count=0) -> Path:
    worker = PreprocessorWrapper(
        preprocessor, additional_graph_attributes=additional_graph_attributes
    )
    total_batches = (total_count + batch_size - 1) // batch_size

    with h5py.File(str(out_file), "w") as f:
        for batch in tqdm(batcher(crystals, batch_size), total=total_batches, desc="Processing batches"):
            graphs = [worker(crystal) for crystal in batch]

            graphlist = GraphList.from_nx_graphs(
                graphs,
                node_attribute_names=preprocessor.node_attributes,
                edge_attribute_names=preprocessor.edge_attributes,
                graph_attribute_names=preprocessor.graph_attributes + additional_graph_attributes,
            )

            HDFGraphList.from_graphlist(f, graphlist)
            del graphs, graphlist
            gc.collect()
        f.attrs["preprocessor_config"] = json.dumps(
            preprocessor.get_config(), indent=2
        )
