import gzip
import json
import os

import pytest
import rootutils
from torch.utils.data import DataLoader, get_worker_info

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.loading.components.dataloading import UnboundedSequenceIterable
from src.data.loading.components.iterators import JsonlDataIterator


class MockDatasetConfig:
    def __init__(self):
        self.iterate_per_row = True
        self.preprocessing_functions = []
        self.data_iterator = JsonlDataIterator()
        self.file_format = "jsonl.gz"


@pytest.fixture
def split_data_dir(tmp_path):
    data_dir = tmp_path / "repro_data_split"
    data_dir.mkdir()

    file_path_1 = data_dir / "data1.jsonl.gz"
    file_path_2 = data_dir / "data2.jsonl.gz"

    with gzip.open(file_path_1, "wt") as f:
        for i in range(5):
            f.write(json.dumps({"id": i, "content": f"item_{i}"}) + "\n")

    with gzip.open(file_path_2, "wt") as f:
        for i in range(5, 10):
            f.write(json.dumps({"id": i, "content": f"item_{i}"}) + "\n")

    return data_dir, file_path_1, file_path_2


def test_file_level_splitting(split_data_dir):
    """
    Verifies that when `are_files_shared_across_processes=False`,
    files are split across workers (no row-level sharding).
    """
    data_dir, file_path_1, file_path_2 = split_data_dir

    dataset_config = MockDatasetConfig()

    # Simulate normal condition: 2 files, 2 workers -> NO sharing, NO sharding
    dataset = UnboundedSequenceIterable(
        dataset_config=dataset_config,
        data_folder=str(data_dir),
        should_shuffle_rows=False,
        batch_size=1,
        is_for_training=False,
        assign_all_files_per_worker=False,
        are_files_shared_across_processes=False,
    )

    # Set file list explicitly
    dataset.set_list_of_files([str(file_path_1), str(file_path_2)])
    dataset.set_distributed_params(total_workers=1, global_worker_id=0)

    def worker_init_fn(worker_id):
        pass

    # Create DataLoader with 2 workers
    # Expected: Worker 0 gets file 1, Worker 1 gets file 2.
    dataloader = DataLoader(
        dataset, batch_size=None, num_workers=2, worker_init_fn=worker_init_fn
    )

    items = []
    for item in dataloader:
        items.append(item)

    ids = sorted([x["id"] for x in items])

    assert len(items) == 10, f"Expected 10 items, got {len(items)}"
    assert ids == list(range(10)), f"IDs do not match. Got {ids}"
