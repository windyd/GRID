import gzip
import json
import os
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import rootutils
from torch.utils.data import DataLoader, get_worker_info

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.loading.components.dataloading import UnboundedSequenceIterable
from src.data.loading.components.iterators import (
    JsonlDataIterator,
    ParquetDataIterator,
)


class MockDatasetConfig:
    def __init__(self):
        self.iterate_per_row = True
        self.preprocessing_functions = []
        self.data_iterator = JsonlDataIterator()
        self.file_format = "jsonl.gz"


class MockParquetDatasetConfig:
    def __init__(self):
        self.iterate_per_row = True
        self.preprocessing_functions = []
        self.data_iterator = ParquetDataIterator()
        self.file_format = "parquet"


@pytest.fixture
def sharding_data_dir(tmp_path):
    data_dir = tmp_path / "repro_data"
    data_dir.mkdir()
    file_path = data_dir / "data.jsonl.gz"

    with gzip.open(file_path, "wt") as f:
        for i in range(10):
            f.write(json.dumps({"id": i, "content": f"item_{i}"}) + "\n")

    return data_dir, file_path


@pytest.fixture
def parquet_data_dir(tmp_path):
    data_dir = tmp_path / "repro_data_parquet"
    data_dir.mkdir()
    file_path = data_dir / "data.parquet"

    # Create data with 'id' and 'content'
    ids = list(range(10))
    contents = [f"item_{i}" for i in range(10)]

    table = pa.Table.from_pydict({"id": ids, "content": contents})

    pq.write_table(table, file_path)

    return data_dir, file_path


def test_row_level_sharding(sharding_data_dir):
    """
    Verifies that when `are_files_shared_across_processes=True`,
    data is correctly sharded across workers at the row level.
    """
    data_dir, file_path = sharding_data_dir

    dataset_config = MockDatasetConfig()

    # Simulate the condition where files are shared (e.g. 1 file, 2 workers)
    dataset = UnboundedSequenceIterable(
        dataset_config=dataset_config,
        data_folder=str(data_dir),
        should_shuffle_rows=False,
        batch_size=1,
        is_for_training=False,
        assign_all_files_per_worker=False,
        are_files_shared_across_processes=True,  # FORCE SHARDING
    )

    # Set file list explicitly
    dataset.set_list_of_files([str(file_path)])
    dataset.set_distributed_params(total_workers=1, global_worker_id=0)

    def worker_init_fn(worker_id):
        pass

    # Create DataLoader with 2 workers
    # If sharding works, they should split the 10 items (5 each)
    dataloader = DataLoader(
        dataset, batch_size=None, num_workers=2, worker_init_fn=worker_init_fn
    )

    items = []
    for item in dataloader:
        items.append(item)

    ids = sorted([x["id"] for x in items])

    assert len(items) == 10, f"Expected 10 items, got {len(items)}"
    assert ids == list(range(10)), f"IDs do not match. Got {ids}"


def test_parquet_row_level_sharding(parquet_data_dir):
    """
    Verifies that when `are_files_shared_across_processes=True`,
    Parquet data is correctly sharded across workers at the row level.
    """
    data_dir, file_path = parquet_data_dir

    dataset_config = MockParquetDatasetConfig()

    # Simulate the condition where files are shared (e.g. 1 file, 2 workers)
    dataset = UnboundedSequenceIterable(
        dataset_config=dataset_config,
        data_folder=str(data_dir),
        should_shuffle_rows=False,
        batch_size=1,
        is_for_training=False,
        assign_all_files_per_worker=False,
        are_files_shared_across_processes=True,  # FORCE SHARDING
    )

    # Set file list explicitly
    dataset.set_list_of_files([str(file_path)])
    dataset.set_distributed_params(total_workers=1, global_worker_id=0)

    def worker_init_fn(worker_id):
        pass

    # Create DataLoader with 2 workers
    # If sharding works, they should split the 10 items (5 each)
    dataloader = DataLoader(
        dataset, batch_size=None, num_workers=2, worker_init_fn=worker_init_fn
    )

    items = []
    for item in dataloader:
        items.append(item)

    # Extract IDs.
    ids = sorted([x["id"] for x in items])

    assert len(items) == 10, f"Expected 10 items, got {len(items)}"
    assert ids == list(range(10)), f"IDs do not match. Got {ids}"
