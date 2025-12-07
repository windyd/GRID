import os
import shutil
from unittest.mock import patch

import pyarrow.parquet as pq
import rootutils
import torch
from lightning import LightningModule

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.inference_utils import LocalParquetWriter


class MockTrainer:
    def __init__(self):
        self.global_rank = 0


class MockModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.prediction_key_name = "id"
        self.prediction_name = "embedding"


class MockModelOutput:
    def __init__(self, rows):
        self.list_of_row_format = rows


@patch("src.utils.inference_utils.torch.distributed")
def test_local_parquet_writer(mock_dist):
    # Setup
    output_dir = "tests/temp_output_parquet"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    writer = LocalParquetWriter(
        output_dir=output_dir,
        flush_frequency=2,
        write_interval="batch",
        should_merge_files_on_main=True,
        should_merge_list_of_keyed_tensors_to_single_tensor=True,
        prediction_key_name="id",
        prediction_name="embedding",
    )

    module = MockModule()
    # Mocking global_rank manually via a simple class
    trainer = MockTrainer()

    writer.setup(trainer, module, "predict")

    # Create dummy data
    rows1 = [
        {"id": 0, "embedding": torch.tensor([0.1, 0.2], dtype=torch.float32)},
        {"id": 1, "embedding": torch.tensor([0.3, 0.4], dtype=torch.float32)},
    ]
    rows2 = [{"id": 2, "embedding": torch.tensor([0.5, 0.6], dtype=torch.float32)}]

    # Write batch 1 (triggers flush because len=2 and flush_freq=2)
    writer.write_on_batch_end(trainer, module, MockModelOutput(rows1), [], None, 0, 0)

    # Write batch 2 (buffered)
    writer.write_on_batch_end(trainer, module, MockModelOutput(rows2), [], None, 1, 0)

    # Finish prediction (flushes buffer and merges)
    writer.on_predict_end(trainer, module)

    # Check parquet file
    merged_file = os.path.join(output_dir, "merged_predictions.parquet")
    assert os.path.exists(merged_file), "Merged parquet file should exist"

    table = pq.read_table(merged_file)
    assert table.num_rows == 3, f"Expected 3 rows, got {table.num_rows}"

    # Check content
    pylist = table.to_pylist()
    # Sort by id to ensure order
    pylist.sort(key=lambda x: x["id"])

    assert pylist[0]["id"] == 0
    # Approximate check for floats
    assert len(pylist[0]["embedding"]) == 2
    assert abs(pylist[0]["embedding"][0] - 0.1) < 1e-6

    # Check tensor file
    tensor_file = os.path.join(output_dir, "merged_predictions_tensor.pt")
    assert os.path.exists(tensor_file), "Merged tensor file should exist"

    tensor_data = torch.load(tensor_file)
    # Expected shape: (3, 2)
    assert tensor_data.shape == (3, 2)
    assert torch.allclose(
        tensor_data[0], torch.tensor([0.1, 0.2], dtype=torch.float32), atol=1e-6
    )

    # Cleanup
    shutil.rmtree(output_dir)
    print("Test passed!")


if __name__ == "__main__":
    test_local_parquet_writer()
