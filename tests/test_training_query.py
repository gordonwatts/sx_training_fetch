import os
from pathlib import Path

import awkward as ak
import h5py
import numpy as np

from calratio_training_data.training_query import write_training_data_hdf5


def make_simple_ak_array():
    # Create a simple awkward array with two fields
    return ak.Array(
        {
            "pt": np.array([1.0, 2.0, 3.0]),
            "eta": np.array([0.1, 0.2, 0.3]),
            "phi": np.array([0.5, 0.6, 0.7]),
        }
    )


def test_write_training_data_hdf5(tmp_path):
    arr = make_simple_ak_array()
    output_file = tmp_path / "test.hdf5"
    write_training_data_hdf5(arr, output_file)
    assert output_file.exists()
    # Check contents
    with h5py.File(output_file, "r") as f:
        for field in arr.fields:
            np.testing.assert_array_equal(f[field][...], ak.to_numpy(arr[field]))  # type: ignore


def test_write_training_data_hdf5_empty(tmp_path):
    arr = ak.Array(
        {
            "pt": np.array([]),
            "eta": np.array([]),
            "phi": np.array([]),
        }
    )
    output_file = tmp_path / "empty.hdf5"
    write_training_data_hdf5(arr, output_file)
    assert output_file.exists()
    with h5py.File(output_file, "r") as f:
        for field in arr.fields:
            assert field in f
            assert f[field].shape[0] == 0


def test_write_training_data_hdf5_path_type(tmp_path):
    arr = make_simple_ak_array()
    output_file = str(tmp_path / "str_path.hdf5")
    # Should accept str as path
    write_training_data_hdf5(arr, output_file)
    assert os.path.exists(output_file)
    with h5py.File(output_file, "r") as f:
        for field in arr.fields:
            assert field in f
            np.testing.assert_array_equal(f[field][...], ak.to_numpy(arr[field]))  # type: ignore
