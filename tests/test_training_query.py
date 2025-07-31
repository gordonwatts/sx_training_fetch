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


def make_complex_ak_array():
    # Create a simple awkward array with two fields
    tracks = ak.Array(
        [
            {
                "pt": [1.0, 2.0, 3.0, 4.0],
                "eta": [0.1, 0.2, 0.3, 0.4],
                "phi": [0.5, 0.6, 0.7, 0.8],
            },
            {
                "pt": [1.1],
                "eta": [0.11],
                "phi": [0.51],
            },
            {
                "pt": [1.2, 2.2],
                "eta": [0.12, 0.22],
                "phi": [0.52, 0.62],
            },
        ]
    )
    return ak.zip(
        {
            "pt": ak.Array([1.0, 2.0, 3.0]),
            "eta": ak.Array([0.1, 0.2, 0.3]),
            "phi": ak.Array([0.5, 0.6, 0.7]),
            "tracks": tracks,
        },
        depth_limit=1,
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


def test_write_complex_training_data_hdf5(tmp_path):
    arr = make_complex_ak_array()
    output_file = tmp_path / "test.hdf5"
    write_training_data_hdf5(arr, output_file)
    assert output_file.exists()

    with h5py.File(output_file, "r") as f:
        track_data = f["tracks"][...]  # type: ignore
        assert isinstance(track_data, h5py.Dataset)
        assert len(track_data) == 3  # Check number of tracks

        # Check that each has 4 tracks in it.
        assert all(len(track) == 4 for track in track_data)

        # Now, make sure that the 2nd one is padded zeros for entries 1, 2, and 3.
        assert np.all(track_data[1, 1:] == 0)
        assert track_data[1, 0] == 1.1  # Check first entry of second track

        # and for the 3rd track only the 2 and 3rd entries are zeros.
        assert np.all(track_data[2, 0] == 1.2)
        assert np.all(track_data[2, 1] == 2.2)
        assert np.all(track_data[2, 2:] == 0)  # Check

        np.testing.assert_array_equal(f["tracks"][...], ak.to_numpy(arr["tracks"]))  # type: ignore


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
