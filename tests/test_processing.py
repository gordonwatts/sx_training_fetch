import pytest
import awkward as ak
import numpy as np

from calratio_training_data.processing import do_rescaling, do_rotations


def test_do_rotations_clusters():
    # clusters and tracks have the same shape array so this is fine
    cluster_array = ak.Array(
        [
            [
                {"eta": 0.5, "phi": 1.0, "pt": 10.0},
                {"eta": 2.3, "phi": 2.3, "pt": 5.0},
            ],
            [
                {"eta": -2.0, "phi": 1.3, "pt": 20.0},
                {"eta": 0.3, "phi": 6.1, "pt": 4.0},
            ],
        ],
        with_name="Momentum3D",
    )

    jet_array = ak.Array(
        [
            {
                "pt": 12000.0,
                "eta": 0.5,
                "phi": 1.2,
            },
            {
                "pt": 45000,
                "eta": -1.1,
                "phi": 2.8,
            },
        ],
        with_name="Momentum3D",
    )

    do_rotations(cluster_array, "cluster", jet_array)
    assert ak.all(np.abs(cluster_array.eta) < np.pi)
    assert ak.all(np.abs(cluster_array.phi) < np.pi)
    assert ak.all(cluster_array.eta[:, 0] == 0)
    assert ak.all(cluster_array.phi[:, 0] == 0)


def test_do_rotations_tracks():
    track_array = ak.Array(
        [
            [
                {"eta": 0.9, "phi": 8.9, "pt": 10.0},
                {"eta": 3.1, "phi": -1.3, "pt": 5.0},
            ],
            [
                {"eta": -2.4, "phi": 1.43, "pt": 20.0},
                {"eta": 1.42, "phi": 4.23, "pt": 4.0},
            ],
        ],
        with_name="Momentum3D",
    )

    jet_array = ak.Array(
        [
            {
                "pt": 12000.0,
                "eta": 0.5,
                "phi": 1.2,
            },
            {
                "pt": 45000,
                "eta": -1.1,
                "phi": 2.8,
            },
        ],
        with_name="Momentum3D",
    )

    do_rotations(track_array, "track", jet_array)

    assert ak.all(np.abs(track_array.eta) < np.pi)
    assert ak.all(np.abs(track_array.phi) < np.pi)


def test_do_rotations_msegs():
    mseg_array = ak.Array(
        [
            [
                {"etaPos": 0.5, "phiPos": 1.0, "phiDir": 8.5},
                {"etaPos": 2.3, "phiPos": 2.3, "phiDir": 1.9},
            ],
            [
                {"etaPos": 1.2, "phiPos": 6.3, "phiDir": 2.7},
                {"etaPos": 0.3, "phiPos": 2.1, "phiDir": 3.2},
            ],
        ]
    )

    jet_array = ak.Array(
        [
            {
                "pt": 12000.0,
                "eta": 1.5,
                "phi": 1.2,
            },
            {
                "pt": 45000,
                "eta": -1.6,
                "phi": 2.43,
            },
        ],
        with_name="Momentum3D",
    )

    do_rotations(mseg_array, "mseg", jet_array)

    assert ak.all(np.abs(mseg_array.etaPos) < np.pi)
    assert ak.all(np.abs(mseg_array.phiPos) < np.pi)
    assert ak.all(np.abs(mseg_array.phiDir) < np.pi)
