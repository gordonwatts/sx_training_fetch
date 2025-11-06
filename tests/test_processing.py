import awkward as ak

from calratio_training_data.processing import do_rotations


def test_do_rotations_clusters():
    # clusters and tracks have the same shape array so this is fine
    cluster_array = ak.Array(
        [
            [
                {"pt": 10.0, "eta": 0.5, "phi": 1.0},
                {"pt": 5.0, "eta": 2.3, "phi": 2.3},
            ],
            [
                {"pt": 20.0, "eta": -2.0, "phi": 2.1},
                {"pt": 4.0, "eta": 0.3, "phi": 6.1},
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
    correct_array = ak.Array(
        [
            [
                {"pt": 10.0, "eta": 0, "phi": 0},
                {"pt": 5.0, "eta": 1.8, "phi": 1.3},
            ],
            [
                {"pt": 20.0, "eta": 0, "phi": 0},
                {"pt": 4.0, "eta": 2.3, "phi": 2.28},
            ],
        ],
        with_name="Momentum3D",
    )
    do_rotations(cluster_array, "cluster", jet_array)

    # Using to_list to avoid problems with -0 ≠ 0 comparison
    assert (
        ak.round(cluster_array, 2).to_list() == correct_array.to_list()
    )  # rounded for comparison


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

    correct_array = ak.Array(
        [
            [
                {"pt": 10.0, "eta": 0.4, "phi": 1.42},
                {"pt": 5, "eta": 2.6, "phi": -2.5},
            ],
            [
                {"pt": 20.0, "eta": 1.3, "phi": 1.37},
                {"pt": 4.0, "eta": -2.52, "phi": -1.43},
            ],
        ],
        with_name="Momentum3D",
    )

    assert ak.array_equal(
        ak.round(track_array, 2), correct_array
    )  # rounded for comparison


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

    correct_array = ak.Array(
        [
            [
                {"etaPos": -1, "phiPos": -0.2, "phiDir": 1.02},
                {"etaPos": 0.8, "phiPos": 1.1, "phiDir": 0.7},
            ],
            [
                {"etaPos": 2.8, "phiPos": -2.41, "phiDir": 0.27},
                {"etaPos": 1.9, "phiPos": -0.33, "phiDir": 0.77},
            ],
        ]
    )

    assert ak.array_equal(
        ak.round(mseg_array, 2), correct_array
    )  # rounded for comparison
