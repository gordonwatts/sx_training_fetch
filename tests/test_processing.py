import awkward as ak

from calratio_training_data.processing import do_rescaling, do_rotations


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

    assert ak.array_equal(
        ak.round(cluster_array, 2), correct_array
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


def test_do_rescaling_clusters():
    cluster_array = ak.Array(
        [
            [
                {
                    "pt": 10.0,
                    "l1ecal": 0.5,
                    "l2ecal": 1.0,
                    "l3ecal": 0.4,
                    "l4ecal": 1.3,
                    "l1hcal": 2.3,
                    "l2hcal": 0.23,
                    "l3hcal": 8.2,
                    "l4hcal": 9.1,
                },
                {
                    "pt": 16.0,
                    "l1ecal": 3.2,
                    "l2ecal": 1.3,
                    "l3ecal": 5.23,
                    "l4ecal": 5.2,
                    "l1hcal": 1.2,
                    "l2hcal": 4.23,
                    "l3hcal": 3.25,
                    "l4hcal": 9.2,
                },
            ],
            [
                {
                    "pt": 12.0,
                    "l1ecal": 3.2,
                    "l2ecal": 3.8,
                    "l3ecal": 3.2,
                    "l4ecal": 5.2,
                    "l1hcal": 1.3,
                    "l2hcal": 9.3,
                    "l3hcal": 2.4,
                    "l4hcal": 9.1,
                },
                {
                    "pt": 9.0,
                    "l1ecal": 1.2,
                    "l2ecal": 0.9,
                    "l3ecal": 2.23,
                    "l4ecal": 5.43,
                    "l1hcal": 7.32,
                    "l2hcal": 2.12,
                    "l3hcal": 8.23,
                    "l4hcal": 5.52,
                },
            ],
        ],
        with_name="Momentum3D",
    )

    do_rescaling(cluster_array, "cluster")

    correct_array = ak.Array(
        [
            [
                {
                    "l1ecal": 0.02,
                    "l1hcal": 0.1,
                    "l2ecal": 0.04,
                    "l2hcal": 0.01,
                    "l3ecal": 0.02,
                    "l3hcal": 0.36,
                    "l4ecal": 0.06,
                    "l4hcal": 0.4,
                    "pt": -0.07,
                },
                {
                    "l1ecal": 0.1,
                    "l1hcal": 0.04,
                    "l2ecal": 0.04,
                    "l2hcal": 0.13,
                    "l3ecal": 0.16,
                    "l3hcal": 0.1,
                    "l4ecal": 0.16,
                    "l4hcal": 0.28,
                    "pt": -0.05,
                },
            ],
            [
                {
                    "l1ecal": 0.09,
                    "l1hcal": 0.03,
                    "l2ecal": 0.1,
                    "l2hcal": 0.25,
                    "l3ecal": 0.09,
                    "l3hcal": 0.06,
                    "l4ecal": 0.14,
                    "l4hcal": 0.24,
                    "pt": -0.06,
                },
                {
                    "l1ecal": 0.04,
                    "l1hcal": 0.22,
                    "l2ecal": 0.03,
                    "l2hcal": 0.06,
                    "l3ecal": 0.07,
                    "l3hcal": 0.25,
                    "l4ecal": 0.16,
                    "l4hcal": 0.17,
                    "pt": -0.07,
                },
            ],
        ],
        with_name="Momentum3D",
    )

    assert ak.array_equal(ak.round(cluster_array, 2), correct_array)


def test_do_rescaling_tracks():
    track_array = ak.Array(
        [
            [
                {"pt": 10.0, "z0": 530},
                {"pt": 5.0, "z0": 490},
            ],
            [
                {"pt": 20.0, "z0": 500},
                {"pt": 4.0, "z0": 550},
            ],
        ],
        with_name="Momentum3D",
    )

    do_rescaling(track_array, "track")

    correct_array = ak.Array(
        [
            [
                {
                    "z0": 2.12,
                    "pt": -0.065,
                },
                {
                    "z0": 1.96,
                    "pt": -0.076,
                },
            ],
            [
                {
                    "z0": 2,
                    "pt": -0.043,
                },
                {
                    "z0": 2.2,
                    "pt": -0.078,
                },
            ],
        ],
        with_name="Momentum3D",
    )

    assert ak.array_equal(ak.round(track_array, 3), correct_array)
