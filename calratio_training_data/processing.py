import logging
from dataclasses import dataclass
from math import sqrt
from typing import Dict

import awkward as ak
import numpy as np

from calratio_training_data.constants import (
    min_jet_pt,
    max_jet_pt,
)


def processing_func(data: ak.Array, datatype, doRotation=True) -> ak.Array:
    """Process the data for training.

    Args:
        data (ak.Array): Partially processed data
                         containing either clusters, jets, or tracks.
        datatype (str): The type of data being processed (e.g., "cluster", "jet", "track").
        doRotation (bool): Whether to apply geometric rotation to the phi/eta variables.

    Returns:
        processed (dict[str, ak.Array]): Processed data
    """

    # Flattening data to do operations on it
    flat_data = ak.flatten(data, axis=1)
    processed_data = {}

    # Doing eta phi flip on clusters
    # Ensures highest pT cluster is at the center
    if datatype == "cluster" or datatype == "track":
        if doRotation:
            # eta flip
            cluster_sign = ak.sum(np.multiply(flat_data.eta, flat_data.pt), axis=1)
            cluster_sign = np.vectorize(lambda x: 1 * (x >= 0) + (-1) * (x < 0))(
                cluster_sign
            )
            processed_data["eta"] = flat_data.eta * cluster_sign

            # phi flip
            cluster_sign = ak.sum(np.multiply(flat_data.phi, flat_data.pt), axis=1)
            cluster_sign = np.vectorize(lambda x: 1 * (x >= 0) + (-1) * (x < 0))(
                cluster_sign
            )
            processed_data["phi"] = flat_data.phi * cluster_sign

        if datatype == "cluster":
            # Rescaling cluster energy fraction
            summed_energy = (
                flat_data.l1ecal
                + flat_data.l2ecal
                + flat_data.l3ecal
                + flat_data.l4ecal
                + flat_data.l1hcal
                + flat_data.l2hcal
                + flat_data.l3hcal
                + flat_data.l4hcal
            )
            for i in range(1, 5):
                processed_data[f"l{i}ecal"] = flat_data[f"l{i}ecal"] / summed_energy
                processed_data[f"l{i}hcal"] = flat_data[f"l{i}hcal"] / summed_energy

        if datatype == "track":
            # rescale track z0 - restricts distribution to be around 0
            processed_data["z0"] = flat_data.z0 / 250

    # Rescaling pT
    sub_pt = flat_data.pt - min_jet_pt
    processed_data["pt"] = sub_pt / (max_jet_pt - min_jet_pt)

    return processed_data
