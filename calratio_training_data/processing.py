import logging
from dataclasses import dataclass
from math import sqrt
from typing import Dict

import awkward as ak
import numpy as np
from vector._compute.planar.deltaphi import rectify

from calratio_training_data.constants import (
    min_jet_pt,
    max_jet_pt,
)


def relative_angle(jets: ak.Array, objects: ak.Array):
    # Modifies in place the eta and phi to be relative to the jet axis
    objects["eta"] = objects.deltaeta(jets)
    objects["phi"] = objects.deltaphi(jets)


def sort_by_pt(data: ak.Array) -> ak.Array:
    # Sorts the data in place by pT
    new_data_index = ak.argsort(data.pt, axis=1, ascending=False)
    data[new_data_index]


def do_rotations(data: ak.Array, datatype, jets=None) -> ak.Array:
    """
    Do rotations on clusters (tracks, msegs). Done to get the highest cluster (track, mseg) by pT is
    at the center. Ensures NN learns from a standardized set of jets.
    Assumed data is flattened to per jet (instead of per event).

    Args:
        data (ak.Array): Data to be rotated, contained either clusters, tracks, or msegs
        datatype (str): Type of data to be rotated, needed because msegs are rotated differently
            either: cluster, track, mseg
        jets (ak.Array): Set of jets, needed to do track and mseg rotation

    Returns:
        processed (dict[str, ak.Array]): Processed data
    """

    if datatype == "cluster" or datatype == "track":
        # Sort the data if its a cluster or a track
        sort_by_pt(data)
        if datatype == "cluster":
            relative_angle(ak.firsts(data), data)
        if datatype == "track":
            relative_angle(jets, data)

        # eta flip
        eta_sign = ak.sum(np.multiply(data.eta, data.pt), axis=1)
        eta_sign = ak.fill_none(eta_sign, 0)
        eta_sign = np.vectorize(lambda x: 1 * (x >= 0) + (-1) * (x < 0))(eta_sign)
        data["eta"] = data.eta * eta_sign

        # phi flip
        phi_sign = ak.sum(np.multiply(data.phi, data.pt), axis=1)
        phi_sign = ak.fill_none(phi_sign, 0)
        phi_sign = np.vectorize(lambda x: 1 * (x >= 0) + (-1) * (x < 0))(phi_sign)
        data["phi"] = data.phi * phi_sign

    if datatype == "mseg" and data is not None:
        # msegs occasionally empty - check for that else it crashes
        mseg_deta_pos = data.etaPos - jets.eta
        data["etaPos"] = mseg_deta_pos
        mseg_dphi_pos = rectify(np, data.phiPos - jets.phi)
        data["phiPos"] = mseg_dphi_pos
        mseg_dphi_dir = rectify(np, data.phiDir - jets.phi)
        data["phiDir"] = mseg_dphi_dir


def do_rescaling(data: ak.Array, datatype) -> ak.Array:
    """
    Rescaling variables for NN training. Shouldn't be run on msegs.
    Args:
        data (ak.Array): Partially processed data
                         containing either clusters, jets, or tracks.
        datatype (str): The type of data being processed (e.g., "cluster", "jet", "track").
    """

    if datatype == "cluster":
        # Rescaling cluster energy fraction
        summed_energy = (
            data.l1ecal
            + data.l2ecal
            + data.l3ecal
            + data.l4ecal
            + data.l1hcal
            + data.l2hcal
            + data.l3hcal
            + data.l4hcal
        )
        for i in range(1, 5):
            data[f"l{i}ecal"] = data[f"l{i}ecal"] / summed_energy
            data[f"l{i}hcal"] = data[f"l{i}hcal"] / summed_energy

    if datatype == "track":
        # rescale track z0 - restricts distribution to be around 0
        data["z0"] = data.z0 / 250

    # Rescaling pT
    sub_pt = data["pt"] - min_jet_pt
    data["pt"] = sub_pt / (max_jet_pt - min_jet_pt)
