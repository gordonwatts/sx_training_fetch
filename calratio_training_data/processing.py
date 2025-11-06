from typing import Optional
import awkward as ak
import numpy as np
from vector._compute.planar.deltaphi import rectify


def relative_angle(jets: ak.Array, objects: ak.Array):
    # Modifies in place the eta and phi to be relative to the jet axis
    objects["eta"] = objects.deltaeta(jets)
    objects["phi"] = objects.deltaphi(jets)


def sort_by_pt(data: ak.Array) -> ak.Array:
    # Sorts the data in place by pT
    new_data_index = ak.argsort(data.pt, axis=1, ascending=False)
    data[new_data_index]


def do_rotations(data: ak.Array, datatype, jets: Optional[ak.Array] = None):
    """
    Do rotations on clusters (tracks, msegs). Done to get the highest cluster (track, mseg) by pT
    is at the center. Ensures NN learns from a standardized set of jets.
    Assumed data is flattened to per jet (instead of per event).

    WARNING: Data is rotated in place!!!

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
            assert jets is not None, "Jets must be provided for track rotation"
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
        assert jets is not None, "Jets must be provided for mseg rotation"
        mseg_deta_pos = data.etaPos - jets.eta
        data["etaPos"] = mseg_deta_pos
        mseg_dphi_pos = rectify(np, data.phiPos - jets.phi)
        data["phiPos"] = mseg_dphi_pos
        mseg_dphi_dir = rectify(np, data.phiDir - jets.phi)
        data["phiDir"] = mseg_dphi_dir
