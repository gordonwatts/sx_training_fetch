import awkward as ak
import numpy as np

from calratio_training_data.training_query import convert_to_training_data
from calratio_training_data.fetch import DataType


def test_convert_to_training_data_mc_no_rotation():
    """Test convert_to_training_data with datatype=SIGNAL and rotation=False."""
    # Create minimal input data that matches the expected structure
    # The data needs to be an awkward Record so it supports both dict-style and attribute access
    raw_data_dict = {
        # Event info
        "runNumber": ak.Array([123456]),
        "eventNumber": ak.Array([789012]),
        "mcEventWeight": ak.Array([1.0]),
        # Jets - 2 jets in the event
        "jet_pt": ak.Array([[50.0, 60.0]]),
        "jet_eta": ak.Array([[0.5, 1.2]]),
        "jet_phi": ak.Array([[1.0, 2.0]]),
        # Tracks - some tracks in the event
        "track_pT": ak.Array([[10.0, 15.0, 20.0]]),
        "track_eta": ak.Array([[0.4, 0.6, 1.1]]),
        "track_phi": ak.Array([[0.9, 1.1, 1.9]]),
        "track_vertex_nParticles": ak.Array([[3, 3, 3]]),
        "track_d0": ak.Array([[0.1, 0.2, 0.3]]),
        "track_z0": ak.Array([[0.5, 0.6, 0.7]]),
        "track_chiSquared": ak.Array([[1.0, 1.5, 2.0]]),
        "track_PixelShared": ak.Array([[0, 1, 0]]),
        "track_SCTShared": ak.Array([[0, 0, 1]]),
        "track_PixelHoles": ak.Array([[0, 0, 0]]),
        "track_SCTHoles": ak.Array([[0, 1, 0]]),
        "track_PixelHits": ak.Array([[3, 4, 3]]),
        "track_SCTHits": ak.Array([[8, 8, 7]]),
        # Muon segments
        "MSeg_x": ak.Array([[100.0, 200.0]]),
        "MSeg_y": ak.Array([[50.0, 100.0]]),
        "MSeg_z": ak.Array([[300.0, 400.0]]),
        "MSeg_px": ak.Array([[10.0, 15.0]]),
        "MSeg_py": ak.Array([[5.0, 7.0]]),
        "MSeg_pz": ak.Array([[30.0, 40.0]]),
        "MSeg_t0": ak.Array([[0.0, 1.0]]),
        "MSeg_chiSquared": ak.Array([[1.2, 1.5]]),
        # Clusters - nested list matching jets (2 jets, each with 2 clusters)
        "clus_eta": ak.Array([[[0.5, 0.6, 1.2, 1.3]]]),
        "clus_phi": ak.Array([[[1.0, 1.1, 2.0, 2.1]]]),
        "clus_pt": ak.Array([[[5.0, 6.0, 7.0, 8.0]]]),
        "clus_l1hcal": ak.Array([[[100.0, 110.0, 120.0, 130.0]]]),
        "clus_l2hcal": ak.Array([[[200.0, 210.0, 220.0, 230.0]]]),
        "clus_l3hcal": ak.Array([[[300.0, 310.0, 320.0, 330.0]]]),
        "clus_l4hcal": ak.Array([[[400.0, 410.0, 420.0, 430.0]]]),
        "clus_l1ecal": ak.Array([[[500.0, 510.0, 520.0, 530.0]]]),
        "clus_l2ecal": ak.Array([[[600.0, 610.0, 620.0, 630.0]]]),
        "clus_l3ecal": ak.Array([[[700.0, 710.0, 720.0, 730.0]]]),
        "clus_l4ecal": ak.Array([[[800.0, 810.0, 820.0, 830.0]]]),
        "clus_time": ak.Array([[[-14, -4, 4, 14]]]),
        # LLP truth particles - 1 LLP in the event, close to first jet
        # First jet is at (eta=0.5, phi=1.0), so LLP should be nearby (deltaR < 0.4)
        # Also Lxy must be between 1200-4000 for central region (eta < 1.4)
        "LLP_eta": ak.Array([[0.52]]),  # Very close to first jet's eta=0.5
        "LLP_phi": ak.Array([[1.02]]),  # Very close to first jet's phi=1.0
        "LLP_pt": ak.Array([[100.0]]),
        "LLP_pdgid": ak.Array([[35]]),
        "LLP_Lz": ak.Array([[1500.0]]),
        "LLP_Lxy": ak.Array([[1500.0]]),  # Within valid range 1200-4000
    }

    # Convert to awkward Record to support both dict and attribute access
    raw_data = ak.Array([raw_data_dict])[0]

    ds_name = "ds_test_mH23_ms13"

    # Call the function
    result = convert_to_training_data(
        raw_data, DataType.SIGNAL, ds_name, rotation=False
    )

    # Basic checks - ensure the function runs without error and returns an array
    assert result is not None
    assert isinstance(result, ak.Array)

    # Check that we have some jets in the output
    assert len(result) > 0

    # Check that expected fields are present
    assert "runNumber" in ak.fields(result)
    assert "eventNumber" in ak.fields(result)
    assert "mcEventWeight" in ak.fields(result)
    assert "pt" in ak.fields(result)
    assert "eta" in ak.fields(result)
    assert "phi" in ak.fields(result)
    assert "tracks" in ak.fields(result)
    assert "clusters" in ak.fields(result)
    assert "msegs" in ak.fields(result)
    assert "llp" in ak.fields(result)


def test_convert_to_training_no_llps():
    """Test convert_to_training_data with datatype=SIGNAL and rotation=False."""
    # Create minimal input data that matches the expected structure
    # The data needs to be an awkward Record so it supports both dict-style and attribute access
    raw_data_dict = {
        # Event info
        "runNumber": ak.Array([123456]),
        "eventNumber": ak.Array([789012]),
        "mcEventWeight": ak.Array([1.0]),
        # Jets - 2 jets in the event
        "jet_pt": ak.Array([[50.0, 60.0]]),
        "jet_eta": ak.Array([[0.5, 1.2]]),
        "jet_phi": ak.Array([[1.0, 2.0]]),
        # Tracks - some tracks in the event
        "track_pT": ak.Array([[10.0, 15.0, 20.0]]),
        "track_eta": ak.Array([[0.4, 0.6, 1.1]]),
        "track_phi": ak.Array([[0.9, 1.1, 1.9]]),
        "track_vertex_nParticles": ak.Array([[3, 3, 3]]),
        "track_d0": ak.Array([[0.1, 0.2, 0.3]]),
        "track_z0": ak.Array([[0.5, 0.6, 0.7]]),
        "track_chiSquared": ak.Array([[1.0, 1.5, 2.0]]),
        "track_PixelShared": ak.Array([[0, 1, 0]]),
        "track_SCTShared": ak.Array([[0, 0, 1]]),
        "track_PixelHoles": ak.Array([[0, 0, 0]]),
        "track_SCTHoles": ak.Array([[0, 1, 0]]),
        "track_PixelHits": ak.Array([[3, 4, 3]]),
        "track_SCTHits": ak.Array([[8, 8, 7]]),
        # Muon segments
        "MSeg_x": ak.Array([[100.0, 200.0]]),
        "MSeg_y": ak.Array([[50.0, 100.0]]),
        "MSeg_z": ak.Array([[300.0, 400.0]]),
        "MSeg_px": ak.Array([[10.0, 15.0]]),
        "MSeg_py": ak.Array([[5.0, 7.0]]),
        "MSeg_pz": ak.Array([[30.0, 40.0]]),
        "MSeg_t0": ak.Array([[0.0, 1.0]]),
        "MSeg_chiSquared": ak.Array([[1.2, 1.5]]),
        # Clusters - nested list matching jets (2 jets, each with 2 clusters)
        "clus_eta": ak.Array([[[0.5, 0.6, 1.2, 1.3]]]),
        "clus_phi": ak.Array([[[1.0, 1.1, 2.0, 2.1]]]),
        "clus_pt": ak.Array([[[5.0, 6.0, 7.0, 8.0]]]),
        "clus_l1hcal": ak.Array([[[100.0, 110.0, 120.0, 130.0]]]),
        "clus_l2hcal": ak.Array([[[200.0, 210.0, 220.0, 230.0]]]),
        "clus_l3hcal": ak.Array([[[300.0, 310.0, 320.0, 330.0]]]),
        "clus_l4hcal": ak.Array([[[400.0, 410.0, 420.0, 430.0]]]),
        "clus_l1ecal": ak.Array([[[500.0, 510.0, 520.0, 530.0]]]),
        "clus_l2ecal": ak.Array([[[600.0, 610.0, 620.0, 630.0]]]),
        "clus_l3ecal": ak.Array([[[700.0, 710.0, 720.0, 730.0]]]),
        "clus_l4ecal": ak.Array([[[800.0, 810.0, 820.0, 830.0]]]),
        "clus_time": ak.Array([[[-14, -4, 4, 14]]]),
        # No LLPs at all.
        "LLP_eta": ak.Array([]),
        "LLP_phi": ak.Array([]),
        "LLP_pt": ak.Array([]),
        "LLP_pdgid": ak.Array([]),
        "LLP_Lz": ak.Array([]),
        "LLP_Lxy": ak.Array([]),
    }

    # Convert to awkward Record to support both dict and attribute access
    raw_data = ak.Array([raw_data_dict])[0]

    ds_name = "ds_test_mH23_ms13"

    # Call the function
    result = convert_to_training_data(
        raw_data, DataType.SIGNAL, ds_name, rotation=False
    )

    # Basic checks - ensure the function runs without error and returns an array
    assert result is not None
    assert isinstance(result, ak.Array)

    # Check that we have some jets in the output
    assert len(result) == 0


def test_convert_to_training_data_bib_select_min_emf():
    """Test that BIB datatype selects only the jet with minimum EMF per event."""
    # Create test data with 3 jets per event with different EMF values
    # Jet 0: EMF=0.3 (highest)
    # Jet 1: EMF=0.1 (minimum - should be selected)
    # Jet 2: EMF=0.2 (middle)
    raw_data_dict = {
        # Event info
        "runNumber": ak.Array([123456]),
        "eventNumber": ak.Array([789012]),
        "mcEventWeight": ak.Array([1.0]),
        # Jets - 3 jets in the event with different EMF values
        "jet_pt": ak.Array([[50.0, 60.0, 70.0]]),
        "jet_eta": ak.Array([[0.5, 1.2, 0.8]]),
        "jet_phi": ak.Array([[1.0, 2.0, 1.5]]),
        # Tracks - 3 tracks in the event
        "track_pT": ak.Array([[10.0, 15.0, 20.0]]),
        "track_eta": ak.Array([[0.4, 0.6, 1.1]]),
        "track_phi": ak.Array([[0.9, 1.1, 1.9]]),
        "track_vertex_nParticles": ak.Array([[3, 3, 3]]),
        "track_d0": ak.Array([[0.1, 0.2, 0.3]]),
        "track_z0": ak.Array([[0.5, 0.6, 0.7]]),
        "track_chiSquared": ak.Array([[1.0, 1.5, 2.0]]),
        "track_PixelShared": ak.Array([[0, 1, 0]]),
        "track_SCTShared": ak.Array([[0, 0, 1]]),
        "track_PixelHoles": ak.Array([[0, 0, 0]]),
        "track_SCTHoles": ak.Array([[0, 1, 0]]),
        "track_PixelHits": ak.Array([[3, 4, 3]]),
        "track_SCTHits": ak.Array([[8, 8, 7]]),
        # Muon segments
        "MSeg_x": ak.Array([[100.0, 200.0]]),
        "MSeg_y": ak.Array([[50.0, 100.0]]),
        "MSeg_z": ak.Array([[300.0, 400.0]]),
        "MSeg_px": ak.Array([[10.0, 15.0]]),
        "MSeg_py": ak.Array([[5.0, 7.0]]),
        "MSeg_pz": ak.Array([[30.0, 40.0]]),
        "MSeg_t0": ak.Array([[0.0, 1.0]]),
        "MSeg_chiSquared": ak.Array([[1.2, 1.5]]),
        # Clusters - 3 jets with different numbers of clusters
        "clus_eta": ak.Array([[[0.5, 0.6], [1.2, 1.3], [0.8, 0.9]]]),
        "clus_phi": ak.Array([[[1.0, 1.1], [2.0, 2.1], [1.5, 1.6]]]),
        "clus_pt": ak.Array([[[5.0, 6.0], [7.0, 8.0], [6.5, 7.5]]]),
        "clus_l1hcal": ak.Array([[[100.0, 110.0], [120.0, 130.0], [115.0, 125.0]]]),
        "clus_l2hcal": ak.Array([[[200.0, 210.0], [220.0, 230.0], [215.0, 225.0]]]),
        "clus_l3hcal": ak.Array([[[300.0, 310.0], [320.0, 330.0], [315.0, 325.0]]]),
        "clus_l4hcal": ak.Array([[[400.0, 410.0], [420.0, 430.0], [415.0, 425.0]]]),
        "clus_l1ecal": ak.Array([[[500.0, 510.0], [520.0, 530.0], [515.0, 525.0]]]),
        "clus_l2ecal": ak.Array([[[600.0, 610.0], [620.0, 630.0], [615.0, 625.0]]]),
        "clus_l3ecal": ak.Array([[[700.0, 710.0], [720.0, 730.0], [715.0, 725.0]]]),
        "clus_l4ecal": ak.Array([[[800.0, 810.0], [820.0, 830.0], [815.0, 825.0]]]),
        "clus_time": ak.Array([[[-14.0, -4.0], [4.0, 14.0], [0.0, 10.0]]]),
        # BIB-specific: EMF fractions for the 3 jets
        # Jet 1 (index 1) has minimum EMF of 0.1
        "jet_emf": ak.Array([[0.3, 0.1, 0.2]]),
    }

    # Convert to awkward Record to support both dict and attribute access
    raw_data = ak.Array([raw_data_dict])[0]

    ds_name = "data24_dataset"

    # Call the function
    result = convert_to_training_data(raw_data, DataType.BIB, ds_name, rotation=False)

    # Verify the function runs and returns an array
    assert result is not None
    assert isinstance(result, ak.Array)

    # Verify only one jet is in the output (the minimum EMF jet)
    assert len(result) == 1

    # Verify it's the correct jet (jet 1 with eta=1.2, phi=2.0, pt=60.0)
    assert abs(float(result.eta[0]) - 1.2) < 0.001
    assert abs(float(result.phi[0]) - 2.0) < 0.001
    assert abs(float(result.pt[0]) - 60.0) < 0.001

    # Verify expected fields are present
    assert "runNumber" in ak.fields(result)
    assert "eventNumber" in ak.fields(result)
    assert "mcEventWeight" in ak.fields(result)
    assert "tracks" in ak.fields(result)
    assert "clusters" in ak.fields(result)
    assert "msegs" in ak.fields(result)

    # Verify jet_emf is NOT in the output (it's only for selection, not part of output)
    assert "jet_emf" not in ak.fields(result)


def test_convert_to_training_data_bib_multiple_events():
    """Test BIB EMF selection with multiple events to verify per-event selection."""
    # Create test data with 2 events, each with 3 jets
    raw_data_dict = {
        # Event info
        "runNumber": ak.Array([123456, 123457]),
        "eventNumber": ak.Array([789012, 789013]),
        "mcEventWeight": ak.Array([1.0, 1.0]),
        # Jets - 3 jets per event
        "jet_pt": ak.Array([[50.0, 60.0, 70.0], [45.0, 55.0, 65.0]]),
        "jet_eta": ak.Array([[0.5, 1.2, 0.8], [0.3, 1.1, 0.9]]),
        "jet_phi": ak.Array([[1.0, 2.0, 1.5], [0.9, 1.9, 1.4]]),
        # Tracks
        "track_pT": ak.Array([[10.0, 15.0, 20.0], [12.0, 17.0, 22.0]]),
        "track_eta": ak.Array([[0.4, 0.6, 1.1], [0.3, 0.5, 1.0]]),
        "track_phi": ak.Array([[0.9, 1.1, 1.9], [0.8, 1.0, 1.8]]),
        "track_vertex_nParticles": ak.Array([[3, 3, 3], [3, 3, 3]]),
        "track_d0": ak.Array([[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]]),
        "track_z0": ak.Array([[0.5, 0.6, 0.7], [0.55, 0.65, 0.75]]),
        "track_chiSquared": ak.Array([[1.0, 1.5, 2.0], [1.1, 1.6, 2.1]]),
        "track_PixelShared": ak.Array([[0, 1, 0], [1, 0, 1]]),
        "track_SCTShared": ak.Array([[0, 0, 1], [1, 0, 0]]),
        "track_PixelHoles": ak.Array([[0, 0, 0], [0, 0, 0]]),
        "track_SCTHoles": ak.Array([[0, 1, 0], [1, 0, 1]]),
        "track_PixelHits": ak.Array([[3, 4, 3], [4, 3, 4]]),
        "track_SCTHits": ak.Array([[8, 8, 7], [7, 8, 8]]),
        # Muon segments
        "MSeg_x": ak.Array([[100.0, 200.0], [110.0, 210.0]]),
        "MSeg_y": ak.Array([[50.0, 100.0], [60.0, 110.0]]),
        "MSeg_z": ak.Array([[300.0, 400.0], [310.0, 410.0]]),
        "MSeg_px": ak.Array([[10.0, 15.0], [11.0, 16.0]]),
        "MSeg_py": ak.Array([[5.0, 7.0], [6.0, 8.0]]),
        "MSeg_pz": ak.Array([[30.0, 40.0], [31.0, 41.0]]),
        "MSeg_t0": ak.Array([[0.0, 1.0], [0.5, 1.5]]),
        "MSeg_chiSquared": ak.Array([[1.2, 1.5], [1.3, 1.6]]),
        # Clusters
        "clus_eta": ak.Array(
            [[[0.5, 0.6], [1.2, 1.3], [0.8, 0.9]], [[0.3, 0.4], [1.1, 1.2], [0.9, 1.0]]]
        ),
        "clus_phi": ak.Array(
            [[[1.0, 1.1], [2.0, 2.1], [1.5, 1.6]], [[0.9, 1.0], [1.9, 2.0], [1.4, 1.5]]]
        ),
        "clus_pt": ak.Array(
            [[[5.0, 6.0], [7.0, 8.0], [6.5, 7.5]], [[5.5, 6.5], [7.5, 8.5], [7.0, 8.0]]]
        ),
        "clus_l1hcal": ak.Array(
            [
                [[100.0, 110.0], [120.0, 130.0], [115.0, 125.0]],
                [[105.0, 115.0], [125.0, 135.0], [120.0, 130.0]],
            ]
        ),
        "clus_l2hcal": ak.Array(
            [
                [[200.0, 210.0], [220.0, 230.0], [215.0, 225.0]],
                [[205.0, 215.0], [225.0, 235.0], [220.0, 230.0]],
            ]
        ),
        "clus_l3hcal": ak.Array(
            [
                [[300.0, 310.0], [320.0, 330.0], [315.0, 325.0]],
                [[305.0, 315.0], [325.0, 335.0], [320.0, 330.0]],
            ]
        ),
        "clus_l4hcal": ak.Array(
            [
                [[400.0, 410.0], [420.0, 430.0], [415.0, 425.0]],
                [[405.0, 415.0], [425.0, 435.0], [420.0, 430.0]],
            ]
        ),
        "clus_l1ecal": ak.Array(
            [
                [[500.0, 510.0], [520.0, 530.0], [515.0, 525.0]],
                [[505.0, 515.0], [525.0, 535.0], [520.0, 530.0]],
            ]
        ),
        "clus_l2ecal": ak.Array(
            [
                [[600.0, 610.0], [620.0, 630.0], [615.0, 625.0]],
                [[605.0, 615.0], [625.0, 635.0], [620.0, 630.0]],
            ]
        ),
        "clus_l3ecal": ak.Array(
            [
                [[700.0, 710.0], [720.0, 730.0], [715.0, 725.0]],
                [[705.0, 715.0], [725.0, 735.0], [720.0, 730.0]],
            ]
        ),
        "clus_l4ecal": ak.Array(
            [
                [[800.0, 810.0], [820.0, 830.0], [815.0, 825.0]],
                [[805.0, 815.0], [825.0, 835.0], [820.0, 830.0]],
            ]
        ),
        "clus_time": ak.Array(
            [
                [[-14.0, -4.0], [4.0, 14.0], [0.0, 10.0]],
                [[-13.0, -3.0], [5.0, 15.0], [1.0, 11.0]],
            ]
        ),
        # BIB-specific: EMF fractions
        # Event 1: Jet 0 has minimum EMF=0.15
        # Event 2: Jet 2 has minimum EMF=0.12
        "jet_emf": ak.Array([[0.15, 0.25, 0.35], [0.32, 0.28, 0.12]]),
    }

    # Convert to awkward Record
    raw_data = ak.Array([raw_data_dict])[0]

    ds_name = "data24_dataset"

    # Call the function with BIB datatype
    result = convert_to_training_data(raw_data, DataType.BIB, ds_name, rotation=False)

    # Verify the function runs and returns an array
    assert result is not None
    assert isinstance(result, ak.Array)

    # Verify we have 2 jets in the output (one per event, each the minimum EMF jet)
    assert len(result) == 2

    # Verify first selected jet is from event 1, jet 0 (eta=0.5, phi=1.0, pt=50.0)
    assert abs(float(result.eta[0]) - 0.5) < 0.001
    assert abs(float(result.phi[0]) - 1.0) < 0.001
    assert abs(float(result.pt[0]) - 50.0) < 0.001

    # Verify second selected jet is from event 2, jet 2 (eta=0.9, phi=1.4, pt=65.0)
    assert abs(float(result.eta[1]) - 0.9) < 0.001
    assert abs(float(result.phi[1]) - 1.4) < 0.001
    assert abs(float(result.pt[1]) - 65.0) < 0.001

    # Verify mcEventWeight is set to 1.0 for BIB data
    assert abs(float(result.mcEventWeight[0]) - 1.0) < 0.001
    assert abs(float(result.mcEventWeight[1]) - 1.0) < 0.001


def test_convert_to_training_data_cr_applies_emf_mask_per_jet():
    """Test CR e-mu event selection and per-jet EMF selection."""
    raw_data_dict = {
        "runNumber": ak.Array([123456, 123457]),
        "eventNumber": ak.Array([789012, 789013]),
        "mcEventWeight": ak.Array([0.5, 1.5]),
        "jet_pt": ak.Array([[50.0, 60.0], [70.0, 80.0]]),
        "jet_eta": ak.Array([[0.5, 1.2], [0.8, 1.1]]),
        "jet_phi": ak.Array([[1.0, 2.0], [1.5, 2.5]]),
        "track_pT": ak.Array([[10.0, 15.0], [12.0, 17.0]]),
        "track_eta": ak.Array([[0.4, 1.1], [0.7, 1.0]]),
        "track_phi": ak.Array([[0.9, 1.9], [1.4, 2.4]]),
        "track_vertex_nParticles": ak.Array([[2, 2], [2, 2]]),
        "track_d0": ak.Array([[0.1, 0.2], [0.15, 0.25]]),
        "track_z0": ak.Array([[0.5, 0.6], [0.55, 0.65]]),
        "track_chiSquared": ak.Array([[1.0, 1.5], [1.1, 1.6]]),
        "track_PixelShared": ak.Array([[0, 1], [1, 0]]),
        "track_SCTShared": ak.Array([[0, 0], [1, 0]]),
        "track_PixelHoles": ak.Array([[0, 0], [0, 0]]),
        "track_SCTHoles": ak.Array([[0, 1], [1, 0]]),
        "track_PixelHits": ak.Array([[3, 4], [4, 3]]),
        "track_SCTHits": ak.Array([[8, 8], [7, 8]]),
        "MSeg_x": ak.Array([[100.0], [110.0]]),
        "MSeg_y": ak.Array([[50.0], [60.0]]),
        "MSeg_z": ak.Array([[300.0], [310.0]]),
        "MSeg_px": ak.Array([[10.0], [11.0]]),
        "MSeg_py": ak.Array([[5.0], [6.0]]),
        "MSeg_pz": ak.Array([[30.0], [31.0]]),
        "MSeg_t0": ak.Array([[0.0], [0.5]]),
        "MSeg_chiSquared": ak.Array([[1.2], [1.3]]),
        "clus_eta": ak.Array([[[0.5], [1.2]], [[0.8], [1.1]]]),
        "clus_phi": ak.Array([[[1.0], [2.0]], [[1.5], [2.5]]]),
        "clus_pt": ak.Array([[[5.0], [7.0]], [[6.5], [8.5]]]),
        "clus_l1hcal": ak.Array([[[100.0], [120.0]], [[115.0], [125.0]]]),
        "clus_l2hcal": ak.Array([[[200.0], [220.0]], [[215.0], [225.0]]]),
        "clus_l3hcal": ak.Array([[[300.0], [320.0]], [[315.0], [325.0]]]),
        "clus_l4hcal": ak.Array([[[400.0], [420.0]], [[415.0], [425.0]]]),
        "clus_l1ecal": ak.Array([[[500.0], [520.0]], [[515.0], [525.0]]]),
        "clus_l2ecal": ak.Array([[[600.0], [620.0]], [[615.0], [625.0]]]),
        "clus_l3ecal": ak.Array([[[700.0], [720.0]], [[715.0], [725.0]]]),
        "clus_l4ecal": ak.Array([[[800.0], [820.0]], [[815.0], [825.0]]]),
        "clus_time": ak.Array([[[-14.0], [4.0]], [[0.0], [10.0]]]),
        "electron_charge": ak.Array([[1], [1]]),
        "muon_charge": ak.Array([[-1], [1]]),
        "jet_emf": ak.Array([[0.99, 0.2], [0.99, 0.99]]),
    }

    raw_data = ak.Array([raw_data_dict])[0]

    result = convert_to_training_data(
        raw_data, DataType.CR_TTBAR, "ttbar_dataset", rotation=False
    )

    assert len(result) == 1
    assert int(result.runNumber[0]) == 123456
    assert int(result.eventNumber[0]) == 789012
    assert abs(float(result.mcEventWeight[0]) - 0.5) < 0.001
    assert abs(float(result.pt[0]) - 50.0) < 0.001
    assert abs(float(result.eta[0]) - 0.5) < 0.001
    assert abs(float(result.phi[0]) - 1.0) < 0.001

def test_convert_to_training_data_ttbar_applies_emf_mask_per_jet():
    """Test TTBAR per-jet EMF selection (hadronic jets, EMF < 0.97)."""
    raw_data_dict = {
        "runNumber": ak.Array([123456, 123457]),
        "eventNumber": ak.Array([789012, 789013]),
        "mcEventWeight": ak.Array([0.5, 1.5]),
        "jet_pt": ak.Array([[50.0, 60.0], [70.0, 80.0]]),
        "jet_eta": ak.Array([[0.5, 1.2], [0.8, 1.1]]),
        "jet_phi": ak.Array([[1.0, 2.0], [1.5, 2.5]]),
        "track_pT": ak.Array([[10.0, 15.0], [12.0, 17.0]]),
        "track_eta": ak.Array([[0.4, 1.1], [0.7, 1.0]]),
        "track_phi": ak.Array([[0.9, 1.9], [1.4, 2.4]]),
        "track_vertex_nParticles": ak.Array([[2, 2], [2, 2]]),
        "track_d0": ak.Array([[0.1, 0.2], [0.15, 0.25]]),
        "track_z0": ak.Array([[0.5, 0.6], [0.55, 0.65]]),
        "track_chiSquared": ak.Array([[1.0, 1.5], [1.1, 1.6]]),
        "track_PixelShared": ak.Array([[0, 1], [1, 0]]),
        "track_SCTShared": ak.Array([[0, 0], [1, 0]]),
        "track_PixelHoles": ak.Array([[0, 0], [0, 0]]),
        "track_SCTHoles": ak.Array([[0, 1], [1, 0]]),
        "track_PixelHits": ak.Array([[3, 4], [4, 3]]),
        "track_SCTHits": ak.Array([[8, 8], [7, 8]]),
        "MSeg_x": ak.Array([[100.0], [110.0]]),
        "MSeg_y": ak.Array([[50.0], [60.0]]),
        "MSeg_z": ak.Array([[300.0], [310.0]]),
        "MSeg_px": ak.Array([[10.0], [11.0]]),
        "MSeg_py": ak.Array([[5.0], [6.0]]),
        "MSeg_pz": ak.Array([[30.0], [31.0]]),
        "MSeg_t0": ak.Array([[0.0], [0.5]]),
        "MSeg_chiSquared": ak.Array([[1.2], [1.3]]),
        "clus_eta": ak.Array([[[0.5], [1.2]], [[0.8], [1.1]]]),
        "clus_phi": ak.Array([[[1.0], [2.0]], [[1.5], [2.5]]]),
        "clus_pt": ak.Array([[[5.0], [7.0]], [[6.5], [8.5]]]),
        "clus_l1hcal": ak.Array([[[100.0], [120.0]], [[115.0], [125.0]]]),
        "clus_l2hcal": ak.Array([[[200.0], [220.0]], [[215.0], [225.0]]]),
        "clus_l3hcal": ak.Array([[[300.0], [320.0]], [[315.0], [325.0]]]),
        "clus_l4hcal": ak.Array([[[400.0], [420.0]], [[415.0], [425.0]]]),
        "clus_l1ecal": ak.Array([[[500.0], [520.0]], [[515.0], [525.0]]]),
        "clus_l2ecal": ak.Array([[[600.0], [620.0]], [[615.0], [625.0]]]),
        "clus_l3ecal": ak.Array([[[700.0], [720.0]], [[715.0], [725.0]]]),
        "clus_l4ecal": ak.Array([[[800.0], [820.0]], [[815.0], [825.0]]]),
        "clus_time": ak.Array([[[-14.0], [4.0]], [[0.0], [10.0]]]),
        "jet_emf": ak.Array([[0.2, 0.99], [0.5, 0.8]]),
    }

    raw_data = ak.Array([raw_data_dict])[0]

    result = convert_to_training_data(
        raw_data, DataType.TTBAR, "ttbar_dataset", rotation=False, desc_label="ttbar_dataset"
    )

    # Event 0 keeps 1 jet (EMF 0.2 < 0.97), event 1 keeps 2 jets (EMF 0.5, 0.8).
    assert len(result) == 3
    assert int(result.runNumber[0]) == 123456
    assert int(result.eventNumber[0]) == 789012
    assert abs(float(result.mcEventWeight[0]) - 0.5) < 0.001
    assert abs(float(result.pt[0]) - 50.0) < 0.001
    assert all(label == 3 for label in result.label)
    assert result.desc_label[0] == "ttbar_dataset"


def test_convert_to_training_no_near_llps():
    """Test convert_to_training_data with datatype=SIGNAL and rotation=False."""
    # Create minimal input data that matches the expected structure
    # The data needs to be an awkward Record so it supports both dict-style and attribute access
    raw_data_dict = {
        # Event info
        "runNumber": ak.Array([123456]),
        "eventNumber": ak.Array([789012]),
        "mcEventWeight": ak.Array([1.0]),
        # Jets - 2 jets in the event
        "jet_pt": ak.Array([[50.0, 60.0]]),
        "jet_eta": ak.Array([[0.5, 1.2]]),
        "jet_phi": ak.Array([[1.0, 2.0]]),
        # Tracks - some tracks in the event
        "track_pT": ak.Array([[10.0, 15.0, 20.0]]),
        "track_eta": ak.Array([[0.4, 0.6, 1.1]]),
        "track_phi": ak.Array([[0.9, 1.1, 1.9]]),
        "track_vertex_nParticles": ak.Array([[3, 3, 3]]),
        "track_d0": ak.Array([[0.1, 0.2, 0.3]]),
        "track_z0": ak.Array([[0.5, 0.6, 0.7]]),
        "track_chiSquared": ak.Array([[1.0, 1.5, 2.0]]),
        "track_PixelShared": ak.Array([[0, 1, 0]]),
        "track_SCTShared": ak.Array([[0, 0, 1]]),
        "track_PixelHoles": ak.Array([[0, 0, 0]]),
        "track_SCTHoles": ak.Array([[0, 1, 0]]),
        "track_PixelHits": ak.Array([[3, 4, 3]]),
        "track_SCTHits": ak.Array([[8, 8, 7]]),
        # Muon segments
        "MSeg_x": ak.Array([[100.0, 200.0]]),
        "MSeg_y": ak.Array([[50.0, 100.0]]),
        "MSeg_z": ak.Array([[300.0, 400.0]]),
        "MSeg_px": ak.Array([[10.0, 15.0]]),
        "MSeg_py": ak.Array([[5.0, 7.0]]),
        "MSeg_pz": ak.Array([[30.0, 40.0]]),
        "MSeg_t0": ak.Array([[0.0, 1.0]]),
        "MSeg_chiSquared": ak.Array([[1.2, 1.5]]),
        # Clusters - nested list matching jets (2 jets, each with 2 clusters)
        "clus_eta": ak.Array([[[0.5, 0.6, 1.2, 1.3]]]),
        "clus_phi": ak.Array([[[1.0, 1.1, 2.0, 2.1]]]),
        "clus_pt": ak.Array([[[5.0, 6.0, 7.0, 8.0]]]),
        "clus_l1hcal": ak.Array([[[100.0, 110.0, 120.0, 130.0]]]),
        "clus_l2hcal": ak.Array([[[200.0, 210.0, 220.0, 230.0]]]),
        "clus_l3hcal": ak.Array([[[300.0, 310.0, 320.0, 330.0]]]),
        "clus_l4hcal": ak.Array([[[400.0, 410.0, 420.0, 430.0]]]),
        "clus_l1ecal": ak.Array([[[500.0, 510.0, 520.0, 530.0]]]),
        "clus_l2ecal": ak.Array([[[600.0, 610.0, 620.0, 630.0]]]),
        "clus_l3ecal": ak.Array([[[700.0, 710.0, 720.0, 730.0]]]),
        "clus_l4ecal": ak.Array([[[800.0, 810.0, 820.0, 830.0]]]),
        "clus_time": ak.Array([[[-14, -4, 4, 14]]]),
        # Also Lxy must be between 1200-4000 for central region (eta < 1.4)
        # No LLP's near any jets
        "LLP_eta": ak.Array([[1.52]]),  # Very far from first jet's eta=0.5
        "LLP_phi": ak.Array([[1.02]]),  # Very close to first jet's phi=1.0
        "LLP_pt": ak.Array([[100.0]]),
        "LLP_pdgid": ak.Array([[35]]),
        "LLP_Lz": ak.Array([[1500.0]]),
        "LLP_Lxy": ak.Array([[1500.0]]),  # Within valid range 1200-4000
    }

    # Convert to awkward Record to support both dict and attribute access
    raw_data = ak.Array([raw_data_dict])[0]

    ds_name = "ds_test_mH23_ms13"

    # Call the function
    result = convert_to_training_data(
        raw_data, DataType.SIGNAL, ds_name, rotation=False
    )

    # Basic checks - ensure the function runs without error and returns an array
    assert result is not None
    assert isinstance(result, ak.Array)

    # Check that we have some jets in the output
    assert len(result) == 0


def test_track_near_jet_selection():
    """Test that only tracks within JET_TRACK_DELTA_R (0.2) of a jet are selected."""
    # Jet at eta=0.5, phi=1.0
    # Track 0: eta=0.55, phi=1.05 (pt=11.0) -> delta_r = sqrt(0.05^2 + 0.05^2) = 0.0707 < 0.2 (matched)
    # Track 1: eta=0.45, phi=0.95 (pt=12.0) -> delta_r = sqrt(0.05^2 + 0.05^2) = 0.0707 < 0.2 (matched)
    # Track 2: eta=0.80, phi=1.00 (pt=13.0) -> delta_r = 0.300 > 0.2 (excluded, eta difference)
    # Track 3: eta=0.50, phi=1.30 (pt=14.0) -> delta_r = 0.300 > 0.2 (excluded, phi difference)
    # Track 4: eta=0.80, phi=1.30 (pt=15.0) -> delta_r = 0.424 > 0.2 (excluded, both)
    raw_data_dict = {
        "runNumber": ak.Array([123456]),
        "eventNumber": ak.Array([789012]),
        "mcEventWeight": ak.Array([1.0]),
        "jet_pt": ak.Array([[50.0]]),
        "jet_eta": ak.Array([[0.5]]),
        "jet_phi": ak.Array([[1.0]]),
        "track_pT": ak.Array([[11.0, 12.0, 13.0, 14.0, 15.0]]),
        "track_eta": ak.Array([[0.55, 0.45, 0.80, 0.50, 0.80]]),
        "track_phi": ak.Array([[1.05, 0.95, 1.00, 1.30, 1.30]]),
        "track_vertex_nParticles": ak.Array([[5, 5, 5, 5, 5]]),
        "track_d0": ak.Array([[0.1, 0.2, 0.3, 0.4, 0.5]]),
        "track_z0": ak.Array([[0.5, 0.6, 0.7, 0.8, 0.9]]),
        "track_chiSquared": ak.Array([[1.1, 1.2, 1.3, 1.4, 1.5]]),
        "track_PixelShared": ak.Array([[0, 1, 0, 0, 0]]),
        "track_SCTShared": ak.Array([[0, 0, 1, 0, 0]]),
        "track_PixelHoles": ak.Array([[0, 0, 0, 1, 0]]),
        "track_SCTHoles": ak.Array([[0, 0, 0, 0, 1]]),
        "track_PixelHits": ak.Array([[3, 4, 3, 4, 3]]),
        "track_SCTHits": ak.Array([[8, 7, 8, 7, 8]]),
        "MSeg_x": ak.Array([[]]),
        "MSeg_y": ak.Array([[]]),
        "MSeg_z": ak.Array([[]]),
        "MSeg_px": ak.Array([[]]),
        "MSeg_py": ak.Array([[]]),
        "MSeg_pz": ak.Array([[]]),
        "MSeg_t0": ak.Array([[]]),
        "MSeg_chiSquared": ak.Array([[]]),
        "clus_eta": ak.Array([[[0.5]]]),
        "clus_phi": ak.Array([[[1.0]]]),
        "clus_pt": ak.Array([[[5.0]]]),
        "clus_l1hcal": ak.Array([[[100.0]]]),
        "clus_l2hcal": ak.Array([[[200.0]]]),
        "clus_l3hcal": ak.Array([[[300.0]]]),
        "clus_l4hcal": ak.Array([[[400.0]]]),
        "clus_l1ecal": ak.Array([[[500.0]]]),
        "clus_l2ecal": ak.Array([[[600.0]]]),
        "clus_l3ecal": ak.Array([[[700.0]]]),
        "clus_l4ecal": ak.Array([[[800.0]]]),
        "clus_time": ak.Array([[[-14.0]]]),
    }
    raw_data = ak.Array([raw_data_dict])[0]
    result = convert_to_training_data(raw_data, DataType.QCD, "test_ds", rotation=False)

    assert len(result) == 1
    # Only track 0 and track 1 should be matched to the jet
    assert len(result.tracks[0]) == 2
    matched_pts = list(result.tracks[0].pt)
    assert abs(matched_pts[0] - 11.0) < 0.001
    assert abs(matched_pts[1] - 12.0) < 0.001

    # Verify track properties are preserved
    assert abs(float(result.tracks[0].eta[0]) - 0.55) < 0.001
    assert abs(float(result.tracks[0].phi[0]) - 1.05) < 0.001
    assert abs(float(result.tracks[0].d0[0]) - 0.1) < 0.001
    assert abs(float(result.tracks[0].z0[0]) - 0.5) < 0.001
    assert int(result.tracks[0].PixelShared[1]) == 1
    assert int(result.tracks[0].PixelHits[0]) == 3


def test_track_near_jet_selection_multi_jet():
    """Test track matching with multiple jets in the same event."""
    # Jet 0 at (0.5, 1.0), Jet 1 at (-1.0, -2.0)
    # Track 0 near Jet 0 only (0.52, 1.02)
    # Track 1 near Jet 1 only (-0.98, -1.98)
    # Track 2 far from both (2.0, 0.0)
    raw_data_dict = {
        "runNumber": ak.Array([123456]),
        "eventNumber": ak.Array([789012]),
        "mcEventWeight": ak.Array([1.0]),
        "jet_pt": ak.Array([[50.0, 60.0]]),
        "jet_eta": ak.Array([[0.5, -1.0]]),
        "jet_phi": ak.Array([[1.0, -2.0]]),
        "track_pT": ak.Array([[10.0, 20.0, 30.0]]),
        "track_eta": ak.Array([[0.52, -0.98, 2.00]]),
        "track_phi": ak.Array([[1.02, -1.98, 0.00]]),
        "track_vertex_nParticles": ak.Array([[3, 3, 3]]),
        "track_d0": ak.Array([[0.1, 0.2, 0.3]]),
        "track_z0": ak.Array([[0.5, 0.6, 0.7]]),
        "track_chiSquared": ak.Array([[1.0, 1.1, 1.2]]),
        "track_PixelShared": ak.Array([[0, 0, 0]]),
        "track_SCTShared": ak.Array([[0, 0, 0]]),
        "track_PixelHoles": ak.Array([[0, 0, 0]]),
        "track_SCTHoles": ak.Array([[0, 0, 0]]),
        "track_PixelHits": ak.Array([[3, 3, 3]]),
        "track_SCTHits": ak.Array([[8, 8, 8]]),
        "MSeg_x": ak.Array([[]]),
        "MSeg_y": ak.Array([[]]),
        "MSeg_z": ak.Array([[]]),
        "MSeg_px": ak.Array([[]]),
        "MSeg_py": ak.Array([[]]),
        "MSeg_pz": ak.Array([[]]),
        "MSeg_t0": ak.Array([[]]),
        "MSeg_chiSquared": ak.Array([[]]),
        "clus_eta": ak.Array([[[0.5], [-1.0]]]),
        "clus_phi": ak.Array([[[1.0], [-2.0]]]),
        "clus_pt": ak.Array([[[5.0], [6.0]]]),
        "clus_l1hcal": ak.Array([[[100.0], [100.0]]]),
        "clus_l2hcal": ak.Array([[[200.0], [200.0]]]),
        "clus_l3hcal": ak.Array([[[300.0], [300.0]]]),
        "clus_l4hcal": ak.Array([[[400.0], [400.0]]]),
        "clus_l1ecal": ak.Array([[[500.0], [500.0]]]),
        "clus_l2ecal": ak.Array([[[600.0], [600.0]]]),
        "clus_l3ecal": ak.Array([[[700.0], [700.0]]]),
        "clus_l4ecal": ak.Array([[[800.0], [800.0]]]),
        "clus_time": ak.Array([[[-14.0], [-14.0]]]),
    }
    raw_data = ak.Array([raw_data_dict])[0]
    result = convert_to_training_data(raw_data, DataType.QCD, "test_ds", rotation=False)

    assert len(result) == 2
    # Jet 0 should have only Track 0
    assert len(result.tracks[0]) == 1
    assert abs(float(result.tracks[0].pt[0]) - 10.0) < 0.001

    # Jet 1 should have only Track 1
    assert len(result.tracks[1]) == 1
    assert abs(float(result.tracks[1].pt[0]) - 20.0) < 0.001


def test_track_near_jet_selection_phi_wraparound():
    """Test that track delta_r calculation properly respects phi wraparound across [-pi, pi]."""
    # Jet at phi = 3.1
    # Track 0 at phi = -3.1 -> delta_phi across branch cut is ~0.083 < 0.2 (matched)
    # Track 1 at phi = 2.7 -> delta_phi is 0.400 > 0.2 (excluded)
    raw_data_dict = {
        "runNumber": ak.Array([123456]),
        "eventNumber": ak.Array([789012]),
        "mcEventWeight": ak.Array([1.0]),
        "jet_pt": ak.Array([[50.0]]),
        "jet_eta": ak.Array([[0.0]]),
        "jet_phi": ak.Array([[3.1]]),
        "track_pT": ak.Array([[10.0, 20.0]]),
        "track_eta": ak.Array([[0.05, 0.00]]),
        "track_phi": ak.Array([[-3.1, 2.7]]),
        "track_vertex_nParticles": ak.Array([[2, 2]]),
        "track_d0": ak.Array([[0.1, 0.2]]),
        "track_z0": ak.Array([[0.5, 0.6]]),
        "track_chiSquared": ak.Array([[1.0, 1.1]]),
        "track_PixelShared": ak.Array([[0, 0]]),
        "track_SCTShared": ak.Array([[0, 0]]),
        "track_PixelHoles": ak.Array([[0, 0]]),
        "track_SCTHoles": ak.Array([[0, 0]]),
        "track_PixelHits": ak.Array([[3, 3]]),
        "track_SCTHits": ak.Array([[8, 8]]),
        "MSeg_x": ak.Array([[]]),
        "MSeg_y": ak.Array([[]]),
        "MSeg_z": ak.Array([[]]),
        "MSeg_px": ak.Array([[]]),
        "MSeg_py": ak.Array([[]]),
        "MSeg_pz": ak.Array([[]]),
        "MSeg_t0": ak.Array([[]]),
        "MSeg_chiSquared": ak.Array([[]]),
        "clus_eta": ak.Array([[[0.0]]]),
        "clus_phi": ak.Array([[[3.1]]]),
        "clus_pt": ak.Array([[[5.0]]]),
        "clus_l1hcal": ak.Array([[[100.0]]]),
        "clus_l2hcal": ak.Array([[[200.0]]]),
        "clus_l3hcal": ak.Array([[[300.0]]]),
        "clus_l4hcal": ak.Array([[[400.0]]]),
        "clus_l1ecal": ak.Array([[[500.0]]]),
        "clus_l2ecal": ak.Array([[[600.0]]]),
        "clus_l3ecal": ak.Array([[[700.0]]]),
        "clus_l4ecal": ak.Array([[[800.0]]]),
        "clus_time": ak.Array([[[-14.0]]]),
    }
    raw_data = ak.Array([raw_data_dict])[0]
    result = convert_to_training_data(raw_data, DataType.QCD, "test_ds", rotation=False)

    assert len(result) == 1
    assert len(result.tracks[0]) == 1
    assert abs(float(result.tracks[0].pt[0]) - 10.0) < 0.001


def test_mseg_near_jet_selection():
    """Test that only muon segments with abs(delta_phi) < JET_MSEG_DELTA_PHI (0.2) are selected.

    Specifically verifies that negative delta_phi with absolute value >= 0.2 is correctly excluded,
    which fixes the bug where delta_phi < 0.2 mistakenly accepted large negative delta_phi.
    """
    # Jet at eta=0.0, phi=1.0
    # MSeg 0: phi = 1.10 (delta_phi = -0.10, abs = 0.10 < 0.2) -> INCLUDED
    # MSeg 1: phi = 0.90 (delta_phi = +0.10, abs = 0.10 < 0.2) -> INCLUDED
    # MSeg 2: phi = 1.50 (delta_phi = -0.50, abs = 0.50 > 0.2) -> EXCLUDED (buggy code included this!)
    # MSeg 3: phi = 0.50 (delta_phi = +0.50, abs = 0.50 > 0.2) -> EXCLUDED
    # MSeg 4: phi = -2.00 (delta_phi ~ -3.00, abs = 3.00 > 0.2) -> EXCLUDED (buggy code included this!)
    phis = [1.10, 0.90, 1.50, 0.50, -2.00]
    r = 100.0

    raw_data_dict = {
        "runNumber": ak.Array([123456]),
        "eventNumber": ak.Array([789012]),
        "mcEventWeight": ak.Array([1.0]),
        "jet_pt": ak.Array([[50.0]]),
        "jet_eta": ak.Array([[0.0]]),
        "jet_phi": ak.Array([[1.0]]),
        "track_pT": ak.Array([[]]),
        "track_eta": ak.Array([[]]),
        "track_phi": ak.Array([[]]),
        "track_vertex_nParticles": ak.Array([[]]),
        "track_d0": ak.Array([[]]),
        "track_z0": ak.Array([[]]),
        "track_chiSquared": ak.Array([[]]),
        "track_PixelShared": ak.Array([[]]),
        "track_SCTShared": ak.Array([[]]),
        "track_PixelHoles": ak.Array([[]]),
        "track_SCTHoles": ak.Array([[]]),
        "track_PixelHits": ak.Array([[]]),
        "track_SCTHits": ak.Array([[]]),
        "MSeg_x": ak.Array([[r * np.cos(p) for p in phis]]),
        "MSeg_y": ak.Array([[r * np.sin(p) for p in phis]]),
        "MSeg_z": ak.Array([[0.0, 10.0, 20.0, 30.0, 40.0]]),
        "MSeg_px": ak.Array([[10.0, 10.0, 10.0, 10.0, 10.0]]),
        "MSeg_py": ak.Array([[5.0, 5.0, 5.0, 5.0, 5.0]]),
        "MSeg_pz": ak.Array([[30.0, 30.0, 30.0, 30.0, 30.0]]),
        "MSeg_t0": ak.Array([[0.0, 1.0, 2.0, 3.0, 4.0]]),
        "MSeg_chiSquared": ak.Array([[1.0, 1.1, 1.2, 1.3, 1.4]]),
        "clus_eta": ak.Array([[[0.0]]]),
        "clus_phi": ak.Array([[[1.0]]]),
        "clus_pt": ak.Array([[[5.0]]]),
        "clus_l1hcal": ak.Array([[[100.0]]]),
        "clus_l2hcal": ak.Array([[[200.0]]]),
        "clus_l3hcal": ak.Array([[[300.0]]]),
        "clus_l4hcal": ak.Array([[[400.0]]]),
        "clus_l1ecal": ak.Array([[[500.0]]]),
        "clus_l2ecal": ak.Array([[[600.0]]]),
        "clus_l3ecal": ak.Array([[[700.0]]]),
        "clus_l4ecal": ak.Array([[[800.0]]]),
        "clus_time": ak.Array([[[-14.0]]]),
    }
    raw_data = ak.Array([raw_data_dict])[0]
    result = convert_to_training_data(raw_data, DataType.QCD, "test_ds", rotation=False)

    assert len(result) == 1
    # Only MSeg 0 and MSeg 1 must be selected
    assert len(result.msegs[0]) == 2
    matched_t0 = list(result.msegs[0].t0)
    assert abs(matched_t0[0] - 0.0) < 0.001
    assert abs(matched_t0[1] - 1.0) < 0.001

    # Verify mseg fields are preserved
    assert "etaPos" in ak.fields(result.msegs[0])
    assert "phiPos" in ak.fields(result.msegs[0])
    assert "etaDir" in ak.fields(result.msegs[0])
    assert "phiDir" in ak.fields(result.msegs[0])
    assert "t0" in ak.fields(result.msegs[0])
    assert "chiSquared" in ak.fields(result.msegs[0])
    assert abs(float(result.msegs[0].phiPos[0]) - 1.10) < 0.01
    assert abs(float(result.msegs[0].phiPos[1]) - 0.90) < 0.01


def test_mseg_near_jet_selection_multi_jet():
    """Test mseg matching with multiple jets, verifying no crosstalk between jets."""
    # Jet 0 at phi=1.0, Jet 1 at phi=-2.0
    # MSeg 0 at phi=1.05 (near Jet 0; delta_phi to Jet 1 is -3.05, abs=3.05 > 0.2)
    # MSeg 1 at phi=-1.95 (near Jet 1; delta_phi to Jet 0 is 2.95, abs=2.95 > 0.2)
    # MSeg 2 at phi=0.00 (far from both; delta_phi to Jet 1 is -2.0, abs=2.0 > 0.2)
    phis = [1.05, -1.95, 0.00]
    r = 100.0

    raw_data_dict = {
        "runNumber": ak.Array([123456]),
        "eventNumber": ak.Array([789012]),
        "mcEventWeight": ak.Array([1.0]),
        "jet_pt": ak.Array([[50.0, 60.0]]),
        "jet_eta": ak.Array([[0.5, -1.0]]),
        "jet_phi": ak.Array([[1.0, -2.0]]),
        "track_pT": ak.Array([[]]),
        "track_eta": ak.Array([[]]),
        "track_phi": ak.Array([[]]),
        "track_vertex_nParticles": ak.Array([[]]),
        "track_d0": ak.Array([[]]),
        "track_z0": ak.Array([[]]),
        "track_chiSquared": ak.Array([[]]),
        "track_PixelShared": ak.Array([[]]),
        "track_SCTShared": ak.Array([[]]),
        "track_PixelHoles": ak.Array([[]]),
        "track_SCTHoles": ak.Array([[]]),
        "track_PixelHits": ak.Array([[]]),
        "track_SCTHits": ak.Array([[]]),
        "MSeg_x": ak.Array([[r * np.cos(p) for p in phis]]),
        "MSeg_y": ak.Array([[r * np.sin(p) for p in phis]]),
        "MSeg_z": ak.Array([[0.0, 0.0, 0.0]]),
        "MSeg_px": ak.Array([[10.0, 10.0, 10.0]]),
        "MSeg_py": ak.Array([[5.0, 5.0, 5.0]]),
        "MSeg_pz": ak.Array([[30.0, 30.0, 30.0]]),
        "MSeg_t0": ak.Array([[10.0, 20.0, 30.0]]),
        "MSeg_chiSquared": ak.Array([[1.0, 1.1, 1.2]]),
        "clus_eta": ak.Array([[[0.5], [-1.0]]]),
        "clus_phi": ak.Array([[[1.0], [-2.0]]]),
        "clus_pt": ak.Array([[[5.0], [6.0]]]),
        "clus_l1hcal": ak.Array([[[100.0], [100.0]]]),
        "clus_l2hcal": ak.Array([[[200.0], [200.0]]]),
        "clus_l3hcal": ak.Array([[[300.0], [300.0]]]),
        "clus_l4hcal": ak.Array([[[400.0], [400.0]]]),
        "clus_l1ecal": ak.Array([[[500.0], [500.0]]]),
        "clus_l2ecal": ak.Array([[[600.0], [600.0]]]),
        "clus_l3ecal": ak.Array([[[700.0], [700.0]]]),
        "clus_l4ecal": ak.Array([[[800.0], [800.0]]]),
        "clus_time": ak.Array([[[-14.0], [-14.0]]]),
    }
    raw_data = ak.Array([raw_data_dict])[0]
    result = convert_to_training_data(raw_data, DataType.QCD, "test_ds", rotation=False)

    assert len(result) == 2
    # Jet 0 should have only MSeg 0 (t0=10.0)
    assert len(result.msegs[0]) == 1
    assert abs(float(result.msegs[0].t0[0]) - 10.0) < 0.001

    # Jet 1 should have only MSeg 1 (t0=20.0)
    assert len(result.msegs[1]) == 1
    assert abs(float(result.msegs[1].t0[0]) - 20.0) < 0.001


def test_mseg_near_jet_selection_phi_wraparound():
    """Test that mseg delta_phi calculation properly respects phi wraparound across [-pi, pi]."""
    # Jet at phi = 3.1
    # MSeg 0 at phi = -3.1 -> abs(delta_phi) across branch cut is ~0.083 < 0.2 (matched)
    # MSeg 1 at phi = 2.5 -> abs(delta_phi) is 0.600 > 0.2 (excluded)
    phis = [-3.1, 2.5]
    r = 100.0

    raw_data_dict = {
        "runNumber": ak.Array([123456]),
        "eventNumber": ak.Array([789012]),
        "mcEventWeight": ak.Array([1.0]),
        "jet_pt": ak.Array([[50.0]]),
        "jet_eta": ak.Array([[0.0]]),
        "jet_phi": ak.Array([[3.1]]),
        "track_pT": ak.Array([[]]),
        "track_eta": ak.Array([[]]),
        "track_phi": ak.Array([[]]),
        "track_vertex_nParticles": ak.Array([[]]),
        "track_d0": ak.Array([[]]),
        "track_z0": ak.Array([[]]),
        "track_chiSquared": ak.Array([[]]),
        "track_PixelShared": ak.Array([[]]),
        "track_SCTShared": ak.Array([[]]),
        "track_PixelHoles": ak.Array([[]]),
        "track_SCTHoles": ak.Array([[]]),
        "track_PixelHits": ak.Array([[]]),
        "track_SCTHits": ak.Array([[]]),
        "MSeg_x": ak.Array([[r * np.cos(p) for p in phis]]),
        "MSeg_y": ak.Array([[r * np.sin(p) for p in phis]]),
        "MSeg_z": ak.Array([[0.0, 0.0]]),
        "MSeg_px": ak.Array([[10.0, 10.0]]),
        "MSeg_py": ak.Array([[5.0, 5.0]]),
        "MSeg_pz": ak.Array([[30.0, 30.0]]),
        "MSeg_t0": ak.Array([[0.0, 1.0]]),
        "MSeg_chiSquared": ak.Array([[1.0, 1.1]]),
        "clus_eta": ak.Array([[[0.0]]]),
        "clus_phi": ak.Array([[[3.1]]]),
        "clus_pt": ak.Array([[[5.0]]]),
        "clus_l1hcal": ak.Array([[[100.0]]]),
        "clus_l2hcal": ak.Array([[[200.0]]]),
        "clus_l3hcal": ak.Array([[[300.0]]]),
        "clus_l4hcal": ak.Array([[[400.0]]]),
        "clus_l1ecal": ak.Array([[[500.0]]]),
        "clus_l2ecal": ak.Array([[[600.0]]]),
        "clus_l3ecal": ak.Array([[[700.0]]]),
        "clus_l4ecal": ak.Array([[[800.0]]]),
        "clus_time": ak.Array([[[-14.0]]]),
    }
    raw_data = ak.Array([raw_data_dict])[0]
    result = convert_to_training_data(raw_data, DataType.QCD, "test_ds", rotation=False)

    assert len(result) == 1
    assert len(result.msegs[0]) == 1
    assert abs(float(result.msegs[0].t0[0]) - 0.0) < 0.001


def test_track_and_mseg_empty_in_event():
    """Test that empty track and mseg containers in an event work correctly with and without rotation."""
    raw_data_dict = {
        "runNumber": ak.Array([123456]),
        "eventNumber": ak.Array([789012]),
        "mcEventWeight": ak.Array([1.0]),
        "jet_pt": ak.Array([[50.0]]),
        "jet_eta": ak.Array([[0.5]]),
        "jet_phi": ak.Array([[1.0]]),
        "track_pT": ak.Array([[]]),
        "track_eta": ak.Array([[]]),
        "track_phi": ak.Array([[]]),
        "track_vertex_nParticles": ak.Array([[]]),
        "track_d0": ak.Array([[]]),
        "track_z0": ak.Array([[]]),
        "track_chiSquared": ak.Array([[]]),
        "track_PixelShared": ak.Array([[]]),
        "track_SCTShared": ak.Array([[]]),
        "track_PixelHoles": ak.Array([[]]),
        "track_SCTHoles": ak.Array([[]]),
        "track_PixelHits": ak.Array([[]]),
        "track_SCTHits": ak.Array([[]]),
        "MSeg_x": ak.Array([[]]),
        "MSeg_y": ak.Array([[]]),
        "MSeg_z": ak.Array([[]]),
        "MSeg_px": ak.Array([[]]),
        "MSeg_py": ak.Array([[]]),
        "MSeg_pz": ak.Array([[]]),
        "MSeg_t0": ak.Array([[]]),
        "MSeg_chiSquared": ak.Array([[]]),
        "clus_eta": ak.Array([[[0.5]]]),
        "clus_phi": ak.Array([[[1.0]]]),
        "clus_pt": ak.Array([[[5.0]]]),
        "clus_l1hcal": ak.Array([[[100.0]]]),
        "clus_l2hcal": ak.Array([[[200.0]]]),
        "clus_l3hcal": ak.Array([[[300.0]]]),
        "clus_l4hcal": ak.Array([[[400.0]]]),
        "clus_l1ecal": ak.Array([[[500.0]]]),
        "clus_l2ecal": ak.Array([[[600.0]]]),
        "clus_l3ecal": ak.Array([[[700.0]]]),
        "clus_l4ecal": ak.Array([[[800.0]]]),
        "clus_time": ak.Array([[[-14.0]]]),
    }
    raw_data = ak.Array([raw_data_dict])[0]

    # Without rotation
    result_no_rot = convert_to_training_data(raw_data, DataType.QCD, "test_ds", rotation=False)
    assert len(result_no_rot) == 1
    assert len(result_no_rot.tracks[0]) == 0
    assert len(result_no_rot.msegs[0]) == 0

    # With rotation
    result_rot = convert_to_training_data(raw_data, DataType.QCD, "test_ds", rotation=True)
    assert len(result_rot) == 1
    assert len(result_rot.tracks[0]) == 0
    assert len(result_rot.msegs[0]) == 0

