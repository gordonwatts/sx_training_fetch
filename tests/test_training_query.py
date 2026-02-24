import awkward as ak

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

    # Call the function
    result = convert_to_training_data(raw_data, DataType.SIGNAL, rotation=False)

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

    # Call the function
    result = convert_to_training_data(raw_data, DataType.SIGNAL, rotation=False)

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

    # Call the function with BIB datatype
    result = convert_to_training_data(raw_data, DataType.BIB, rotation=False)

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

    # Call the function with BIB datatype
    result = convert_to_training_data(raw_data, DataType.BIB, rotation=False)

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

    # Call the function
    result = convert_to_training_data(raw_data, DataType.SIGNAL, rotation=False)

    # Basic checks - ensure the function runs without error and returns an array
    assert result is not None
    assert isinstance(result, ak.Array)

    # Check that we have some jets in the output
    assert len(result) == 0
