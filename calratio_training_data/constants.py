# The jet-track delta R for inclusion
JET_TRACK_DELTA_R = 0.2

# The delta phi for msegments to be considered for a jet
JET_MSEG_DELTA_PHI = 0.2

# The delta R between a LLP and a jet for the jet to be considered from the LLP
LLP_JET_DELTA_R = 0.4

# Info specifying what range LLPs are valid for training in.
# These are *detector* coordinates, not relative to the PV's location.

# Below this LLP's are central. Above they are in the endcap.
LLP_central_eta_cut = 1.4

# Calorimeter Lxy is 1200 mm to 4000 mm
LLP_Lxy_min = 1200
LLP_Lxy_max = 4000

# Calorimeter Lz is 3500 mm to 6000 mm (abs)
LLP_Lz_min = 3500
LLP_Lz_max = 6000

# Min/Max Jet pT
# Used for rescaling clus/track/jet pT
min_jet_pt = 40  # GeV
max_jet_pt = 500  # GeV
