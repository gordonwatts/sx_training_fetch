# Doing this because of a weird bug in Jupyter where crashes with something in traitlets and ip.

did_LLP1 = "mc23_13p6TeV:mc23_13p6TeV.802746.Py8EG_Zprime2EJs_Ld20_rho40_pi10_Zp2600_l1.deriv.DAOD_LLP1.e8531_s4159_r15530_p6463"
did_PHYSLITE = "mc23_13p6TeV:mc23_13p6TeV.802746.Py8EG_Zprime2EJs_Ld20_rho40_pi10_Zp2600_l1.deriv.DAOD_PHYSLITE.e8531_s4159_r15530_p6491"

import awkward as ak
import numpy as np
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex import Sample, ServiceXSpec, dataset, deliver
from servicex_analysis_utils import to_awk

from calratio_training_data import RunConfig, fetch_training_data

# Define the base query
base_query = FuncADLQueryPHYSLITE()

# Query to fetch muons and MET
query = base_query.Select(
    lambda e: {
        "jets": e.Jets().Where(lambda j: j.pt() >= 40 and abs(j.eta()) < 2.5),
        "event_info": e.EventInfo("EventInfo"),
    }
).Select(
    lambda e: {
        "jet_pt": e.jets.Select(lambda jet: jet.pt() / 1000.0),
        "jet_eta": e.jets.Select(lambda jet: jet.eta()),
        "jet_phi": e.jets.Select(lambda jet: jet.phi()),
        "run": e.event_info.runNumber(),
        "event": e.event_info.eventNumber(),
    }
)

# Fetch the data
data = to_awk(
    deliver(
        ServiceXSpec(
            Sample=[
                Sample(
                    Name="did_PHYSLITE",
                    Dataset=dataset.Rucio(did_PHYSLITE),
                    Query=query,
                )
            ]
        ),
    )
)["did_PHYSLITE"]

# Next, reformat it so it is per-jet, the way our training data is
data_PHYSLITE = ak.values_astype(
    ak.zip(
        {
            "pt": ak.flatten(data["jet_pt"]),
            "eta": ak.flatten(data["jet_eta"]),
            "phi": ak.flatten(data["jet_phi"]),
            "runNumber": ak.flatten(
                ak.broadcast_arrays(data["run"], data["jet_pt"])[0], axis=1
            ),
            "eventNumber": ak.flatten(
                ak.broadcast_arrays(data["event"], data["jet_pt"])[0], axis=1
            ),
        },
        with_name="Momentum3D",
    ),
    np.float32,
)

ak.to_parquet(
    data_PHYSLITE,
    "runit_PHYSLITE.parquet",
)

print("PHYSLITE data:")
data_PHYSLITE.type.show()

data_LLP1 = fetch_training_data(
    did_LLP1, RunConfig(run_locally=False, ignore_cache=False)
)
print("LLP1 data:")
data_LLP1.type.show()
ak.to_parquet(
    data_LLP1,
    "runit_LLP1.parquet",
)

import duckdb

con = duckdb.connect()

con.execute("CREATE TABLE t1 AS SELECT * FROM 'runit_PHYSLITE.parquet'")
con.execute("CREATE TABLE t2 AS SELECT * FROM 'runit_LLP1.parquet'")

query = """
SELECT 
    t1.runNumber, t1.eventNumber,
    t1.eta AS eta1, t2.eta AS eta2,
    t1.phi AS phi1, t2.phi AS phi2,
    t1.* EXCLUDE (runNumber, eventNumber, eta, phi),
    t2.* EXCLUDE (runNumber, eventNumber, eta, phi)
FROM t1
JOIN t2
  ON t1.runNumber = t2.runNumber
 AND t1.eventNumber = t2.eventNumber
 AND ABS(t1.eta - t2.eta) < 0.2
 AND LEAST(
       ABS(t1.phi - t2.phi),
       2*PI() - ABS(t1.phi - t2.phi)
     ) < 0.2
"""


con.execute(
    f"""
COPY (
  {query}
) TO 'joined.parquet' (FORMAT 'parquet');
"""
)
