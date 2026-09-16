# sx_training_fetch

Code to fetch training datasets from ATLAS derivations for the Run 3 per-jet CalRatio NN. While designed to run against LLP1, really, it will work on anything that has the data.

## Contributing

The main home for this repo is on [GitHub](https://github.com/gordonwatts/sx_training_fetch). Over there please feel free to:

* [Open an issue](https://github.com/gordonwatts/sx_training_fetch/issues)
* [Submit a PR](https://github.com/gordonwatts/sx_training_fetch/pulls)
* [Examine the roadmap](https://github.com/users/gordonwatts/projects/3)

Any other mirrors are for archival purposes only and their issues and MR's aren't frequently checked!

## Installation

What you'll need on your system:

1. To run against a local file:
  a. `docker` should be installed
1. To run against a web dataset or a RUCIO dataset
  a. `servicex.yaml` file to a ServiceX instance.

## Usage

### Fetching Data

This command fetches the data from a sample and formats it as regular training input.

```text
> calratio_training_data fetch --help
                                                                                                                              
 Usage: calratio_training_data fetch [OPTIONS] DATA_TYPE:{signal|qcd|data|bib|t                                               
                                     tbar|cr_ttbar|cr_dijet_mc|cr_dijet_data|cr                                               
                                     _data} DATASET                                                                           
                                                                                                                              
 Fetch training data for cal ratio.

╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    data_type      DATA_TYPE:{signal|qcd|data|bib|ttbar|cr_ttbar|cr_d  Type of data to fetch (signal, qcd, data, bib,     │
│                     ijet_mc|cr_dijet_data|cr_data}                      ttbar, cr_ttbar, cr_data, cr_dijet_mc,             │
│                                                                         cr_dijet_data)                                     │
│                                                                         [required]                                         │
│ *    dataset        TEXT                                                The data source [required]                         │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --verbose       -v                   INTEGER  Increase verbosity level (use -v for INFO, -vv for DEBUG) [default: 0]       │
│ --ignore-cache                                Ignore cache and fetch fresh data                                            │
│ --local                                       Run ServiceX locally (requires docker)                                       │
│ --output        -o                   TEXT     Output file path [default: training.parquet]                                 │
│ --rotation          --no-rotation             Applies/does not apply rotations on cluster, track, mseg eta and phi         │
│                                               variables. Rotations applied by default.                                     │
│                                               [default: rotation]                                                          │
│ --sx-backend                         TEXT     ServiceX backend Name. Default is to use what is in your `servicex.yaml`     │
│                                               file.                                                                        │
│ --n-files       -n                   INTEGER  Number of files to process in the dataset. Default is to process all files.  │
│ --help                                        Show this message and exit.                                                  │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Some notes:

* Output data will be written in files called `training_000.parquet` by default. The `000` is to keep files to the 2 GB size.
* Default running means you need to run nothing but the data type and the DID dataset.
* If no jets are written out, rerun with `-v` to see if there are any messages that give you a hint.
* The `training_xxx.parquet` files are not deleted at the start of a run. Take care not to get confused by subsequent runs!

The dataset type:

* `signal` - Expect to find LLP's and only emits jets that are aligned with the LLPs
* `qcd` - Will extract all good jets
* `data` - Will extract all good jets from events that have fired a signal trigger
* `bib` - Will extract jets that match a BIB trigger, but not the tighter signal triggers.
* `ttbar` - Will extract all good hadronic jets (per-jet EMF < 0.97) from ttbar events
* `cr_ttbar` - Control region ttbar extraction (opposite-sign e-mu events, per-jet EMF > 0.97)
* `cr_data` - Control region data extraction
* `cr_dijet_mc` - Dijet control region MC extraction (see below)
* `cr_dijet_data` - Dijet control region data extraction (see below)

#### The dijet control region

`cr_dijet_mc` and `cr_dijet_data` select a QCD-dominated dijet control region. Events must fire
`HLT_j400_pf_ftf_preselj225_L1J100` and contain at least two good training jets, and then pass all of:

| Cut | Value |
|-----|-------|
| Leading jet pT | > 400 GeV |
| Subleading jet pT | > 60 GeV |
| \|Delta phi(lead, sublead)\| | > 3.0 rad |
| Dijet pT asymmetry | < 0.3 |
| H_T,Miss | < 120 GeV |

Only the 5 leading jets per surviving event are written out. The cut values live in `constants.py` as
the `CR_DIJET_*` constants.

For `cr_dijet_mc`, `mcEventWeight` is scaled by the sample cross-section looked up by DSID from the PMG
cross-section database. By default this is read from the central CVMFS copy; set `CALRATIO_PMG_XSEC_DB`
to point at a different file. Note that this scaling normalises each delivered chunk independently and
only over events passing the selection, so the weights are suitable for comparing shapes but not for
absolute normalisation.

As of this writing only `qcd` and `signal` are implemented.

### Where can the data be located?

In all cases we are expecting a LLP1-type derivation. The data can be in a number of locations:

1. **Local File** You can either specify the path, or use the standard `file` url: `file:///tmp/mydata.root`. If you just specify the path, the file must exist or the system might guess you are trying to do another file source. **NOTE** this only works in a branch of this code (removed functionality because it wasn't robust).
1. **URL** The file should be accessible by anyone anywhere (e.g. public). The dataset can be processed locally or remotely in this case (see the `--local` option).
    a. If the URL is a CERNBox URL, it can be converted to a `xrootd` address and accessed more efficiently that way - if you are running on a remote `servicex` instance. To correctly use a cernbox url, go to the file in CERNBOX, click on the details option from the drop down, and select the 'Direct Link' option.
1. **Rucio Dataset** You can specify just the dataset name, or prefix it with `rucio://`. The rucio DID scope must be present.

Note that this will use a remote ServiceX executable if it can - it will only use the local service if you are running on a local machine.
