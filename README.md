# sx_training_fetch

Code to fetch training datasets from ATLAS derivations for the Run 3 per-jet CalRatio NN. While designed to run against LLP1, really, it will work on anything that has the data.

## Installation

What you'll need on your system:

1. To run against a local file:
  a. `docker` should be installed
1. To run against a web dataset or a RUCIO dataset
  a. `servicex.yaml` file to a ServiceX instance.

## Usage

```text
> calratio_training_data --help

```

Some notes

### Where is the data?

The data can be in a number of locations

1. **Local File** You can either specify the path, or use the standard `file` url: `file:///tmp/mydata.root`. If you just specify the path, the file must exist or the system might guess you are trying to do another file source.
1. **URL** The file should be accessible by anyone anywhere (e.g. public). The dataset can be processed locally or remotely in this case (see the `--local` option).
    a. If the URL is a CERNBox URL, it can be converted to a `xrootd` address and accessed more efficiently that way - if you are running on a remote `servicex` instance. To correctly use a cernbox url, go to the file in CERNBOX, click on the details option from the drop down, and select the 'Direct Link' option.
1. **Rucio Dataset** You can specify just the dataset name, or prefix it with `rucio://`. The rucio DID scope must be present.

Note that this will use a remote ServiceX executable if it can - it will only use the local service if you are running on a local machine.