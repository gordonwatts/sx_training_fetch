from pathlib import Path

import pytest
from calratio_training_data.sx_utils import (
    find_dataset,
    SXLocationOptions,
    build_sx_spec,
)
from servicex import dataset


def test_find_dataset_file(tmp_path: Path):
    "Make sure a local file comes back as a file"
    input_datafile = tmp_path / "myfile.root"
    input_datafile.touch()

    ds, location_opt = find_dataset(str(input_datafile))

    assert location_opt == SXLocationOptions.mustUseLocal
    assert isinstance(ds, dataset.FileList)
    assert len(ds.files) == 1
    assert ds.files[0] == str(input_datafile)


def test_find_dataset_file_relative(tmp_path: Path):
    "Use a direct relative path in the file name and make sure it works"
    input_datafile = tmp_path / "my_data.root"
    input_datafile.touch()
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(subdir)

        ds, location_opt = find_dataset("../my_data.root")
        assert location_opt == SXLocationOptions.mustUseLocal
        assert isinstance(ds, dataset.FileList)
        assert len(ds.files) == 1
        assert str(Path(ds.files[0]).resolve()) == str(input_datafile)


def test_find_dataset_file_relative_no_clues(tmp_path: Path):
    "Use a direct relative path in the file name and make sure it works"
    input_datafile = tmp_path / "my_data.root"
    input_datafile.touch()
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp_path)

        ds, location_opt = find_dataset("my_data.root")
        assert location_opt == SXLocationOptions.mustUseLocal
        assert isinstance(ds, dataset.FileList)
        assert len(ds.files) == 1
        assert ds.files[0] == str(input_datafile)


def test_find_dataset_file_with_scheme(tmp_path: Path):
    "Make sure a local file comes back as a file"
    input_datafile = tmp_path / "myfile.root"
    input_datafile.touch()

    ds, location_opt = find_dataset(input_datafile.as_uri())

    assert location_opt == SXLocationOptions.mustUseLocal
    assert isinstance(ds, dataset.FileList)
    assert len(ds.files) == 1
    assert ds.files[0] == str(input_datafile)


def test_find_dataset_local_non_file(tmp_path: Path):
    "Clearly a file path, and does not exist"

    with pytest.raises(ValueError) as e:
        find_dataset(str(tmp_path / "bogus_file.root"))

    assert "not exist" in str(e)


def test_find_dataset_random_http():
    "A url, but not pointing to anything special"

    http_location = "http://onedirve.myfiles.net/dm_evidence/events.root"
    ds, location_opt = find_dataset(http_location)

    assert location_opt == SXLocationOptions.anyLocation
    assert isinstance(ds, dataset.FileList)
    assert len(ds.files) == 1
    assert ds.files[0] == http_location


def test_find_dataset_rucio():
    "A rucio dataset id"

    rucio_did = (
        "mc16_13TeV:mc16_13TeV"
        ".311423.MGPy8EG_A14NNPDF23_NNPDF31ME_HSS_LLP_mH600_mS150_lthigh.deriv."
        "DAOD_EXOT15.e7357_e5984_s3234_r10201_r10210_p4696"
    )

    ds, location_opt = find_dataset(rucio_did)

    assert location_opt == SXLocationOptions.mustUseRemote
    assert isinstance(ds, dataset.Rucio)
    assert ds.scheme == "rucio"
    assert ds.did == f"rucio://{rucio_did}"


def test_find_dataset_rucio_with_spec():
    "A rucio dataset id with a scheme in front of it"

    rucio_did = (
        "mc16_13TeV:mc16_13TeV"
        ".311423.MGPy8EG_A14NNPDF23_NNPDF31ME_HSS_LLP_mH600_mS150_lthigh.deriv."
        "DAOD_EXOT15.e7357_e5984_s3234_r10201_r10210_p4696"
    )

    ds, location_opt = find_dataset("rucio://" + rucio_did)

    assert location_opt == SXLocationOptions.mustUseRemote
    assert isinstance(ds, dataset.Rucio)
    assert ds.scheme == "rucio"
    assert ds.did == f"rucio://{rucio_did}"


def test_find_dataset_cernbox():
    "A url, but pointing at cernbox"

    http_location = (
        "https://cernbox.cern.ch/files/spaces/eos/user/g/gwatts/public/data/mc16_13TeV"
        ".311423.MGPy8EG_A14NNPDF23_NNPDF31ME_HSS_LLP_mH600_mS150_lthigh.deriv."
        "DAOD_EXOT15.e7357_e5984_s3234_r10201_r10210_p4696/DAOD_EXOT15.26710706._000001"
        ".pool.root.1"
    )

    ds, location_opt = find_dataset(http_location)

    assert location_opt == SXLocationOptions.mustUseRemote
    assert isinstance(ds, dataset.FileList)
    assert len(ds.files) == 1
    assert (
        ds.files[0]
        == "root://eospublic.cern.ch/eos/user/g/gwatts/public/data/mc16_13TeV.311423."
        "MGPy8EG_A14NNPDF23_NNPDF31ME_HSS_LLP_mH600_mS150_lthigh.deriv.DAOD_EXOT15."
        "e7357_e5984_s3234_r10201_r10210_p4696/DAOD_EXOT15.26710706._000001.pool.root.1"
    )


def test_find_dataset_cernbox_quick_link():
    "A url, but pointing at cernbox"

    http_location = "https://cernbox.cern.ch/s/NMsiw60J6FCpY8M"

    ds, location_opt = find_dataset(http_location, prefer_local=False)

    assert location_opt == SXLocationOptions.anyLocation
    assert isinstance(ds, dataset.FileList)
    assert len(ds.files) == 1
    assert ds.files[0] == "https://cernbox.cern.ch/s/NMsiw60J6FCpY8M"


def test_find_dataset_cernbox_local():
    "A url, but not pointing to anything special"

    http_location = (
        "https://cernbox.cern.ch/files/spaces/eos/user/g/gwatts/public/data/mc16_13TeV"
        ".311423.MGPy8EG_A14NNPDF23_NNPDF31ME_HSS_LLP_mH600_mS150_lthigh.deriv."
        "DAOD_EXOT15.e7357_e5984_s3234_r10201_r10210_p4696/DAOD_EXOT15.26710706._000001"
        ".pool.root.1"
    )

    ds, location_opt = find_dataset(http_location, prefer_local=True)

    assert location_opt == SXLocationOptions.anyLocation
    assert isinstance(ds, dataset.FileList)
    assert len(ds.files) == 1
    assert (
        ds.files[0]
        == "https://cernbox.cern.ch/files/spaces/eos/user/g/gwatts/public/data/mc16_13TeV"
        ".311423.MGPy8EG_A14NNPDF23_NNPDF31ME_HSS_LLP_mH600_mS150_lthigh.deriv."
        "DAOD_EXOT15.e7357_e5984_s3234_r10201_r10210_p4696/DAOD_EXOT15.26710706._000001"
        ".pool.root.1"
    )


def test_build_sx_spec_local(mocker):
    "Make sure backend returns as local reference"
    mocker.patch(
        "calratio_training_data.sx_utils.find_dataset",
        return_value=(
            dataset.FileList(files=["dummy_file.root"]),
            SXLocationOptions.mustUseLocal,
        ),
    )
    spec, backend_name, adaptor = build_sx_spec("my_query", "a_ds")

    assert backend_name == "local-backend"


def test_build_sx_spec_remote_only(mocker):
    "Make sure backend returns as remote reference"
    mocker.patch(
        "calratio_training_data.sx_utils.find_dataset",
        return_value=(
            dataset.FileList(files=["dummy_file.root"]),
            SXLocationOptions.mustUseRemote,
        ),
    )
    spec, backend_name, adaptor = build_sx_spec("my_query", "a_ds")

    assert backend_name == "af.uchicago"


def test_build_sx_spec_either(mocker):
    "Make sure backend returns as either - so go remote"
    mocker.patch(
        "calratio_training_data.sx_utils.find_dataset",
        return_value=(
            dataset.FileList(files=["dummy_file.root"]),
            SXLocationOptions.anyLocation,
        ),
    )
    spec, backend_name, adaptor = build_sx_spec("my_query", "a_ds")

    assert backend_name == "af.uchicago"


def test_build_sx_spec_prefer_local(mocker):
    "Force local backend using prefer_local even when find_dataset returns remote option."
    mocker.patch(
        "calratio_training_data.sx_utils.find_dataset",
        return_value=(
            # Even if we mock a remote FileList, prefer_local should force local.
            dataset.FileList(files=["dummy_file.root"]),
            SXLocationOptions.anyLocation,
        ),
    )
    spec, backend_name, adaptor = build_sx_spec("my_query", "a_ds", prefer_local=True)

    assert backend_name == "local-backend"


def test_build_sx_spec_prefer_local_remote_forced(mocker):
    """Prefer local but dataset requires remote backend."""
    mocker.patch(
        "calratio_training_data.sx_utils.find_dataset",
        return_value=(
            dataset.FileList(files=["dummy_file.root"]),
            SXLocationOptions.mustUseRemote,
        ),
    )
    spec, backend_name, adaptor = build_sx_spec(
        "my_query", "a_ds", prefer_local=True
    )

    assert backend_name == "af.uchicago"
