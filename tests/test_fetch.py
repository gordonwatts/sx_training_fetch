from typer.testing import CliRunner

from calratio_training_data.fetch import app


def test_fetch_command_default_download_false(monkeypatch):
    runner = CliRunner()
    captured = {}

    def _mock_fetch_training_data_to_file(dataset, run_config):
        captured["dataset"] = dataset
        captured["run_config"] = run_config

    monkeypatch.setattr(
        "calratio_training_data.training_query.fetch_training_data_to_file",
        _mock_fetch_training_data_to_file,
    )

    result = runner.invoke(app, ["fetch", "signal", "my_dataset"])

    assert result.exit_code == 0
    assert captured["dataset"] == "my_dataset"
    assert captured["run_config"].download is False


def test_fetch_command_download_flag_true(monkeypatch):
    runner = CliRunner()
    captured = {}

    def _mock_fetch_training_data_to_file(dataset, run_config):
        captured["dataset"] = dataset
        captured["run_config"] = run_config

    monkeypatch.setattr(
        "calratio_training_data.training_query.fetch_training_data_to_file",
        _mock_fetch_training_data_to_file,
    )

    result = runner.invoke(app, ["fetch", "signal", "my_dataset", "-d"])

    assert result.exit_code == 0
    assert captured["dataset"] == "my_dataset"
    assert captured["run_config"].download is True
