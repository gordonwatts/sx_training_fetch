import subprocess
import sys


def test_package_init_is_lazy() -> None:
    """Importing the package should not import the heavy training query module."""
    command = (
        "import calratio_training_data, sys; "
        "print('calratio_training_data.training_query' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_package_init_lazy_symbol_access() -> None:
    """Accessing a public symbol should trigger loading training_query lazily."""
    command = (
        "import calratio_training_data, sys; "
        "_ = calratio_training_data.RunConfig; "
        "print('calratio_training_data.training_query' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "True"
