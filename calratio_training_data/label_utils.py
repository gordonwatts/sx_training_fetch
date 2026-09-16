import re


def extract_param_block(s: str) -> str | None:
    """
    Extracts parameter information about a given dataset
    Should output a string like "mH600_mS40_ct80" or similar
    """
    # Only consider part before ".deriv"
    prefix = s.split(".deriv")[0]

    # Pattern: lowercase-led parameters chained with underscores
    pattern = r"(?:[a-z][a-zA-Z]*[0-9]+(?:_[a-z][a-zA-Z]*[0-9]+)+)"

    matches = re.findall(pattern, prefix)

    return matches[-1] if matches else None
