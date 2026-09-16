"""Classify reconstructed jets as hard-scatter or pile-up.

Classification is based on dR matching to AntiKt4TruthJets (hard-scatter only collection),
performed in training_query.py.  The result is encoded in jet_truth_vtx_index:
  0  → matched within dR < TRUTH_JET_DR_CUT  → hard-scatter
  -1 → no match                               → pile-up

CLI usage:
    python -m calratio_training_data.jet_origin <parquet_file> [<parquet_file> ...]
"""

import argparse
import sys

import awkward as ak
import numpy as np


class JetOrigin:
    HARD_SCATTER = 1
    PILEUP = 0


def classify_jet_origin(truth_vtx_index: ak.Array) -> ak.Array:
    """Return per-jet origin labels (JetOrigin values as int32).

    Parameters
    ----------
    truth_vtx_index : per-jet array
        0  = matched to a hard-scatter truth jet (AntiKt4TruthJets, dR < 0.3)
        -1 = no match → pile-up
    """
    vtx = ak.values_astype(truth_vtx_index, np.int32)
    return ak.values_astype(
        ak.where(vtx == 0, np.int32(JetOrigin.HARD_SCATTER), np.int32(JetOrigin.PILEUP)),
        np.int32,
    )


def summary(origins: ak.Array) -> dict:
    """Return counts and pile-up indices for a flat array of classifications."""
    flat = ak.to_numpy(ak.flatten(origins) if origins.ndim > 1 else origins)
    pileup_indices = np.where(flat == JetOrigin.PILEUP)[0]
    return {
        "hard_scatter": int(np.sum(flat == JetOrigin.HARD_SCATTER)),
        "pileup": int(len(pileup_indices)),
        "total": len(flat),
        "pileup_indices": pileup_indices,
    }


def _print_summary(stats: dict, label: str = "", show_indices: bool = False) -> None:
    prefix = f"[{label}] " if label else ""
    total = stats["total"]
    if total == 0:
        print(f"{prefix}No jets found.")
        return
    print(f"{prefix}Total jets  : {total:>8,}")
    print(f"{prefix}Hard-scatter: {stats['hard_scatter']:>8,}  ({100 * stats['hard_scatter'] / total:.1f}%)")
    print(f"{prefix}Pile-up     : {stats['pileup']:>8,}  ({100 * stats['pileup'] / total:.1f}%)")
    if show_indices and stats["pileup"] > 0:
        print(f"{prefix}Pile-up indices: {stats['pileup_indices'].tolist()}")


def run_on_file(path: str) -> tuple[ak.Array, dict]:
    """Load a parquet file, classify jets, and return (data, stats)."""
    data = ak.from_parquet(path)

    if "jet_truth_vtx_index" not in data.fields:
        raise KeyError(
            f"{path}: field 'jet_truth_vtx_index' not found. "
            f"Available fields: {data.fields}\n"
            "Re-run the training query with the updated training_query.py to populate this field."
        )

    origins = classify_jet_origin(data["jet_truth_vtx_index"])
    return data, summary(origins)


def filter_pileup(data: ak.Array, stats: dict) -> ak.Array:
    """Return only hard-scatter jets (removes pile-up rows)."""
    hs_mask = np.ones(stats["total"], dtype=bool)
    hs_mask[stats["pileup_indices"]] = False
    return data[hs_mask]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Classify jets as hard-scatter or pile-up from a training parquet file."
    )
    parser.add_argument("files", nargs="+", metavar="FILE", help="Parquet file(s) to process")
    parser.add_argument(
        "--show-indices",
        action="store_true",
        help="Print the flat row indices of all pile-up jets",
    )
    parser.add_argument(
        "--filter-pileup",
        action="store_true",
        help="Keep only hard-scatter jets (requires --output)",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Write filtered jets to this parquet file (requires --filter-pileup)",
    )
    args = parser.parse_args(argv)

    if args.filter_pileup and not args.output:
        parser.error("--filter-pileup requires --output")
    if args.output and not args.filter_pileup:
        parser.error("--output requires --filter-pileup")

    combined = {"hard_scatter": 0, "pileup": 0, "total": 0, "pileup_indices": np.array([], dtype=np.intp)}
    offset = 0
    filtered_chunks = []

    for path in args.files:
        try:
            data, stats = run_on_file(path)
        except (FileNotFoundError, KeyError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

        _print_summary(stats, label=path, show_indices=args.show_indices)
        print()

        if args.filter_pileup:
            filtered_chunks.append(filter_pileup(data, stats))

        combined["hard_scatter"] += stats["hard_scatter"]
        combined["pileup"] += stats["pileup"]
        combined["total"] += stats["total"]
        combined["pileup_indices"] = np.concatenate(
            [combined["pileup_indices"], stats["pileup_indices"] + offset]
        )
        offset += stats["total"]

    if len(args.files) > 1:
        print("--- Combined ---")
        _print_summary(combined, show_indices=args.show_indices)

    if args.filter_pileup:
        out = ak.concatenate(filtered_chunks, axis=0)
        ak.to_parquet(out, args.output, compression="ZSTD", compression_level=-7)
        print(f"\nWrote {len(out):,} hard-scatter jets to {args.output}")


if __name__ == "__main__":
    main()
