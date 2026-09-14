"""Pull a named set of genes out of the combined allele table.

``scripts/prepare_alleles.py`` writes every gene; this selects the ones you
actually want and reports what each looks like, so a panel can go straight into
the inference.

Usage
-----
    python scripts/extract_genes.py --list data/genelists/ochiai_panel.txt
    python scripts/extract_genes.py --gene Sox2 --gene Nanog --npy
"""

import argparse
import os

import numpy as np

from stochtf.paths import PROCESSED_DATA_DIR, processed

#: Retired or renamed symbols -> the release-96 name carrying the same Ensembl
#: gene id. Each was resolved via the Ensembl xrefs endpoint and checked against
#: the release-96 GTF; the gene id is given so the mapping can be re-checked.
ALIASES = {
    "6330407J23Rik": "Soga3",    # ENSMUSG00000038916
    "Ppap2a": "Plpp1",           # ENSMUSG00000021759
    "Myst4": "Kat6b",            # ENSMUSG00000021767
    "E130012A19Rik": "Epop",     # ENSMUSG00000043439
    "1110032A13Rik": "Rbfa",     # ENSMUSG00000024570
    "8430410A17Rik": "Hmces",    # ENSMUSG00000030060
    "Ctgf": "Ccn2",              # ENSMUSG00000019997
    # B3gnt1 is ambiguous: it resolves to both B4gat1 (ENSMUSG00000047379, the
    # accepted rename after the enzyme was reclassified) and B3gnt2
    # (ENSMUSG00000051650, which carried the alias historically). B4gat1 is
    # taken here; override if the source list meant the other.
    "B3gnt1": "B4gat1",
}


def resolve(symbol, present, lookup):
    """(name in the table, how it was matched) or (None, reason)."""
    if symbol in present:
        return symbol, "exact"
    if symbol in ALIASES and ALIASES[symbol] in present:
        return ALIASES[symbol], f"alias of {symbol}"
    if symbol.lower() in lookup:
        return lookup[symbol.lower()], "case-insensitive"
    return None, "not in the release-96 annotation"


def main():
    """Parses arguments and writes the per-gene count vectors."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", dest="list_file", default=None,
                    help="file of gene symbols, one per line")
    ap.add_argument("--gene", action="append", default=[],
                    help="a symbol; repeatable, combined with --list")
    ap.add_argument("--counts", default=None,
                    help="combined table (default processed/allele_counts.npz)")
    ap.add_argument("--out", default=None, help="output .npz")
    ap.add_argument("--npy", action="store_true",
                    help="also write processed/<gene>.npy per gene, both "
                         "alleles concatenated")
    args = ap.parse_args()

    wanted = list(args.gene)
    if args.list_file:
        with open(args.list_file, encoding="utf-8") as fh:
            wanted += [line.strip() for line in fh if line.strip()]
    if not wanted:
        raise SystemExit("nothing requested; pass --list or --gene")

    source = args.counts or processed("allele_counts.npz")
    if not os.path.exists(source):
        raise SystemExit(f"{source} not found. Run "
                         "scripts/prepare_alleles.py first.")
    store = np.load(source, allow_pickle=False)
    genes, cells = store["genes"], store["cells"]
    c129, ccast = store["counts_129"], store["counts_cast"]

    present = set(genes)
    lookup = {g.lower(): g for g in genes}
    index = {g: i for i, g in enumerate(genes)}

    rows, names, notes, missing = [], [], [], []
    for symbol in wanted:
        name, how = resolve(symbol, present, lookup)
        if name is None:
            missing.append((symbol, how))
            continue
        if name in names:                      # two aliases of the same gene
            notes.append(f"{symbol} duplicates {name}, kept once")
            continue
        rows.append(index[name])
        names.append(name)
        if how != "exact":
            notes.append(f"{symbol} -> {name} ({how})")

    rows = np.array(rows)
    sub129, subcast = c129[rows], ccast[rows]
    print(f"{len(wanted)} requested, {len(names)} extracted, "
          f"{len(missing)} unresolved")
    for note in notes:
        print(f"  note: {note}")
    for symbol, why in missing:
        print(f"  dropped: {symbol} ({why})")

    print(f"\n{'gene':>16} {'mean':>9} {'Fano':>8} {'zeros %':>8} "
          f"{'129 share':>10}")
    for k, name in enumerate(names):
        both = np.concatenate([sub129[k], subcast[k]]).astype(float)
        total = both.sum()
        fano = both.var() / both.mean() if both.mean() > 0 else np.nan
        share = sub129[k].sum() / total if total else np.nan
        print(f"{name:>16} {both.mean():9.2f} {fano:8.2f} "
              f"{np.mean(both == 0) * 100:8.1f} {share:10.3f}")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    stem = (os.path.splitext(os.path.basename(args.list_file))[0]
            if args.list_file else "selected")
    out = args.out or processed(f"{stem}_counts.npz")
    np.savez_compressed(out, counts_129=sub129, counts_cast=subcast,
                        genes=np.array(names), cells=cells,
                        requested=np.array(wanted),
                        note=np.array("counts_<allele>[g, c] is gene genes[g] "
                                      "in cell cells[c]; alleles are separate "
                                      "realisations, concatenate to fit"))
    print(f"\nWrote {out} ({os.path.getsize(out) / 1e3:.0f} kB)")

    if args.npy:
        for k, name in enumerate(names):
            vector = np.concatenate([sub129[k], subcast[k]])
            np.save(processed(f"{name.lower()}.npy"), vector)
        print(f"Wrote {len(names)} per-gene .npy files to {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()
