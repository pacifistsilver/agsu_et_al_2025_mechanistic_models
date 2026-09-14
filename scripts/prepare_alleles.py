"""Combine the 129 and CAST allele tables into per-gene, per-allele counts.

Usage
-----
    python scripts/prepare_alleles.py 
    python scripts/prepare_alleles.py --gene Sox2 --gene Nanog
"""

import argparse
import gzip
import os
import re
import urllib.request

import numpy as np
import polars as pl

from stochtf.paths import DATA_DIR, PROCESSED_DATA_DIR, processed

#: The tables as distributed. data/raw is checked first so a fresh
#: scripts/download_data.py run is picked up, then the committed location.
SEARCH_DIRS = [os.path.join(DATA_DIR, "raw"), os.path.join(DATA_DIR, "ochiai")]
FILES = {"129": "GSE132589_ASEcount_G1_129.txt",
         "CAST": "GSE132589_ASEcount_G1_CAST.txt"}

#: Contemporary with the deposit; see the module docstring.
GTF_URL = ("https://ftp.ensembl.org/pub/release-96/gtf/mus_musculus/"
           "Mus_musculus.GRCm38.96.gtf.gz")
GTF_CACHE = os.path.join(DATA_DIR, "raw", "Mus_musculus.GRCm38.96.gtf.gz")

_TRANSCRIPT = re.compile(r'transcript_id "([^"]+)"')
_GENE_NAME = re.compile(r'gene_name "([^"]+)"')
_GENE_ID = re.compile(r'gene_id "([^"]+)"')


def locate(filename):
    for directory in SEARCH_DIRS:
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            return path
    raise SystemExit(
        f"{filename} not found in {' or '.join(SEARCH_DIRS)}.\n"
        "Fetch it with: python scripts/download_data.py")


def read_allele_table(path):
    """Reads one allele count table.

    Args:
        path: Path to the GEO allele-specific count table.

    Returns:
        The table as a dataframe, indexed by transcript.
    """
    """(transcript ids, cell names, counts) from one allele table.

    The header is read by hand because it is one field short of every data
    row; letting the CSV reader infer names from it misaligns every column.
    """
    with open(path, encoding="utf-8") as fh:
        cells = [name.strip('"') for name in fh.readline().rstrip("\n").split(" ")]

    frame = pl.read_csv(path, has_header=False, skip_rows=1, separator=" ",
                        new_columns=["TRANSCRIPT_ID"] + cells,
                        schema_overrides={"TRANSCRIPT_ID": pl.String})
    if frame.width != len(cells) + 1:
        raise SystemExit(f"{path}: expected {len(cells) + 1} columns, "
                         f"got {frame.width}")

    transcripts = frame["TRANSCRIPT_ID"].to_numpy().astype(str)
    counts = frame.drop("TRANSCRIPT_ID").to_numpy().astype(np.int32)
    return transcripts, np.array(cells), counts


def transcript_to_gene(cache=GTF_CACHE, url=GTF_URL):
    """Map every Ensembl transcript id to its gene symbol, from the GTF."""
    if not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, cache + ".part")
        os.replace(cache + ".part", cache)

    mapping = {}
    with gzip.open(cache, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            # transcript_id appears on every feature of a transcript; one hit
            # per transcript is enough, so skip lines already covered.
            found = _TRANSCRIPT.search(line)
            if not found or found.group(1) in mapping:
                continue
            name = _GENE_NAME.search(line) or _GENE_ID.search(line)
            if name:
                mapping[found.group(1)] = name.group(1)
    return mapping


def collapse_to_genes(transcripts, counts, mapping):
    """Sum transcript rows into one row per gene symbol."""
    symbols = np.array([mapping.get(t, "") for t in transcripts])
    keep = symbols != ""
    genes, index = np.unique(symbols[keep], return_inverse=True)

    totals = np.zeros((genes.size, counts.shape[1]), dtype=np.int64)
    np.add.at(totals, index, counts[keep])
    return genes, totals, int(keep.sum()), int((~keep).sum())


def main():
    """Parses arguments and builds the combined allele table."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gene", action="append", default=[],
                    help="also write data/processed/<gene>.npy, both alleles "
                         "concatenated; repeatable")
    ap.add_argument("--out", default=None,
                    help="output .npz (default data/processed/allele_counts.npz)")
    args = ap.parse_args()

    print("Mapping transcripts to genes")
    mapping = transcript_to_gene()
    print(f"  {len(mapping):,} transcripts in the annotation")

    per_allele, genes, cells = {}, None, None
    for allele, filename in FILES.items():
        path = locate(filename)
        transcripts, allele_cells, counts = read_allele_table(path)
        print(f"\n{allele}: {os.path.basename(path)}")
        print(f"  {counts.shape[0]:,} transcripts x {counts.shape[1]} cells")

        stripped = np.array([c.rsplit("_", 1)[0] for c in allele_cells])
        if cells is None:
            cells = stripped
        elif not np.array_equal(cells, stripped):
            raise SystemExit("the two tables list different cells; align them "
                             "before collapsing")

        symbols, totals, mapped, unmapped = collapse_to_genes(
            transcripts, counts, mapping)
        print(f"  {mapped:,} transcripts mapped, {unmapped:,} unmapped")
        print(f"  -> {symbols.size:,} genes")

        if genes is None:
            genes = symbols
        elif not np.array_equal(genes, symbols):
            raise SystemExit("gene sets differ between alleles")
        per_allele[allele] = totals

    counts_129, counts_cast = per_allele["129"], per_allele["CAST"]
    total = counts_129.sum() + counts_cast.sum()
    expressed = ((counts_129 + counts_cast).sum(axis=1) > 0).sum()
    print(f"\n{genes.size:,} genes x {cells.size} cells x 2 alleles")
    print(f"  {total:,} reads total, {expressed:,} genes with any signal")
    print(f"  129 share {counts_129.sum() / total:.3f}, "
          f"CAST share {counts_cast.sum() / total:.3f}")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    out = args.out or processed("allele_counts.npz")
    np.savez_compressed(out, counts_129=counts_129.astype(np.int32),
                        counts_cast=counts_cast.astype(np.int32),
                        genes=genes, cells=cells,
                        note=np.array("counts_<allele>[g, c] is gene genes[g] "
                                      "in cell cells[c]"))
    print(f"\nWrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")

    for symbol in args.gene:
        where = np.flatnonzero(genes == symbol)
        if where.size == 0:
            print(f"  {symbol}: not in the annotation, skipped")
            continue
        row = where[0]
        # Concatenated, not summed: two independent realisations per cell.
        vector = np.concatenate([counts_129[row], counts_cast[row]])
        path = processed(f"{symbol.lower()}.npy")
        np.save(path, vector)
        print(f"  {symbol}: n={vector.size} mean={vector.mean():.2f} "
              f"Fano={vector.var() / vector.mean():.2f} -> {path}")


if __name__ == "__main__":
    main()
