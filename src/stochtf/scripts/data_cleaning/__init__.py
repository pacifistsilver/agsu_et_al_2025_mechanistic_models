""" download and combine allele data
Source is Ochiai et al., GEO accession GSE132589
Usage: 

    download   GEO .txt.gz          -> data/raw/*.txt
    prepare    join + gene symbols  -> data/processed/allele_counts.npz
    extract    pick a gene panel    -> data/processed/<panel>_counts.npz

    python -m stochtf.scripts.data_cleaning download
    python -m stochtf.scripts.data_cleaning prepare --gene Sox2 --gene Nanog
python -m stochtf.scripts.data_cleaning extract --list data/genelists/ochiai_panel.txt
"""

import gzip
import hashlib
import os
import re
import shutil
import sys
import urllib.request

import numpy as np

from stochtf.paths import DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, processed

GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE132nnn/GSE132589/suppl/"

# allele -> (filename once decompressed, expected SHA256). The checksums are
# None until someone records them: `shasum -a 256 data/raw/*.txt`, then paste
# them in and commit.
RAW_FILES = {
    "129": ("GSE132589_ASEcount_G1_129.txt", None),
    "CAST": ("GSE132589_ASEcount_G1_CAST.txt", None),
}

# data/raw first, so a fresh download wins over the copy committed to the repo.
SEARCH_DIRS = [os.path.join(DATA_DIR, "raw"), os.path.join(DATA_DIR, "ochiai")]

# Goes into every .npz we write, so whoever loads one knows the layout.
ALLELE_NOTE = ("counts_<allele>[g, c] is gene genes[g] in cell cells[c]; "
               "alleles are separate realisations, concatenate to fit")

# Contemporary with the deposit, so the transcript ids in the tables resolve.
GTF_URL = ("https://ftp.ensembl.org/pub/release-96/gtf/mus_musculus/"
           "Mus_musculus.GRCm38.96.gtf.gz")
GTF_CACHE = os.path.join(DATA_DIR, "raw", "Mus_musculus.GRCm38.96.gtf.gz")

_TRANSCRIPT = re.compile(r'transcript_id "([^"]+)"')
_GENE_NAME = re.compile(r'gene_name "([^"]+)"')
_GENE_ID = re.compile(r'gene_id "([^"]+)"')

# Retired or renamed symbols -> the release-96 name carrying the same Ensembl
# gene id. Each was resolved through the Ensembl xrefs endpoint and checked
# against the release-96 GTF; the gene id is here so you can re-check it.
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


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def raw_url(filename):
    """GEO download URL for one of the decompressed filenames in RAW_FILES."""
    return GEO_BASE + filename + ".gz"


def locate(filename):
    """Find a raw allele table, wherever it came from."""
    for directory in SEARCH_DIRS:
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            return path
    raise SystemExit(
        f"{filename} not found in {' or '.join(SEARCH_DIRS)}.\n"
        "Fetch it with: python -m stochtf.scripts.data_cleaning download")


def gene_stats(vector):
    """(mean, Fano, fraction of zeros) for one gene's counts."""
    values = np.asarray(vector, dtype=float)
    mean = values.mean()
    fano = values.var() / mean if mean > 0 else np.nan
    return mean, fano, float(np.mean(values == 0))


def write_gene_npy(name, counts_129, counts_cast, row):
    """Save one gene to processed/<gene>.npy, both alleles concatenated.

    Concatenated and not summed: the two alleles are independent realisations
    of the same promoter, so stacking them doubles the sample instead of
    averaging the noise away.
    """
    vector = np.concatenate([counts_129[row], counts_cast[row]])
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    path = processed(f"{name.lower()}.npy")
    np.save(path, vector)
    return path, vector


# ---------------------------------------------------------------------------
# step 1: download
# ---------------------------------------------------------------------------

def sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download_one(name, url, dest):
    tmp = dest + ".gz.part"
    print(f"Downloading {name} from {url}")
    urllib.request.urlretrieve(url, tmp)
    print(f"Decompressing {name}")
    with gzip.open(tmp, "rb") as src, open(dest, "wb") as out:
        shutil.copyfileobj(src, out)
    os.remove(tmp)


def step_download(args):
    """Fetch the two allele tables from GEO into data/raw/."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    failed = False

    for name, expected in RAW_FILES.values():
        dest = os.path.join(RAW_DATA_DIR, name)

        if not os.path.exists(dest):
            if args.check:
                print(f"MISSING  {name}")
                failed = True
                continue
            download_one(name, raw_url(name), dest)

        size_mb = os.path.getsize(dest) / (1 << 20)
        if expected is None:
            print(f"present  {name}  ({size_mb:.0f} MB, no checksum recorded)")
            continue

        digest = sha256(dest)
        if digest == expected:
            print(f"OK       {name}  ({size_mb:.0f} MB)")
        else:
            print(f"MISMATCH {name}\n  expected {expected}\n  got      {digest}")
            failed = True

    if failed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# step 2: prepare
# ---------------------------------------------------------------------------

def read_allele_table(path):
    """(transcript ids, cell names, counts) from one allele table.

    The header is read by hand because it is one field short of every data
    row; letting the CSV reader infer names from it misaligns every column.
    """
    import polars as pl

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


def step_prepare(args):
    """Join the two allele tables into one gene-by-cell store."""
    print("Mapping transcripts to genes")
    mapping = transcript_to_gene()
    print(f"  {len(mapping):,} transcripts in the annotation")

    per_allele, genes, cells = {}, None, None
    for allele, (filename, _) in RAW_FILES.items():
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
                        note=np.array(ALLELE_NOTE))
    print(f"\nWrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")

    for symbol in args.gene:
        where = np.flatnonzero(genes == symbol)
        if where.size == 0:
            print(f"  {symbol}: not in the annotation, skipped")
            continue
        path, vector = write_gene_npy(symbol, counts_129, counts_cast, where[0])
        mean, fano, _ = gene_stats(vector)
        print(f"  {symbol}: n={vector.size} mean={mean:.2f} "
              f"Fano={fano:.2f} -> {path}")


# ---------------------------------------------------------------------------
# step 3: extract
# ---------------------------------------------------------------------------

def resolve(symbol, present, lookup):
    """(name in the table, how it was matched) or (None, reason)."""
    if symbol in present:
        return symbol, "exact"
    if symbol in ALIASES and ALIASES[symbol] in present:
        return ALIASES[symbol], f"alias of {symbol}"
    if symbol.lower() in lookup:
        return lookup[symbol.lower()], "case-insensitive"
    return None, "not in the release-96 annotation"


def step_extract(args):
    """Pull a named set of genes out of the combined allele table."""
    wanted = list(args.gene)
    if args.list_file:
        with open(args.list_file, encoding="utf-8") as fh:
            wanted += [line.strip() for line in fh if line.strip()]
    if not wanted:
        raise SystemExit("nothing requested; pass --list or --gene")

    source = args.counts or processed("allele_counts.npz")
    if not os.path.exists(source):
        raise SystemExit(f"{source} not found. Run "
                         "python -m stochtf.scripts.data_cleaning prepare first.")
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
        both = np.concatenate([sub129[k], subcast[k]])
        mean, fano, zeros = gene_stats(both)
        total = both.sum()
        share = sub129[k].sum() / total if total else np.nan
        print(f"{name:>16} {mean:9.2f} {fano:8.2f} "
              f"{zeros * 100:8.1f} {share:10.3f}")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    stem = (os.path.splitext(os.path.basename(args.list_file))[0]
            if args.list_file else "selected")
    out = args.out or processed(f"{stem}_counts.npz")
    np.savez_compressed(out, counts_129=sub129, counts_cast=subcast,
                        genes=np.array(names), cells=cells,
                        requested=np.array(wanted),
                        note=np.array(ALLELE_NOTE))
    print(f"\nWrote {out} ({os.path.getsize(out) / 1e3:.0f} kB)")

    if args.npy:
        for k, name in enumerate(names):
            write_gene_npy(name, sub129, subcast, k)
        print(f"Wrote {len(names)} per-gene .npy files to {PROCESSED_DATA_DIR}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m stochtf.scripts.data_cleaning",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)

    p = sub.add_parser("download", help="fetch the raw tables from GEO")
    p.add_argument("--check", action="store_true",
                   help="only verify files already present")
    p.set_defaults(func=step_download)

    p = sub.add_parser("prepare", help="join the alleles into one gene table")
    p.add_argument("--gene", action="append", default=[],
                   help="also write data/processed/<gene>.npy, both alleles "
                        "concatenated; repeatable")
    p.add_argument("--out", default=None,
                   help="output .npz (default processed/allele_counts.npz)")
    p.set_defaults(func=step_prepare)

    p = sub.add_parser("extract", help="pull a gene panel out of that table")
    p.add_argument("--list", dest="list_file", default=None,
                   help="file of gene symbols, one per line")
    p.add_argument("--gene", action="append", default=[],
                   help="a symbol; repeatable, combined with --list")
    p.add_argument("--counts", default=None,
                   help="combined table (default processed/allele_counts.npz)")
    p.add_argument("--out", default=None, help="output .npz")
    p.add_argument("--npy", action="store_true",
                   help="also write processed/<gene>.npy per gene, both "
                        "alleles concatenated")
    p.set_defaults(func=step_extract)

    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
