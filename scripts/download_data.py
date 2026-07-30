"""Download the allele-resolved scRNA-seq counts from GEO.

These two files were previously committed to the repository (~200 MB together),
which is why the git history had to be rewritten. They are public, so they are
fetched on demand into data/raw/ instead.

Source: Ochiai et al., GEO accession GSE132589.

Usage
-----
    python scripts/download_data.py
    python scripts/download_data.py --check   # verify checksums only
"""

import argparse
import hashlib
import os
import sys
import urllib.request

from stochtf.paths import RAW_DATA_DIR

BASE = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE132nnn/GSE132589/suppl/")

#: Filename -> (URL, expected SHA256). Checksums are of the decompressed files
#: as they were used for the paper; fill these in from your local copies with
#: `shasum -a 256 data/raw/*.txt` and commit the result.
FILES = {
    "GSE132589_ASEcount_G1_129.txt": (
        BASE + "GSE132589_ASEcount_G1_129.txt.gz",
        None,
    ),
    "GSE132589_ASEcount_G1_CAST.txt": (
        BASE + "GSE132589_ASEcount_G1_CAST.txt.gz",
        None,
    ),
}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(name, url, dest):
    import gzip
    import shutil

    tmp = dest + ".gz.part"
    print(f"Downloading {name} from {url}")
    urllib.request.urlretrieve(url, tmp)
    print(f"Decompressing {name}")
    with gzip.open(tmp, "rb") as src, open(dest, "wb") as out:
        shutil.copyfileobj(src, out)
    os.remove(tmp)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="only verify files already present")
    args = ap.parse_args()

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    failed = False

    for name, (url, expected) in FILES.items():
        dest = os.path.join(RAW_DATA_DIR, name)

        if not os.path.exists(dest):
            if args.check:
                print(f"MISSING  {name}")
                failed = True
                continue
            download(name, url, dest)

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


if __name__ == "__main__":
    main()
