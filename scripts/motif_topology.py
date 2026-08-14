"""Genomic evidence on the promoter topology: two sites, or one contested site?

The inference in :mod:`stochtf.inference.models` fits two competing pictures of
the same promoter:

``heterodimer``  SOX2 and NANOG occupy *separate* sites and can be bound at once
``monomer``      they compete for *one* site, so only one can be bound

Stationary counts constrain that comparison only weakly (see
:mod:`stochtf.inference.identifiability`), so this script asks the genome
instead. If the two factors' recognition sequences overlap in the regulatory
DNA of these genes, they cannot be occupied simultaneously and the exclusive
picture is the right one; if the sites are distinct and separated, both can be
occupied and the two-site picture is.

What it does
------------
1. Look up each gene in Ensembl and collect the regulatory features (promoters,
   enhancers, open chromatin) annotated within +/- WINDOW of the gene body.
2. Download those sequences.
3. Scan them with JASPAR position weight matrices for SOX2 and NANOG, plus the
   POU5F1::SOX2 composite, on both strands.
4. For every SOX2 hit, find the nearest NANOG hit and record whether their
   footprints overlap.
5. Compare against two nulls: shuffled sequence (are the motifs there at all?)
   and randomly repositioned NANOG hits (is the arrangement non-random?).

Everything is fetched from public APIs at run time; nothing is vendored.

Usage
-----
    python scripts/motif_topology.py
    python scripts/motif_topology.py --window 50000 --threshold 0.85
"""

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from stochtf.paths import RESULTS_DIR

ENSEMBL = "https://rest.ensembl.org"
JASPAR = "https://jaspar.elixir.no/api/v1"

#: The four genes the counts were extracted for.
GENES = {"nanog": "Nanog", "sox2": "Sox2", "esrrb": "Esrrb", "rex1": "Zfp42"}

#: JASPAR CORE vertebrate matrices. The composite is included because in ES
#: cells SOX2 largely binds DNA together with OCT4 rather than alone.
MOTIFS = {
    "SOX2": "MA0143.5",
    "NANOG": "MA2339.1",
    "POU5F1::SOX2": "MA0142.1",
}

#: Regulatory feature types worth scanning. CTCF sites are structural and are
#: dropped.
KEEP_FEATURES = {"promoter", "enhancer", "open_chromatin_region",
                 "promoter_flanking_region", "TF_binding_site"}

#: Half-width of the search window around each gene body. Wide enough to reach
#: the Sox2 control region, which sits ~100 kb downstream of Sox2 itself.
WINDOW = 150_000

#: Fraction of the way from the worst to the best possible score at which a
#: match is called. 0.8 is the usual JASPAR-scanner default.
THRESHOLD = 0.80

PSEUDOCOUNT = 0.25
BACKGROUND = 0.25

COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


# ----------------------------------------------------------------------
# fetching
# ----------------------------------------------------------------------

#: Responses are cached on disk: Ensembl is the slow part by a wide margin
#: (minutes of latency against seconds of compute) and rate-limits under load,
#: so a rerun with different motif settings should not refetch the genome.
_CACHE = {"path": None, "data": {}}


def load_cache(path):
    _CACHE["path"] = path
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            _CACHE["data"] = json.load(fh)


def save_cache():
    if _CACHE["path"]:
        with open(_CACHE["path"], "w", encoding="utf-8") as fh:
            json.dump(_CACHE["data"], fh)


def fetch(url, accept="application/json", tries=5):
    """GET with backoff and an on-disk cache.

    Ensembl rate-limits and returns 503 under load, hence the retry.
    """
    if url in _CACHE["data"]:
        return _CACHE["data"][url]
    for attempt in range(tries):
        req = urllib.request.Request(
            url, headers={"Accept": accept, "User-Agent": "stochtf/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                body = response.read().decode()
            value = json.loads(body) if accept == "application/json" else body
            _CACHE["data"][url] = value
            return value
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503) and attempt < tries - 1:
                wait = float(exc.headers.get("Retry-After", 2 ** attempt))
                time.sleep(min(wait, 30))
                continue
            raise
        except urllib.error.URLError:
            if attempt < tries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError(f"gave up on {url}")


def gene_locus(symbol):
    d = fetch(f"{ENSEMBL}/lookup/symbol/mus_musculus/{symbol}?expand=0")
    return {"chrom": d["seq_region_name"], "start": d["start"],
            "end": d["end"], "strand": d["strand"],
            "assembly": d["assembly_name"]}


def regulatory_features(locus, window):
    lo = max(1, locus["start"] - window)
    hi = locus["end"] + window
    feats = fetch(f"{ENSEMBL}/overlap/region/mus_musculus/"
                  f"{locus['chrom']}:{lo}-{hi}?feature=regulatory")
    kept = []
    for f in feats:
        kind = f.get("description") or f.get("feature_type") or "unknown"
        if kind not in KEEP_FEATURES:
            continue
        kept.append({"kind": kind, "start": f["start"], "end": f["end"],
                     "id": f.get("id", "")})
    kept.sort(key=lambda f: f["start"])
    return kept


def region_sequence(chrom, start, end):
    fasta = fetch(f"{ENSEMBL}/sequence/region/mus_musculus/"
                  f"{chrom}:{start}..{end}?coord_system_version=GRCm39",
                  accept="text/x-fasta")
    return "".join(line.strip() for line in fasta.splitlines()
                   if not line.startswith(">")).upper()


def jaspar_pwm(matrix_id):
    """Log-odds matrix (4 x L, rows ACGT) and its name."""
    d = fetch(f"{JASPAR}/matrix/{matrix_id}/?format=json")
    pfm = d["pfm"]
    counts = np.array([[float(v) for v in pfm[b]] for b in "ACGT"])
    totals = counts.sum(axis=0)
    probs = (counts + PSEUDOCOUNT) / (totals + 4 * PSEUDOCOUNT)
    return np.log2(probs / BACKGROUND), d.get("name", matrix_id)


# ----------------------------------------------------------------------
# scanning
# ----------------------------------------------------------------------

def encode(seq):
    """Sequence -> integer codes, with -1 for anything that is not ACGT."""
    lookup = np.full(256, -1, dtype=np.int8)
    for code, base in enumerate("ACGT"):
        lookup[ord(base)] = code
    return lookup[np.frombuffer(seq.encode(), dtype=np.uint8)]


def scan(pwm, codes):
    """Score every window on the forward strand; NaN where the window has an N."""
    width = pwm.shape[1]
    n = codes.size - width + 1
    if n <= 0:
        return np.empty(0)
    windows = np.lib.stride_tricks.sliding_window_view(codes, width)
    valid = (windows >= 0).all(axis=1)
    scores = np.full(n, np.nan)
    safe = np.where(windows < 0, 0, windows)
    contribution = pwm[safe, np.arange(width)]
    scores[valid] = contribution.sum(axis=1)[valid]
    return scores


def score_threshold(pwm, p_value, scale=100.0):
    """Score at which a match is called, for a given per-position p-value.

    A relative-score cutoff ("80% of the best possible") is the usual quick
    choice but says nothing about how often the motif turns up by chance, and
    for short degenerate matrices like these it admits background at a rate
    that swamps any real site. So the exact null distribution of the score is
    built instead, by convolving the per-position distributions under the
    background model, and the threshold is read off its tail.
    """
    ints = np.rint(pwm * scale).astype(np.int64)
    offset = int(ints.min(axis=0).sum())
    shifted = ints - ints.min(axis=0)

    distribution = np.ones(1)
    for j in range(pwm.shape[1]):
        column = shifted[:, j]
        extended = np.zeros(distribution.size + int(column.max()))
        for base in range(4):
            start = int(column[base])
            extended[start:start + distribution.size] += distribution * BACKGROUND
        distribution = extended

    tail = np.cumsum(distribution[::-1])[::-1]
    reachable = np.flatnonzero(tail <= p_value)
    index = int(reachable[0]) if reachable.size else distribution.size - 1
    return (index + offset) / scale


def find_hits(pwm, seq, threshold):
    """Hits on both strands as (start, end, strand, score), 0-based.

    ``threshold`` is an absolute log-odds score from :func:`score_threshold`.
    """
    width = pwm.shape[1]
    hits = []
    for strand, s in (("+", seq), ("-", seq.translate(COMPLEMENT)[::-1])):
        scores = scan(pwm, encode(s))
        for i in np.flatnonzero(scores >= threshold):
            start = i if strand == "+" else len(seq) - i - width
            hits.append((int(start), int(start + width), strand,
                         float(scores[i])))
    hits.sort()
    return hits


# ----------------------------------------------------------------------
# the measurement
# ----------------------------------------------------------------------

def pair_up(sox_hits, nanog_hits):
    """For each SOX2 hit, its nearest NANOG hit: (gap, overlapping?).

    ``gap`` is the number of bases between the two footprints; it is negative
    when they overlap, and that overlap is what makes simultaneous occupancy
    impossible.
    """
    pairs = []
    for s_start, s_end, _, _ in sox_hits:
        best = None
        for n_start, n_end, _, _ in nanog_hits:
            gap = max(s_start, n_start) - min(s_end, n_end)
            if best is None or gap < best:
                best = gap
        if best is not None:
            pairs.append((best, best < 0))
    return pairs


def shuffled_hit_count(pwm, seq, threshold, replicates, rng):
    """How many hits does the same sequence give once its bases are shuffled?"""
    bases = list(seq)
    counts = []
    for _ in range(replicates):
        rng.shuffle(bases)
        counts.append(len(find_hits(pwm, "".join(bases), threshold)))
    return np.array(counts, dtype=float)


def repositioned_overlap(sox_hits, n_nanog, width, length, replicates, rng):
    """Overlap fraction when NANOG hits are dropped at random positions.

    Keeps the SOX2 hits and the NANOG hit *count* fixed and only destroys the
    arrangement, so it isolates whether the observed spacing means anything.
    """
    if not sox_hits or n_nanog == 0 or length <= width:
        return np.empty(0)
    out = []
    for _ in range(replicates):
        fake = []
        for _ in range(n_nanog):
            start = rng.randrange(0, length - width)
            fake.append((start, start + width, "+", 1.0))
        pairs = pair_up(sox_hits, fake)
        out.append(np.mean([o for _, o in pairs]) if pairs else 0.0)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", type=int, default=WINDOW,
                    help=f"bp either side of the gene body (default {WINDOW})")
    ap.add_argument("--pvalue", type=float, default=1e-4,
                    help="per-position p-value at which a match is called "
                         "(default 1e-4, the usual motif-scanning choice)")
    ap.add_argument("--replicates", type=int, default=200,
                    help="null replicates per region (default 200)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(RESULTS_DIR,
                                                     "motif_topology"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    load_cache(os.path.join(args.outdir, "ensembl_cache.json"))

    print(f"Fetching JASPAR matrices (p < {args.pvalue:g} per position)")
    pwms, thresholds = {}, {}
    for name, matrix_id in MOTIFS.items():
        pwm, label = jaspar_pwm(matrix_id)
        pwms[name] = pwm
        thresholds[name] = score_threshold(pwm, args.pvalue)
        # Information content says how specific the motif is -- NANOG's is
        # famously degenerate, which limits what any scan of it can show.
        probs = 2 ** pwm * BACKGROUND
        info = float((probs * np.log2(probs / BACKGROUND)).sum())
        print(f"  {name:14s} {matrix_id:10s} {label:14s} width {pwm.shape[1]:2d}"
              f"  info {info:5.1f} bits  score cutoff {thresholds[name]:5.2f}")

    hit_rows = []
    per_gene = {}
    for key, symbol in GENES.items():
        locus = gene_locus(symbol)
        feats = regulatory_features(locus, args.window)
        print(f"\n{key} ({symbol})  chr{locus['chrom']}:{locus['start']}-"
              f"{locus['end']}  {locus['assembly']}  "
              f"{len(feats)} regulatory features in +/-{args.window // 1000} kb")

        totals = {name: 0 for name in MOTIFS}
        gaps, overlaps, null_overlaps, observed_bp = [], [], [], 0
        enrich = {name: [] for name in MOTIFS}

        for feat in feats:
            seq = region_sequence(locus["chrom"], feat["start"], feat["end"])
            if len(seq) < 30:
                continue
            observed_bp += len(seq)
            hits = {name: find_hits(pwm, seq, thresholds[name])
                    for name, pwm in pwms.items()}
            for name, found in hits.items():
                totals[name] += len(found)
                for start, end, strand, score in found:
                    hit_rows.append((key, feat["kind"],
                                     feat["start"] + start, feat["start"] + end,
                                     strand, name, round(score, 4)))

            pairs = pair_up(hits["SOX2"], hits["NANOG"])
            gaps.extend(gap for gap, _ in pairs)
            overlaps.extend(over for _, over in pairs)

            null = repositioned_overlap(hits["SOX2"], len(hits["NANOG"]),
                                        pwms["NANOG"].shape[1], len(seq),
                                        args.replicates, rng)
            if null.size:
                null_overlaps.append(null)

            for name in ("SOX2", "NANOG", "POU5F1::SOX2"):
                counts = shuffled_hit_count(pwms[name], seq, thresholds[name],
                                            max(args.replicates // 10, 20), rng)
                enrich[name].append((len(hits[name]), counts.mean()))
            time.sleep(0.05)  # be polite to the Ensembl endpoint

        per_gene[key] = {
            "features": len(feats), "bp": observed_bp, "totals": totals,
            "gaps": gaps, "overlaps": overlaps,
            "null": np.concatenate(null_overlaps) if null_overlaps
            else np.empty(0),
            "enrich": {n: (sum(o for o, _ in v), sum(e for _, e in v))
                       for n, v in enrich.items()},
        }
        g = per_gene[key]
        print(f"  scanned {observed_bp:,} bp")
        for name in MOTIFS:
            obs, exp = g["enrich"][name]
            ratio = obs / exp if exp > 0 else float("inf")
            chance = 2 * observed_bp * args.pvalue
            print(f"    {name:14s} {obs:4d} hits   shuffled {exp:6.1f}   "
                  f"chance {chance:5.1f}   enrichment {ratio:5.2f}x")
        if overlaps:
            print(f"    SOX2 hits with an overlapping NANOG hit: "
                  f"{np.mean(overlaps) * 100:.1f}%  "
                  f"(random arrangement: {g['null'].mean() * 100:.1f}%)")
            print(f"    median gap to nearest NANOG hit: "
                  f"{np.median(gaps):.0f} bp")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("POOLED OVER ALL FOUR LOCI")
    print("=" * 70)
    all_gaps = np.array([g for d in per_gene.values() for g in d["gaps"]])
    all_over = np.array([o for d in per_gene.values() for o in d["overlaps"]])
    nulls = [d["null"] for d in per_gene.values() if d["null"].size]
    all_null = np.concatenate(nulls) if nulls else np.empty(0)

    total_bp = sum(d["bp"] for d in per_gene.values())
    print(f"  {total_bp:,} bp of regulatory sequence scanned\n")
    for name in MOTIFS:
        total = sum(d["totals"][name] for d in per_gene.values())
        shuffled = sum(d["enrich"][name][1] for d in per_gene.values())
        ratio = total / shuffled if shuffled > 0 else float("inf")
        print(f"  {name:14s} {total:5d} hits   shuffled {shuffled:6.1f}   "
              f"enrichment {ratio:5.2f}x")

    # SOX2 binds ES-cell enhancers largely as half of an OCT4:SOX2 element
    # rather than alone. If its standalone hits sit inside composite hits, the
    # partner sharing the element is OCT4, not NANOG.
    inside = 0
    sox_total = 0
    for key in per_gene:
        sox = [r for r in hit_rows if r[0] == key and r[5] == "SOX2"]
        comp = [r for r in hit_rows if r[0] == key and r[5] == "POU5F1::SOX2"]
        sox_total += len(sox)
        for _, _, s_start, s_end, _, _, _ in sox:
            if any(c_start <= s_start and s_end <= c_end
                   for _, _, c_start, c_end, _, _, _ in comp):
                inside += 1
    if sox_total:
        print(f"\n  SOX2 hits falling inside a POU5F1::SOX2 composite: "
              f"{inside}/{sox_total} ({inside / sox_total * 100:.0f}%)")

    print(f"\n  SOX2 hits with a NANOG hit in the same region: {all_gaps.size}")
    if all_gaps.size < 20:
        # Saying "x% overlap" off a handful of pairs would be noise dressed up
        # as a result, so the shortfall is reported instead.
        print("  -> too few co-occurring sites to test overlap against "
              "spacing.\n     This analysis cannot separate the exclusive and "
              "two-site\n     topologies at these loci; see the notes in the "
              "module docstring.")
    else:
        print(f"  overlapping (cannot be co-bound): "
              f"{all_over.mean() * 100:.1f}%")
        if all_null.size:
            print(f"  same under random arrangement:   "
                  f"{all_null.mean() * 100:.1f}%")
        for q in (10, 25, 50, 75, 90):
            print(f"    {q:2d}th percentile gap: "
                  f"{np.percentile(all_gaps, q):7.0f} bp")

    report = os.path.join(args.outdir, "motif_hits.tsv")
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("gene\tfeature\tstart\tend\tstrand\tmotif\tscore\n")
        for row in hit_rows:
            fh.write("\t".join(str(v) for v in row) + "\n")
    print(f"\nWrote {len(hit_rows)} motif hits to {report}")


if __name__ == "__main__":
    main()
