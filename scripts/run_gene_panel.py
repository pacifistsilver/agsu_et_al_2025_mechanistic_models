"""Fit the joint monomer/heterodimer model to every gene in a panel.

For each gene the topology is a sampled parameter, so one run returns
P(heterodimer | counts) and a Bayes factor alongside the rate posteriors. See
``stochtf.inference.models.JointModel``.

Allele structure. A cell carries two copies and the assay reports them
separately, so the two are concatenated into 2 x n_cells independent
realisations of one promoter. Genes with no signal at all on one allele are the
exception: in a male F1 line the X-linked genes have a single copy, and
concatenating would append n_cells structural zeros that a promoter model reads
as an enormous OFF fraction. Those are fitted on the expressed allele alone.

Signal. Genes whose mean count is below ``--min-mean`` are skipped; a promoter
model fitted to counts that are almost all zero is not estimating switching
rates, it is estimating nothing.

Results are written after every gene, and an existing results file is read back
on startup so an interrupted run resumes where it stopped.

Usage
-----
    python scripts/run_gene_panel.py --panel data/processed/ochiai_panel_counts.npz
    python scripts/run_gene_panel.py --panel ... --draws 300 --chains 2
"""

import argparse
import json
import os
import time

import numpy as np

from stochtf import paths
from stochtf.inference.models import MODELS, model_probabilities

SUMMARISE = ("alpha_s", "alpha_n", "k_y")


def counts_for(row, c129, ccast):
    """(count vector, how the alleles were used) for one gene."""
    a, b = c129[row].astype(float), ccast[row].astype(float)
    if a.sum() == 0 and b.sum() > 0:
        return b, "CAST only (no 129 signal)"
    if b.sum() == 0 and a.sum() > 0:
        return a, "129 only (no CAST signal)"
    return np.concatenate([a, b]), "both alleles"


def load_done(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return {row["gene"]: row for row in json.load(fh)}


def save(path, results):
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(list(results.values()), fh, indent=1)
    os.replace(tmp, path)


def main():
    """Parses arguments and fits the joint model per gene."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=paths.processed("ochiai_panel_counts.npz"))
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--min-mean", type=float, default=1.0,
                    help="skip genes below this mean count (default 1.0)")
    ap.add_argument("--limit", type=int, default=None,
                    help="fit at most this many genes, for a trial run")
    ap.add_argument("--only", action="append", default=[],
                    help="fit just these genes; repeatable")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    store = np.load(args.panel, allow_pickle=False)
    genes = store["genes"]
    c129, ccast = store["counts_129"], store["counts_cast"]

    out = args.out or paths.results("panel_joint_fits.json")
    results = load_done(out)
    if results:
        print(f"resuming: {len(results)} genes already fitted in {out}")

    todo = [(i, g) for i, g in enumerate(genes) if g not in results]
    if args.only:
        wanted = set(args.only)
        todo = [(i, g) for i, g in todo if g in wanted]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(genes)} genes in the panel, {len(todo)} to fit "
          f"({args.draws} draws x {args.chains} chains each)\n")

    start = time.perf_counter()
    for k, (row, gene) in enumerate(todo, start=1):
        data, how = counts_for(row, c129, ccast)
        if data.mean() < args.min_mean:
            results[gene] = {"gene": str(gene), "status": "skipped",
                             "reason": f"mean {data.mean():.3f} below "
                                       f"{args.min_mean}",
                             "n": int(data.size),
                             "mean": float(data.mean()), "alleles": how}
            save(out, results)
            print(f"[{k}/{len(todo)}] {gene:>12}  skipped, mean "
                  f"{data.mean():.2f}")
            continue

        model = MODELS["joint"]()
        model.fit(data, sampler_config={"draws": args.draws,
                                        "chains": args.chains},
                  progressbar=False, sample_prior=False)
        probs = model_probabilities(model.idata)

        record = {"gene": str(gene), "status": "fitted", "alleles": how,
                  "n": int(data.size), "mean": float(data.mean()),
                  "fano": float(data.var() / data.mean()),
                  "p_heterodimer": probs["heterodimer"],
                  "bayes_factor": probs["bayes_factor_het_over_mono"]}
        post = model.idata["posterior"]
        for name in SUMMARISE:
            x = np.asarray(post[name]).ravel()
            record[name] = [float(np.percentile(x, 3)), float(np.median(x)),
                            float(np.percentile(x, 97))]
        results[gene] = record
        save(out, results)

        rate = (time.perf_counter() - start) / k
        print(f"[{k}/{len(todo)}] {gene:>12}  mean {data.mean():7.2f}  "
              f"Fano {record['fano']:7.2f}  P(het) {record['p_heterodimer']:.3f}"
              f"  BF {record['bayes_factor']:7.2f}   "
              f"~{rate * (len(todo) - k) / 60:.0f} min left")

    fitted = [r for r in results.values() if r["status"] == "fitted"]
    if fitted:
        bf = np.array([r["bayes_factor"] for r in fitted])
        print(f"\n{len(fitted)} genes fitted, "
              f"{len(results) - len(fitted)} skipped")
        print(f"Bayes factor: min {bf.min():.2f}  median {np.median(bf):.2f}  "
              f"max {bf.max():.2f}")
        print(f"  favouring heterodimer (BF>3): {(bf > 3).sum()}")
        print(f"  favouring monomer   (BF<1/3): {(bf < 1 / 3).sum()}")
        print(f"  undecided                   : "
              f"{((bf >= 1 / 3) & (bf <= 3)).sum()}")
    print(f"\nResults in {out}")


if __name__ == "__main__":
    main()
