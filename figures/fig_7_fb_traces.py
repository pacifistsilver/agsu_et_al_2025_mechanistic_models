"""Fig 7: monomer and heterodimer trajectories at the four points of fig 5.

``fig_5_distributions`` sweeps the (k_on,s, k_on,n) plane and measures how far
apart the two topologies' stationary count distributions are, then opens four
points of that plane -- A, B, C, D -- to show the distributions themselves.
This figure is the same four points seen in time: at each one, both promoters
are run with the *same* rates and their trajectories are drawn on the same
axes, so what separates two nearly identical count distributions is visible as
promoter dynamics rather than inferred from a distance.

The rate convention is fig 5's, so the points land where its markers are:
unbinding held at (k_off,s, k_off,n) = (0.05, 0.20), transcription at
k_y = 20 gamma, and only the binding rates move.  Both topologies take those
same four rates; nothing is matched or retuned between them, which is the point
-- ``fig_7`` asks what the topology alone does, not what a matched moment does.

    A  k_on = (0.01, 0.01)   d_TV = 0.006   deeply bursty, both ON ~20% of the time
    B  k_on = (0.10, 0.40)   d_TV = 0.105   the ridge: the two are furthest apart
    C  k_on = (1.00, 1.00)   d_TV = 0.046   mostly ON, bursts are long
    D  k_on = (100, 100)     d_TV = 0.001   saturated; binding is no longer rate limiting

Reading down the rows, the promoter goes from rare, well separated bursts to
permanent occupancy, and the two topologies' laws converge as it does: at D the
site is re-bound so fast after every release that whether the factors can be
co-bound stops mattering at all.

The topologies themselves:

monomer (M)
    one site the two factors compete for, S <- 0 -> N.  Occupancy is exclusive,
    so an ON period is a single sojourn and ends at beta_s or beta_n.

heterodimer (HD)
    two independent sites, transcribing whenever either is bound.  From 11 a
    site can release and rebind without the promoter ever going quiet, so ON
    periods run longer than either beta alone would give.

Trajectories are drawn by Gillespie on the same jump chains the inference layer
uses (:data:`stochtf.inference.models.PROMOTERS`), and the stationary laws come
from the same sFSP solve fig 5 and the likelihood use
(:mod:`stochtf.cme.stationary`), so neither is a second implementation of
something already in the package -- the d_TV printed here reproduces fig 5's.

Usage
-----
    python figures/fig_7_fb_traces.py
    python figures/fig_7_fb_traces.py --seed 7 --span 600
"""

import argparse

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numba import njit

from stochtf.cme import stationary as cme_stationary
from stochtf.inference.models import PROMOTERS, chain_generator
from stochtf.plotting import output_path, use_paper_style

#: Time is measured in mRNA lifetimes throughout, so gamma = 1.
GAMMA = 1.0

#: Transcription rate while ON, as in fig 4 and fig 5.  It caps the mean at
#: k_y/gamma = 20, which is what saturation at point D means.
K_Y = 20.0 * GAMMA

#: Unbinding rates held fixed across the plane, as in ``fig_5_distributions``.
BETA_S, BETA_N = 0.05, 0.20

#: (label, marker, k_on_s, k_on_n) -- fig 5's four sampled points, in its order.
POINTS = (("A", "o", 0.01, 0.01),
          ("B", "s", 0.10, 0.40),
          ("C", "^", 1.00, 1.00),
          ("D", "D", 100.0, 100.0))

#: Plotted window and the discarded head that removes the y = 0 start.
T_SPAN = 900.0
T_BURN = 100.0

#: Samples taken along the plotted window; also what the histograms count.
N_SAMPLES = 6000

PROMOTER_ORDER = ("monomer", "heterodimer")
LABELS = {"monomer": "M", "heterodimer": "HD"}
#: fig 5's colours, so a point reads the same in both figures.
COLOURS = {"monomer": "#c0562b", "heterodimer": "#1f5f8b"}


# ----------------------------------------------------------------------
# trajectories
# ----------------------------------------------------------------------

@njit(cache=True)
def _trajectory(starts, targets, rates, act, k_y, gamma, dt, n_grid, s0,
                seed, max_switch):
    """Gillespie on a flat jump chain, sampled onto a uniform time grid.

    The chain is the (starts, targets, rates, act) form the inference layer
    builds, with the mRNA birth-death appended: birth at k_y*act[s], death at
    gamma*y.  Only grid samples of y are kept -- storing every event would be
    ~40k points per trace and is not what the figure draws -- while promoter
    switches are recorded exactly, because the ON/OFF ribbons resolve gaps
    shorter than dt.
    """
    np.random.seed(seed)
    grid_y = np.empty(n_grid, np.int64)
    sw_t = np.empty(max_switch)
    sw_s = np.empty(max_switch, np.int64)

    s = s0
    y = 0
    t = 0.0
    sw_t[0] = 0.0
    sw_s[0] = s
    n_sw = 1
    g = 0

    while g < n_grid:
        lo, hi = starts[s], starts[s + 1]
        total = k_y * act[s] + gamma * y
        for j in range(lo, hi):
            total += rates[j]
        t_next = t - np.log(np.random.random()) / total

        # the state holds over [t, t_next), so every grid point in there sees it
        while g < n_grid and g * dt < t_next:
            grid_y[g] = y
            g += 1
        t = t_next
        if g >= n_grid:
            break

        u = np.random.random() * total
        acc = 0.0
        fired = -1
        for j in range(lo, hi):
            acc += rates[j]
            if u < acc:
                fired = j
                break
        if fired >= 0:
            s = targets[fired]
            if n_sw < max_switch:
                sw_t[n_sw] = t
                sw_s[n_sw] = s
                n_sw += 1
        elif u < acc + k_y * act[s]:
            y += 1
        else:
            y -= 1

    return grid_y, sw_t[:n_sw], sw_s[:n_sw]


def simulate(promoter, rates4, seed, span, burn, n_samples):
    """One trajectory: grid samples over the plotted window and its ON intervals.

    The promoter starts from its stationary occupancy and the mRNA pool from
    empty; ``burn`` lifetimes are then simulated and discarded, which is what
    removes the y = 0 transient rather than any equilibration assumption.
    """
    a_s, b_s, a_n, b_n = rates4
    starts, targets, rates, act, pi = PROMOTERS[promoter](a_s, b_s, a_n, b_n)
    rng = np.random.default_rng(seed)
    s0 = int(rng.choice(pi.size, p=pi))

    dt = span / n_samples
    n_grid = int(round((span + burn) / dt))
    grid_y, sw_t, sw_s = _trajectory(
        starts, targets, rates, np.asarray(act, np.float64), K_Y, GAMMA,
        dt, n_grid, s0, int(rng.integers(1, 2**31 - 1)), 1 << 20)

    t_grid = np.arange(n_samples) * dt
    return (t_grid, grid_y[n_grid - n_samples:],
            on_intervals(sw_t, sw_s, act, burn, burn + span))


def on_intervals(sw_t, sw_s, act, t0, t1):
    """(start, width) of every burst inside [t0, t1), measured from t0.

    Runs of consecutive transcribing states are merged: the heterodimer switches
    among 10/01/11 without ever going quiet, and only the return to the silent
    state ends a burst.  Splitting at every switch would draw one burst as
    several and inflate the count by a factor of ~20 at saturation.
    """
    on = np.asarray(act)[sw_s] > 0
    ends = np.append(sw_t[1:], max(t1, sw_t[-1]))

    edges = np.flatnonzero(np.diff(on.astype(np.int8)) != 0) + 1
    heads = np.r_[0, edges]
    tails = np.r_[edges, on.size] - 1
    burst = on[heads]
    start, stop = sw_t[heads[burst]], ends[tails[burst]]

    inside = (stop > t0) & (start < t1)
    start = np.clip(start[inside], t0, t1)
    stop = np.clip(stop[inside], t0, t1)
    return np.column_stack([start - t0, stop - start])


# ----------------------------------------------------------------------
# exact kinetics at one point
# ----------------------------------------------------------------------

def kinetics(promoter, rates4):
    """Stationary law and exact burst kinetics for one topology at one point.

    ``tau_on`` is the MFPT from the entry distribution of the transcribing set
    back to the silent state, solved from the generator itself -- both chains
    have a single silent state, so returns to it are regeneration points and
    f = 1/(tau_on + tau_off) and b = k_y tau_on are exact, not bursty-limit
    approximations.  The mean the sFSP solve reports is checked against
    k_y p_on/gamma, which needs no truncation.
    """
    Q, act = chain_generator(promoter, *rates4)
    live = np.flatnonzero(np.asarray(act) > 0)
    tau = np.linalg.solve(-Q[np.ix_(live, live)], np.ones(live.size))
    entry = Q[0, live] / Q[0, live].sum()

    tau_on = float(entry @ tau)
    tau_off = -1.0 / Q[0, 0]
    p_on = float(cme_stationary.promoter_stationary(Q)[live].sum())

    pmf = cme_stationary.stationary_pmf(Q, act, K_Y, GAMMA)
    mean, _, fano = cme_stationary.moments(Q, act, K_Y, GAMMA)
    assert abs(mean - K_Y * p_on / GAMMA) < 1e-8 * max(mean, 1.0)

    return {"pmf": pmf, "mean": mean, "fano": fano, "p_on": p_on,
            "tau_on": tau_on, "tau_off": tau_off,
            "f": 1.0 / (tau_on + tau_off), "b": K_Y * tau_on}


def total_variation(p, q):
    """d_TV between two stationary laws, zero-extended to a common grid.

    Each sFSP solve stops at whatever grid its own certificate accepts, so the
    two can differ by a few counts; zero-extending is what the certificate
    already bounds, and it is what fig 5 does.
    """
    n = max(p.size, q.size)
    a, b = np.zeros(n), np.zeros(n)
    a[:p.size], b[:q.size] = p, q
    return 0.5 * float(np.abs(a - b).sum())


# ----------------------------------------------------------------------
# the figure
# ----------------------------------------------------------------------

def draw(results, span):
    """One row per point: ON ribbons, both traces, both stationary laws."""
    fig = plt.figure(figsize=(9.9, 8.6))
    outer = fig.add_gridspec(len(results), 2, width_ratios=[3.5, 1.0],
                             hspace=0.40, wspace=0.11, left=0.06, right=0.985,
                             top=0.895, bottom=0.068)

    y_top = 1.06 * max(float(row[name]["counts"].max())
                       for row in results for name in PROMOTER_ORDER)

    for i, row in enumerate(results):
        left = outer[i, 0].subgridspec(2, 1, height_ratios=[0.17, 1.0],
                                       hspace=0.10)
        right = outer[i, 1].subgridspec(2, 1, height_ratios=[0.17, 1.0],
                                        hspace=0.10)
        ax_rib = fig.add_subplot(left[0])
        ax_tr = fig.add_subplot(left[1])
        ax_pdf = fig.add_subplot(right[1])

        for k, name in enumerate(PROMOTER_ORDER):
            res = row[name]
            ax_rib.broken_barh(res["on"], (1.05 - k, 0.9),
                               facecolor=COLOURS[name], alpha=0.45, linewidth=0)
            ax_tr.plot(res["t"], res["counts"], color=COLOURS[name], lw=0.7,
                       alpha=0.8, drawstyle="steps-post")
            ax_tr.axhline(res["mean"], ls="--", lw=0.9, color=COLOURS[name],
                          alpha=0.9, zorder=1)

            # the two laws can sit on top of each other, so the second is
            # dashed: a single visible curve would read as one being missing
            ax_pdf.plot(res["pmf"], np.arange(res["pmf"].size),
                        color=COLOURS[name], lw=1.6 if k == 0 else 1.1,
                        ls="-" if k == 0 else (0, (2.6, 1.8)), zorder=3 + k)
            hist = res["hist"]
            ax_pdf.fill_betweenx(np.arange(hist.size), 1e-6, hist, step="mid",
                                 color=COLOURS[name], alpha=0.20, linewidth=0)

        ax_rib.set_xlim(0, span)
        ax_rib.set_ylim(0, 2)
        ax_rib.set_xticks([])
        ax_rib.set_yticks([])
        for side in ax_rib.spines.values():
            side.set_visible(False)
        ax_rib.set_ylabel("ON", rotation=0, ha="right", va="center",
                          fontsize=6.5, color="0.35", labelpad=8)
        ax_rib.plot([0.0], [1.0], marker=row["marker"], ms=5.5, mfc="w",
                    mec="k", mew=0.8, transform=ax_rib.transAxes,
                    clip_on=False)

        ax_tr.set_xlim(0, span)
        ax_tr.set_ylim(0, y_top)
        ax_tr.set_ylabel("mRNA")
        ax_tr.text(0.006, 0.965, row["label"], transform=ax_tr.transAxes,
                   va="top", ha="left", fontsize=8,
                   bbox=dict(fc="w", ec="0.8", lw=0.5, alpha=0.85,
                             boxstyle="square,pad=0.28"))

        # log P, as in fig 5: at A almost all the mass sits on y = 0 and a
        # linear axis would hide the transcribing shoulder entirely
        ax_pdf.set_xscale("log")
        ax_pdf.set_xlim(1e-5, 1.0)
        ax_pdf.set_xticks([1e-4, 1e-2, 1e0])
        ax_pdf.set_ylim(0, y_top)
        ax_pdf.set_yticklabels([])
        ax_pdf.grid(True, alpha=0.25, lw=0.5)
        means = "   ".join(f"{LABELS[n]} {row[n]['mean']:.1f}"
                           for n in PROMOTER_ORDER)
        ax_pdf.text(0.96, 0.965, rf"$\langle y\rangle$   {means}",
                    transform=ax_pdf.transAxes, ha="right", va="top",
                    fontsize=6.5, color="0.3")

        if i == len(results) - 1:
            ax_tr.set_xlabel(r"time  (mRNA lifetimes, $1/\gamma$)")
            ax_pdf.set_xlabel(r"$P(y)$")

    handles = [Line2D([], [], color=COLOURS[n], lw=1.4, label=LABELS[n])
               for n in PROMOTER_ORDER]
    handles += [Line2D([], [], color="0.35", lw=1.3, label="exact CME"),
                Patch(fc="0.35", alpha=0.20, label=f"trace ({N_SAMPLES})"),
                Line2D([], [], color="0.35", lw=0.9, ls="--",
                       label=r"$\langle y\rangle$")]
    fig.legend(handles=handles, loc="upper left", ncol=5, fontsize=7.5,
               bbox_to_anchor=(0.058, 0.958), handlelength=1.6,
               columnspacing=1.6)
    fig.suptitle("Monomer and heterodimer trajectories at the four "
                 r"$k_{\rm on}$ points of the $d_{\rm TV}$ plane (fig 5), "
                 "bursty (A) to saturated (D)",
                 x=0.058, ha="left", y=0.988, fontsize=11)
    return fig


# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=3,
                        help="base seed; each trace gets its own offset")
    parser.add_argument("--span", type=float, default=T_SPAN,
                        help="plotted window, in mRNA lifetimes")
    parser.add_argument("--burn", type=float, default=T_BURN,
                        help="discarded head, in mRNA lifetimes")
    parser.add_argument("--samples", type=int, default=N_SAMPLES,
                        help="grid samples taken along the window")
    args = parser.parse_args()

    use_paper_style()
    results = []
    for i, (tag, marker, k_on_s, k_on_n) in enumerate(POINTS):
        rates4 = (k_on_s, BETA_S, k_on_n, BETA_N)
        row = {"tag": tag, "marker": marker, "rates": rates4}

        for k, name in enumerate(PROMOTER_ORDER):
            res = kinetics(name, rates4)
            t, counts, on = simulate(name, rates4, args.seed + 17 * i + k,
                                     args.span, args.burn, args.samples)
            hist = np.bincount(counts, minlength=res["pmf"].size).astype(float)
            res.update({"t": t, "counts": counts, "on": on,
                        "hist": hist / hist.sum()})
            row[name] = res

        row["tv"] = total_variation(row["heterodimer"]["pmf"],
                                    row["monomer"]["pmf"])
        row["label"] = (f"{tag}   $k_{{\\rm on}} = ({k_on_s:g},\\ {k_on_n:g})$"
                        f"   $d_{{\\rm TV}} = {row['tv']:.3f}$")
        results.append(row)

    fig = draw(results, args.span)
    for suffix in ("svg", "png"):
        fig.savefig(output_path(f"fig_7_fb_traces.{suffix}"),
                    bbox_inches="tight", facecolor="w")
    plt.close(fig)

    # ---- numbers worth quoting ---------------------------------------
    print(f"k_off = ({BETA_S}, {BETA_N}),  k_y = {K_Y:g} gamma,  "
          f"{args.span:g} lifetimes per trace")
    for row in results:
        k_on_s, _, k_on_n, _ = row["rates"]
        print(f"point {row['tag']}:  k_on = ({k_on_s:g}, {k_on_n:g})   "
              f"d_TV = {row['tv']:.3f}")
        for name in PROMOTER_ORDER:
            res = row[name]
            print(f"  {LABELS[name]:>2}  p_on = {res['p_on']:.3f}   "
                  f"f = {res['f']:.4f} gamma   b = {res['b']:.1f}   "
                  f"tau_on = {res['tau_on']:.2f}   Fano = {res['fano']:.3f}   "
                  f"<y> = {res['mean']:.2f} (trace {res['counts'].mean():.2f})"
                  f"   bursts drawn = {len(res['on'])}")
    print(f"  wrote {output_path('fig_7_fb_traces.png')}")


if __name__ == "__main__":
    main()
