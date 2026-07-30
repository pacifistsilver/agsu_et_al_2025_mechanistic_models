"""Solve a reaction network by Finite State Projection.

Replaces simple_expand.py and support_expand.py, which were two copies of the
same burr08 driver differing only in the expander.

The promoter models delegate to the drivers already in the package
(``stochtf.cme.fsp_heterodimer`` / ``fsp_homodimer``), which set up the recorder
targets and projections each model needs.

The old ``monomer.py`` is not carried over: it built ``heterodimer_model``
despite its name, had its plotting commented out, and imported a module
(``fsp_flux_util``) that is not in the repository, so it raised ImportError.

Usage
-----
    python scripts/run_fsp.py burr08 --expander simple
    python scripts/run_fsp.py burr08 --expander support
    python scripts/run_fsp.py heterodimer
    python scripts/run_fsp.py homodimer
"""

import argparse

import numpy

import stochtf.cme.domain
import stochtf.cme.fsp.simple_expander
import stochtf.cme.fsp.solver
import stochtf.cme.fsp.support_expander
import stochtf.cme.recorder
import stochtf.cme.statistics
from stochtf.cme import fsp_example_util, fsp_heterodimer, fsp_homodimer
from stochtf.cme.models import burr08

#: SimpleExpander grows the entire domain along transitions to this depth.
SIMPLE_EXPANDER_DEPTH = 3
#: SupportExpander grows only around the support of the compressed solution.
SUPPORT_EXPANDER_DEPTH = 1
SUPPORT_EXPANDER_EPSILON = 1.0e-7

#: Error budget for the solution at the final time.
EPSILON = 1.0e-2


def run_burr08(expander_kind):
    """The burr08 model, which is initially stiff so it starts with finer steps."""
    model = burr08.create_model()
    initial_states = stochtf.cme.domain.from_iter((model.initial_state,))

    if expander_kind == "simple":
        expander = stochtf.cme.fsp.simple_expander.SimpleExpander(
            model.transitions, depth=SIMPLE_EXPANDER_DEPTH,
        )
    else:
        expander = stochtf.cme.fsp.support_expander.SupportExpander(
            model.transitions,
            depth=SUPPORT_EXPANDER_DEPTH,
            epsilon=SUPPORT_EXPANDER_EPSILON,
        )

    fsp_solver = stochtf.cme.fsp.solver.create(
        model,
        initial_states,
        expander,
        time_dependencies=burr08.create_time_dependencies(),
    )

    time_steps = numpy.concatenate((
        numpy.linspace(0.0, 1.0, 10),
        numpy.linspace(2.0, 16.0, 15),
    ))
    max_error_per_step = EPSILON / numpy.size(time_steps)

    recorder = stochtf.cme.recorder.create((model.species, model.species_counts))
    domains = []

    for i, t in enumerate(time_steps):
        print("STEP t = %g" % t)
        fsp_solver.step(t, max_error_per_step)
        if i % 3 == 0:
            print("recording solution and domain")
            p, _ = fsp_solver.y
            recorder.write(t, p)
            domains.append(numpy.array(fsp_solver.domain_states))
    print("OK")

    print("plotting solution and domain")
    fsp_example_util.plot_solution_and_domain(recorder[("A", "B")], domains)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=["burr08", "heterodimer", "homodimer"],
                    nargs="?", default="burr08")
    ap.add_argument("--expander", choices=["simple", "support"], default="support",
                    help="burr08 only; the promoter drivers fix their own expander")
    args = ap.parse_args()

    if args.model == "burr08":
        run_burr08(args.expander)
    elif args.model == "heterodimer":
        fsp_heterodimer.main()
    else:
        fsp_homodimer.main()


if __name__ == "__main__":
    main()
