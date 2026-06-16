"""Baseline controllers for city_map_2: fixed-time and max-pressure.

Usage:
    python -m src.eval_baseline --mode fixed
    python -m src.eval_baseline --mode max-pressure --scenario hard

Uses the SAME evaluation harness as eval_best (same seeds x route files, same
mean ± spread report), so the baselines and the trained model are measured under
identical conditions and are directly comparable.
"""

import argparse
import os

import src.params as P  # scenario-dependent values read via P.X (see apply_scenario)
from src.eval_common import (
    evaluate,
    fixed_action_fn,
    max_pressure_action_fn,
    resolve_routes,
    summarize,
)
from src.params import apply_scenario, build_baseline_log_file, scenario_is_hard
from src.utils import make_log_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fixed", "max-pressure"], default="fixed")
    parser.add_argument(
        "--scenario",
        choices=["easy", "hard"],
        default=None,
        help="Trudność ruchu. Domyślnie bierze USE_HARD_TRAFFIC z params.py.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seedy SUMO. Domyślnie zestaw ewaluacyjny dla scenariusza.",
    )
    parser.add_argument(
        "--routes",
        nargs="+",
        default=None,
        help="Pliki tras (ścieżki lub nazwy w City_map_2). Domyślnie: held-out.",
    )
    parser.add_argument(
        "--show",
        type=int,
        choices=[0, 1],
        default=0,
        help="1 = wizualizacja w SUMO GUI (każdy przejazd osobno); 0 = headless.",
    )
    args = parser.parse_args()

    apply_scenario(scenario_is_hard(args.scenario))
    seeds = args.seeds if args.seeds is not None else list(P.TRAIN_EVAL_SEED)
    if args.show:
        seeds = seeds[:1]
    routes = resolve_routes(args.routes)
    action_fn = fixed_action_fn if args.mode == "fixed" else max_pressure_action_fn

    log_file_path = build_baseline_log_file(args.mode)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log = make_log_fn(log_file)
        log(f"Baseline: {args.mode}")
        log(f"Seedy: {seeds}")
        log(f"Pliki tras: {[os.path.basename(r) for r in routes]}")

        results = evaluate(
            action_fn,
            routes,
            seeds,
            use_gui=bool(args.show),
            fixed_ts=(args.mode == "fixed"),
            log=log,
        )
        summarize(results, log, args.mode)


if __name__ == "__main__":
    main()
