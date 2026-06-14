"""Robust evaluation of a trained model.

Runs the greedy policy over several seeds (and optionally several route files) on
held-out demand and reports each run plus mean ± spread of the KPIs, so the
comparison to the baselines is defensible (not a single lucky episode). With
--baselines it also runs fixed-time and max-pressure on the *same* seeds/routes,
so one command gives the full comparison.

Usage:
    python -m src.eval_best                               # latest best, held-out, eval seeds
    python -m src.eval_best --weights 1 --scenario hard --baselines
    python -m src.eval_best --scenario easy               # test the model on easy traffic
    python -m src.eval_best --routes city2_hard_eval.rou.xml other.rou.xml
"""

import argparse
import os

import src.params as P  # scenario-dependent values read via P.X (see apply_scenario)
from src.eval_common import (
    evaluate,
    fixed_action_fn,
    greedy_action_fn,
    load_agents,
    max_pressure_action_fn,
    network_dims,
    resolve_routes,
    summarize,
)
from src.params import (
    apply_scenario,
    build_eval_log_file,
    resolve_eval_weights_file,
    resolve_weights_for_run,
    scenario_is_hard,
)
from src.utils import make_log_fn, resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        default=None,
        metavar="RUN_ID|PATH",
        help="Numer treningu (np. 1) lub ścieżka do wag. Domyślnie: najnowszy *_best.pth.",
    )
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
        "--baselines",
        action="store_true",
        help="Oceń też fixed-time i max-pressure na tych samych trasach/seedach.",
    )
    parser.add_argument(
        "--show",
        type=int,
        choices=[0, 1],
        default=0,
        help="1 = wizualizacja w SUMO GUI; 0 (domyślnie) = headless.",
    )
    args = parser.parse_args()

    apply_scenario(scenario_is_hard(args.scenario))

    weights = args.weights
    if weights is None:
        weights = resolve_eval_weights_file()
    elif weights.isdigit():
        weights = resolve_weights_for_run(int(weights))

    seeds = args.seeds if args.seeds is not None else list(P.TRAIN_EVAL_SEED)
    routes = resolve_routes(args.routes)
    use_gui = bool(args.show)
    device = resolve_device()

    log_file_path = build_eval_log_file()
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log = make_log_fn(log_file)
        log(f"Ewaluacja best | wagi: {os.path.basename(weights)}")
        log(f"Seedy: {seeds}")
        log(f"Pliki tras: {[os.path.basename(r) for r in routes]}")

        ts_ids, obs_dims, act_dims = network_dims(routes[0])
        agents = load_agents(weights, ts_ids, obs_dims, act_dims, device)

        log("\n[model]")
        model_res = evaluate(
            greedy_action_fn(agents, ts_ids, obs_dims, device),
            routes, seeds, use_gui=use_gui, log=log,
        )
        summarize(model_res, log, "model")

        if args.baselines:
            log("\n[fixed-time]")
            summarize(
                evaluate(fixed_action_fn, routes, seeds, use_gui=use_gui, fixed_ts=True, log=log),
                log, "fixed-time",
            )
            log("\n[max-pressure]")
            summarize(
                evaluate(max_pressure_action_fn, routes, seeds, use_gui=use_gui, log=log),
                log, "max-pressure",
            )


if __name__ == "__main__":
    main()
