"""Generate random trips and routes for the city_map_2 network.

Origins are sampled from fringe entry edges, destinations from fringe exit
edges with a configurable bias toward hotspot destinations. Departures follow
a uniform spread or a mid-window Gaussian "rush" peak. The trips file is
converted to a routes file with duarouter.

Usage:
    python -m src.City_map_2.generate_traffic            # easy scenario
    python -m src.City_map_2.generate_traffic --hard     # harder scenario
    python -m src.City_map_2.generate_traffic --num-vehicles 1600 --profile peak
"""

import argparse
import os
import random
import subprocess
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import src.setup_sumo  # noqa: F401  (sets SUMO_HOME before sumolib lookups)
import sumolib

from src.params import (
    NET_FILE,
    ROUTE_FILE_EASY,
    ROUTE_FILE_HARD,
    TRAFFIC_HOTSPOT_DESTINATIONS,
    TRAFFIC_HOTSPOT_RATIO,
    TRAFFIC_HOTSPOT_RATIO_HARD,
    TRAFFIC_MAX_TIME,
    TRAFFIC_NUM_VEHICLES,
    TRAFFIC_NUM_VEHICLES_HARD,
    TRIPS_FILE_EASY,
    TRIPS_FILE_HARD,
    eval_route_file,
    traffic_pool_files,
)


def _sample_depart(rng: random.Random, max_time: float, profile: str) -> float:
    """Sample a departure time. 'peak' = Gaussian rush centred mid-window."""
    if profile == "peak":
        mu, sigma = max_time / 2.0, max_time / 5.0
        for _ in range(20):
            t = rng.gauss(mu, sigma)
            if 0.0 <= t <= max_time:
                return t
        return min(max(t, 0.0), max_time)
    return rng.uniform(0.0, max_time)


def generate_traffic(
    net_file: str,
    trips_file: str,
    route_file: str,
    num_vehicles: int,
    max_time: int,
    hotspot_ratio: float,
    profile: str = "uniform",
    seed: int | None = None,
) -> None:
    rng = random.Random(seed)
    net = sumolib.net.readNet(net_file)

    edges = [e for e in net.getEdges() if e.getFunction() == ""]
    edge_ids = {e.getID() for e in edges}
    sources = [e.getID() for e in edges if not e.getIncoming()]
    sinks = [e.getID() for e in edges if not e.getOutgoing()]
    if not sources or not sinks:
        raise RuntimeError(
            f"Network {net_file} has no fringe entry/exit edges to generate trips."
        )

    hotspots = [h for h in TRAFFIC_HOTSPOT_DESTINATIONS if h in edge_ids]
    missing = sorted(set(TRAFFIC_HOTSPOT_DESTINATIONS) - set(hotspots))
    if missing:
        print(f"UWAGA: hotspoty spoza sieci pominięte: {missing}")

    trips = []
    for i in range(num_vehicles):
        origin = rng.choice(sources)
        if hotspots and rng.random() < hotspot_ratio:
            dest = rng.choice(hotspots)
        else:
            dest = rng.choice(sinks)
        for _ in range(10):
            if dest != origin:
                break
            dest = rng.choice(sinks)
        depart = _sample_depart(rng, float(max_time), profile)
        trips.append((depart, f"veh{i}", origin, dest))
    trips.sort(key=lambda t: t[0])

    with open(trips_file, "w", encoding="utf-8") as f:
        f.write("<routes>\n")
        for depart, vid, origin, dest in trips:
            f.write(
                f'    <trip id="{vid}" depart="{depart:.2f}" from="{origin}" to="{dest}"/>\n'
            )
        f.write("</routes>\n")

    duarouter = sumolib.checkBinary("duarouter")
    subprocess.run(
        [
            duarouter,
            "-n",
            net_file,
            "--route-files",
            trips_file,
            "-o",
            route_file,
            "--ignore-errors",
            "--no-warnings",
            "--seed",
            str(seed if seed is not None else 42),
        ],
        check=True,
    )
    print(
        f"Zapisano {len(trips)} pojazdów (profil={profile}, hotspot_ratio={hotspot_ratio}): "
        f"{trips_file} -> {route_file}"
    )


def _generate_pool(hard, num_vehicles, max_time, hotspot_ratio, profile):
    """Generate the training pool + a held-out eval file (domain randomization).

    Each file uses a distinct seed → a different demand realisation. Trips go to
    a temp file and duarouter's .alt.xml is removed, so only the .rou.xml files
    land in City_map_2.
    """
    pool = traffic_pool_files(hard)
    eval_file = eval_route_file(hard)
    targets = [(route, 1000 + i) for i, route in enumerate(pool)] + [(eval_file, 9999)]
    print(
        f"Generuję pulę {len(pool)} plików treningowych + 1 held-out "
        f"(hard={hard}, pojazdów={num_vehicles}, profil={profile})..."
    )
    for route_file, seed in targets:
        base = os.path.basename(route_file)[: -len(".rou.xml")]
        trips_file = os.path.join(tempfile.gettempdir(), f"{base}.trips.xml")
        generate_traffic(
            NET_FILE, trips_file, route_file, num_vehicles, max_time,
            hotspot_ratio, profile, seed,
        )
        for junk in (trips_file, route_file[: -len(".rou.xml")] + ".rou.alt.xml"):
            if os.path.exists(junk):
                os.remove(junk)
    print(f"Pula gotowa w {os.path.dirname(eval_file)} (held-out: {os.path.basename(eval_file)}).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hard",
        action="store_true",
        help="Generuj trudniejszy scenariusz (więcej pojazdów, szczyt popytu, "
        "silniejsze hotspoty) do plików *_hard.",
    )
    parser.add_argument(
        "--pool",
        action="store_true",
        help="Generuj pulę treningową + held-out eval (domain randomization) "
        "zamiast pojedynczego pliku.",
    )
    parser.add_argument("--num-vehicles", type=int, default=None)
    parser.add_argument("--max-time", type=int, default=TRAFFIC_MAX_TIME)
    parser.add_argument("--hotspot-ratio", type=float, default=None)
    parser.add_argument("--profile", choices=["uniform", "peak"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--trips-file", default=None)
    parser.add_argument("--route-file", default=None)
    args = parser.parse_args()

    if args.hard:
        num_vehicles = args.num_vehicles or TRAFFIC_NUM_VEHICLES_HARD
        hotspot_ratio = (
            args.hotspot_ratio
            if args.hotspot_ratio is not None
            else TRAFFIC_HOTSPOT_RATIO_HARD
        )
        profile = args.profile or "peak"
        trips_file = args.trips_file or TRIPS_FILE_HARD
        route_file = args.route_file or ROUTE_FILE_HARD
    else:
        num_vehicles = args.num_vehicles or TRAFFIC_NUM_VEHICLES
        hotspot_ratio = (
            args.hotspot_ratio
            if args.hotspot_ratio is not None
            else TRAFFIC_HOTSPOT_RATIO
        )
        profile = args.profile or "uniform"
        trips_file = args.trips_file or TRIPS_FILE_EASY
        route_file = args.route_file or ROUTE_FILE_EASY

    if args.pool:
        _generate_pool(args.hard, num_vehicles, args.max_time, hotspot_ratio, profile)
        return

    generate_traffic(
        net_file=NET_FILE,
        trips_file=trips_file,
        route_file=route_file,
        num_vehicles=num_vehicles,
        max_time=args.max_time,
        hotspot_ratio=hotspot_ratio,
        profile=profile,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
