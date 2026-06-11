"""Generate random trips and routes for the city_map_2 network.

Origins are sampled from fringe entry edges, destinations from fringe exit
edges with a configurable bias toward hotspot destinations. The trips file is
converted to a routes file with duarouter.

Usage:
    python -m src.City_map_2.generate_traffic [--num-vehicles N] [--max-time S] [--seed S]
"""

import argparse
import os
import random
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import src.setup_sumo  # noqa: F401  (sets SUMO_HOME before sumolib lookups)
import sumolib

from src.params import (
    NET_FILE,
    ROUTE_FILE,
    TRAFFIC_HOTSPOT_DESTINATIONS,
    TRAFFIC_HOTSPOT_RATIO,
    TRAFFIC_MAX_TIME,
    TRAFFIC_NUM_VEHICLES,
    TRIPS_FILE,
)


def generate_traffic(
    net_file: str = NET_FILE,
    trips_file: str = TRIPS_FILE,
    route_file: str = ROUTE_FILE,
    num_vehicles: int = TRAFFIC_NUM_VEHICLES,
    max_time: int = TRAFFIC_MAX_TIME,
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
        if hotspots and rng.random() < TRAFFIC_HOTSPOT_RATIO:
            dest = rng.choice(hotspots)
        else:
            dest = rng.choice(sinks)
        for _ in range(10):
            if dest != origin:
                break
            dest = rng.choice(sinks)
        depart = rng.uniform(0.0, float(max_time))
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
    print(f"Zapisano {len(trips)} pojazdów: {trips_file} -> {route_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-vehicles", type=int, default=TRAFFIC_NUM_VEHICLES)
    parser.add_argument("--max-time", type=int, default=TRAFFIC_MAX_TIME)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--trips-file", default=TRIPS_FILE)
    parser.add_argument("--route-file", default=ROUTE_FILE)
    args = parser.parse_args()
    generate_traffic(
        trips_file=args.trips_file,
        route_file=args.route_file,
        num_vehicles=args.num_vehicles,
        max_time=args.max_time,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
