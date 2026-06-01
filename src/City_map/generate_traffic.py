import random
import subprocess
import os
import shutil
import sumolib

# Try to read defaults from src.params when available
try:
    from src.params import (
        SUMO_HOME,
        TRAFFIC_HOTSPOT_DESTINATIONS,
        TRAFFIC_NUM_VEHICLES,
        TRAFFIC_MAX_TIME,
        TRAFFIC_HOTSPOT_RATIO,
        TRAFFIC_HOTSPOT_COUNT,
    )
except Exception:
    SUMO_HOME = None
    TRAFFIC_HOTSPOT_DESTINATIONS = ()
    TRAFFIC_NUM_VEHICLES = 5000
    TRAFFIC_MAX_TIME = 3600
    TRAFFIC_HOTSPOT_RATIO = 0.7
    TRAFFIC_HOTSPOT_COUNT = 2


def generate_traffic(
    net_file,
    out_trips,
    out_routes,
    num_vehicles=None,
    max_time=None,
    hotspot_ratio=None,
    hotspot_count=None,
):
    if num_vehicles is None:
        num_vehicles = TRAFFIC_NUM_VEHICLES
    if max_time is None:
        max_time = TRAFFIC_MAX_TIME
    if hotspot_ratio is None:
        hotspot_ratio = TRAFFIC_HOTSPOT_RATIO
    if hotspot_count is None:
        hotspot_count = TRAFFIC_HOTSPOT_COUNT

    net = sumolib.net.readNet(net_file)

    # In this map we want routes to start at entrance corners and end at exit corners.
    # For a directed network, entrances are edges whose fromNode is dead_end, while
    # exits are edges whose toNode is dead_end.
    start_edges = []
    end_edges = []
    for edge in net.getEdges():
        if edge.getFunction() == "internal":
            continue
        from_node = edge.getFromNode()
        to_node = edge.getToNode()
        if from_node is not None and from_node.getType() == "dead_end":
            start_edges.append(edge.getID())
        if to_node is not None and to_node.getType() == "dead_end":
            end_edges.append(edge.getID())

    # Fallback, if the network has no dead_end junctions or is circular.
    if not start_edges:
        start_edges = [
            e.getID() for e in net.getEdges() if e.getFunction() != "internal"
        ]
    if not end_edges:
        end_edges = [e.getID() for e in net.getEdges() if e.getFunction() != "internal"]

    origins = list(start_edges)
    destinations = list(end_edges)

    print(
        f"Znaleziono {len(origins)} punktów startowych i {len(destinations)} punktów docelowych."
    )

    # Prefer fixed hotspots so the same exits stay busier in every generated epoch.
    fixed_hotspots = [
        edge for edge in TRAFFIC_HOTSPOT_DESTINATIONS if edge in destinations
    ]
    if fixed_hotspots:
        bottlenecks = fixed_hotspots
    else:
        bottlenecks = random.sample(destinations, min(hotspot_count, len(destinations)))
    print(f"Wybrane hotspoty (korki się tu stworzą): {bottlenecks}")

    with open(out_trips, "w") as f:
        f.write("<routes>\n")
        f.write(
            '  <vType id="car" vClass="passenger" sigma="0.5" acceleration="2.6" deceleration="4.5" maxSpeed="15.0" />\n'
        )

        trips = []
        for i in range(num_vehicles):
            # Losowy krok w czasie (od 0 do max_time)
            depart_time = random.uniform(0, max_time)

            orig = random.choice(origins)

            # hotspot probability controlled by parameter
            if random.random() < hotspot_ratio:
                dest = random.choice(bottlenecks)
            else:
                dest = random.choice(destinations)

            # Zabezpieczenie przed trasą w to samo miejsce
            while orig == dest and len(destinations) > 1:
                dest = random.choice(destinations)

            trips.append({"id": i, "depart": depart_time, "orig": orig, "dest": dest})

        # Posortowane wygenerowane tripy po czasie odjazdu (depart) to WYMÓG silnika SUMO
        trips.sort(key=lambda x: x["depart"])

        for t in trips:
            f.write(
                f'  <trip id="veh_{t["id"]}" type="car" depart="{t["depart"]:.2f}" from="{t["orig"]}" to="{t["dest"]}" />\n'
            )

        f.write("</routes>\n")

    print(f"Wygenerowano pomyślnie plik tripów: {out_trips}")

    # Użycie duaroutera do wygenerowania finalnego pliku z trasami
    print("Uruchamianie duarouter do przeliczenia optymalnych tras...")
    duarouter_exe = shutil.which("duarouter")
    if duarouter_exe is None and SUMO_HOME:
        candidate = os.path.join(SUMO_HOME, "bin", "duarouter.exe")
        if os.path.exists(candidate):
            duarouter_exe = candidate
    if duarouter_exe is None:
        duarouter_exe = "duarouter"

    cmd = [
        duarouter_exe,
        "-n",
        net_file,
        "--route-files",
        out_trips,
        "-o",
        out_routes,
        "--ignore-errors",
        "--no-warnings",
    ]
    subprocess.run(cmd, check=True)
    print(f"Plik tras {out_routes} gotowy!")


if __name__ == "__main__":
    # Wywołujemy funkcję w folderze src/City_map/
    cwd = os.path.dirname(__file__) or "."
    net_path = os.path.join(cwd, "city_map.net.xml")
    trips_path = os.path.join(cwd, "trips.trips.xml")
    routes_path = os.path.join(cwd, "city_map.rou.xml")

    generate_traffic(net_path, trips_path, routes_path)
