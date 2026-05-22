import random
import subprocess
import os
import sumolib

def generate_traffic(net_file, out_trips, out_routes, num_vehicles=5000, max_time=3600):
    net = sumolib.net.readNet(net_file)
    
    # Znajdź punkty początkowe (brak krawędzi wchodzących) 
    # i końcowe (brak krawędzi wychodzących).
    # Jeśli ich nie ma, użyj wszystkich normalnych krawędzi.
    origins = [e.getID() for e in net.getEdges() if e.getFunction() != 'internal' and not e.getIncoming()]
    destinations = [e.getID() for e in net.getEdges() if e.getFunction() != 'internal' and not e.getOutgoing()]
    
    # Fallback, jeśli mapa jest okrężna lub nie ma typowych krawędzi wejściowych/wyjściowych
    if not origins: 
        origins = [e.getID() for e in net.getEdges() if e.getFunction() != 'internal']
    if not destinations: 
        destinations = [e.getID() for e in net.getEdges() if e.getFunction() != 'internal']
        
    print(f"Znaleziono {len(origins)} punktów startowych i {len(destinations)} punktów docelowych.")
    
    # Wybierzmy 2 punkty docelowe, które będą naszymi "hotspotami" - tzn. więcej aut tam jedzie
    bottlenecks = random.sample(destinations, min(2, len(destinations)))
    print(f"Wybrane hotspoty (korki się tu stworzą): {bottlenecks}")
    
    with open(out_trips, 'w') as f:
        f.write('<routes>\n')
        f.write('  <vType id="car" vClass="passenger" sigma="0.5" acceleration="2.6" deceleration="4.5" maxSpeed="15.0" />\n')
        
        trips = []
        for i in range(num_vehicles):
            # Losowy krok w czasie (od 0 do max_time)
            depart_time = random.uniform(0, max_time)
            
            orig = random.choice(origins)
            
            # 70% szans, że samochód jedzie do jednego z hotspotów
            if random.random() < 0.70:
                dest = random.choice(bottlenecks)
            else:
                dest = random.choice(destinations)
                
            # Zabezpieczenie przed trasą w to samo miejsce
            while orig == dest and len(destinations) > 1:
                dest = random.choice(destinations)
                
            trips.append({'id': i, 'depart': depart_time, 'orig': orig, 'dest': dest})
            
        # Posortowane wygenerowane tripy po czasie odjazdu (depart) to WYMÓG silnika SUMO
        trips.sort(key=lambda x: x['depart'])
        
        for t in trips:
            f.write(f'  <trip id="veh_{t["id"]}" type="car" depart="{t["depart"]:.2f}" from="{t["orig"]}" to="{t["dest"]}" />\n')
            
        f.write('</routes>\n')
        
    print(f"Wygenerowano pomyślnie plik tripów: {out_trips}")
    
    # Użycie duaroutera do wygenerowania finalnego pliku z trasami
    print("Uruchamianie duarouter do przeliczenia optymalnych tras...")
    cmd = [
        "duarouter",
        "-n", net_file,
        "--route-files", out_trips,
        "-o", out_routes,
        "--ignore-errors",
        "--no-warnings"
    ]
    subprocess.run(cmd, check=True)
    print(f"Plik tras {out_routes} gotowy!")

if __name__ == "__main__":
    import argparse
    # Wywołujemy funkcję w folderze src/City_map/
    cwd = os.path.dirname(__file__) or "."
    net_path = os.path.join(cwd, "city_map.net.xml")
    trips_path = os.path.join(cwd, "trips.trips.xml")
    routes_path = os.path.join(cwd, "city_map.rou.xml")
    
    generate_traffic(net_path, trips_path, routes_path)
