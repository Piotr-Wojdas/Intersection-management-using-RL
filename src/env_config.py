import os
import sys

# Dodajemy folder główny 'src' do ścieżki Pythona, aby móc swobodnie importować moduły
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym
from src.Sumo.sumo_rl import SumoEnvironment

def create_rl_env(gui=False):
    """
    Krok 2: Tworzenie i konfiguracja środowiska RL.
    """
    map_dir = os.path.join(os.path.dirname(__file__), 'City_map')
    net_file = os.path.join(map_dir, 'city_map.net.xml')
    route_file = os.path.join(map_dir, 'city_map.rou.xml')
    
    # Tworzymy folder na wyniki w src/outputs 
    out_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    # Inicjalizacja środowiska SUMO-RL
    # Domyślnie używa 'diff-waiting-time' (redukcja czasu oczekiwania) jako nagrody (reward).
    # Obserwacje obejmują gęstość ruchu i długość kolejek na pasach.
    # Akcje: 0 (zatrzymaj aktualną fazę zieloną), 1 (przejdź do kolejnej fazy).
    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        out_csv_name=os.path.join(out_dir, "rl_metrics"),
        use_gui=gui,
        num_seconds=3600,             # Ile sekund symulacji przypada na jeden epizod (tutaj 1h)
        delta_time=5,                 # Co ile sekund symulacji agent podejmuje decyzję
        yellow_time=3,                # Czas trwania strefy żółtej przy zmianie świateł
        min_green=5,                  # Minimalny czas trwania zielonego światła
        max_depart_delay=0,           # Pojazdy nie czekają w nieskończoność na błąd, jeśli nie mogą wjechać
        time_to_teleport=300          # Aby uniknąć całkowitego zablokowania w złym modelu (teleport po 5 minutach stania)
    )
    
    return env

if __name__ == "__main__":
    print("Testowanie utworzonego środowiska RL...")
    env = create_rl_env(gui=False)
    
    # Resetowanie środowiska przed rozpoczęciem pierwszego epizodu
    obs = env.reset()
    # W nowyszym gymasium reset() zwraca tuple (obs, info),
    # ale multi-agent sumo-rl (ze starym API) potrafi zwrócić po prostu słownik!
    if isinstance(obs, tuple):
        obs, info = obs
    print("Struktura obserwacji (Stanu):", obs)
    
    # Symulacja kilku losowych kroków agenta (akcji)
    for _ in range(50):
        # Ponieważ jest to środowisko z wieloma skrzyżowaniami (Multi-Agent),
        # musimy wygenerować akcję dla KAŻDEGO z nich podając słownik:
        # np: {'J0': 0, 'J1': 1, ...}
        actions = {ts_id: env.action_spaces(ts_id).sample() for ts_id in env.ts_ids}
        
        step_result = env.step(actions)
        # Podobnie obsługa różnych API gym
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, terminated, info = step_result
            truncated = False
        
        # Przerwanie jeśli skończyliśmy epizod
        # terminated może być pojedynczym booleanem albo słownikiem dla poszczególnych skrzyżowań
        if isinstance(terminated, dict):
            if all(terminated.values()): break
        elif terminated or truncated:
            break
            
    print("Przykładowa obserwacja po 50 krokach:", obs)
    print("Przykładowa nagroda:", reward)
    
    env.close()
    print("Środowisko RL zainicjalizowane z sukcesem!")
