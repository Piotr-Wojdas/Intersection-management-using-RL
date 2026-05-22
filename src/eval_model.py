import os
import sys
import torch
import numpy as np
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.Sumo.sumo_rl import SumoEnvironment
from src.agent_ppo import PPOAgent

def play():
    map_dir = os.path.join(os.path.dirname(__file__), 'City_map')
    # Uruchamiamy środowisko ze zmienną 'use_gui=True' aby włączało się okienko SUMO-GUI
    env = SumoEnvironment(
        net_file=os.path.join(map_dir, 'city_map.net.xml'),
        route_file=os.path.join(map_dir, 'city_map.rou.xml'),
        use_gui=False,
        num_seconds=10000, 
        delta_time=5,
        yellow_time=3,
        min_green=5,
        max_depart_delay=-1
    )
    
    ts_ids = env.ts_ids
    
    # Inicjalizujemy puste sieci neuronowe tak samo jak przy treningu
    agents = {}
    for ts in ts_ids:
        obs_dim = env.observation_spaces(ts).shape[0]
        act_dim = env.action_spaces(ts).n
        agents[ts] = PPOAgent(obs_dim, act_dim)
        # Przełączamy z trybu .train() na tzw. .eval() - model nie gromadzi już śladów na gradienty, 
        # lecz ma za zadanie wyłącznie podejmować optymalne decyzje używając małej wagi procesora.
        agents[ts].eval() 

    # Ładujemy wyuczone mózgi z pliku ppo_models_weights.pth
    model_path = os.path.join(os.path.dirname(__file__), 'outputs', 'ppo_models_weights.pth')
    if os.path.exists(model_path):
        saved_weights = torch.load(model_path)
        for ts in ts_ids:
            agents[ts].load_state_dict(saved_weights[ts])
        print(f"Pomyślnie wczytano pamięć operacyjną agentów z pliku {model_path}.")
    else:
        print(f"UWAGA: Nie odnaleziono wytrenowanych wag w {model_path}.")
        print("Agenci użyją losowego 'niemowlęcego' myślenia!")
        
    obs_dict = env.reset()
    if isinstance(obs_dict, tuple): obs_dict = obs_dict[0]
    
    print("\nZaczynamy fizyczną jazdę!")
    done = False
    
    while not done:
        actions_dict = {}
        for ts in ts_ids:
            obs_ts = torch.tensor(obs_dict[ts], dtype=torch.float32)
            
            # W trybie oceny odcinamy model od szumów i uczenia, pobieramy po prostu decyzyjnie akcje z największym wskaźnikiem poprawności
            with torch.no_grad():
                logits = agents[ts].actor(obs_ts)
                # Maksymalizacja logits bez stochastycznego błądzenia - agent ufa swojemu doświadczeniu w 100%
                action = torch.argmax(logits).item()
                actions_dict[ts] = action
                
        # Agent wykonuje swój ruch 
        res = env.step(actions_dict)
        if len(res) == 5:
            obs_dict, rewards_dict, dones_dict, _, _ = res
        else:
            obs_dict, rewards_dict, dones_dict, _ = res
            
        time.sleep(0.05) # Delikatny delay by gołym okiem dało się podziwiać efekty w GUI symulacji :)
            
        d = dones_dict if not isinstance(dones_dict, dict) else dones_dict[ts_ids[0]]
        if d:
            done = True
            
    print("Ocenianie Agenta zakończone. Trasy całkowicie obsłużone.")
    env.close()

if __name__ == "__main__":
    play()