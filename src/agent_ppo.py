import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.Sumo.sumo_rl import SumoEnvironment

# ========================================== #
# 1. Definicja sieci neuronowej (Od Zera)    #
# Zbudujemy model Actor-Critic, który będzie #
# sterował jednym skrzyżowaniem              #
# ========================================== #
class PPOAgent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(PPOAgent, self).__init__()
        
        # Sieć policy (Aktor) - patrzy na stan (obs) i wyrzuca "prawdopodobieństwa" dla ruchów
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )
        
        # Sieć wartości (Krytyk) - patrzy na stan (obs) i ocenia nagrodę jaka z niego wynika 
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def get_action_and_value(self, obs, action=None):
        logits = self.actor(obs)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), self.critic(obs).squeeze(-1)

# ========================================== #
# 2. Definicja Hyperparametrów algorytmu     #
# ========================================== #
LEARNING_RATE = 3e-4
ROLLOUT_STEPS = 200          # Co ile kroków wgrywamy nowe doświadczenia do agenta
EPOCHS = 4                   # Ile razy próbujemy doskonalić model z zebranych danych
GAMMA = 0.99                 # Dyskontowanie nagród
GAE_LAMBDA = 0.95            # Z jaką siłą ufać przyszłości vs przeszłości (do przewidywań)
CLIP_FRAC = 0.2              # Ucinanie PPO w celu unikania drastycznych zmian z błędem

# Główna funkcja treningowa
def train():
    map_dir = os.path.join(os.path.dirname(__file__), 'City_map')
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
    
    # Tworzymy osobnego agenta dla każdego skrzyżowania w mieście
    agents = {}
    optimizers = {}
    for ts in ts_ids:
        obs_dim = env.observation_spaces(ts).shape[0]
        act_dim = env.action_spaces(ts).n
        agents[ts] = PPOAgent(obs_dim, act_dim)
        optimizers[ts] = optim.Adam(agents[ts].parameters(), lr=LEARNING_RATE)
        
    print(f"Stworzono Agentów od zera (PyTorch) dla {len(ts_ids)} skrzyżowań.")
    
    obs_dict = env.reset()
    if isinstance(obs_dict, tuple): obs_dict = obs_dict[0]

    for update in range(1, 50): # Uczymy 50 pętli po "ROLLOUT_STEPS"
        print(f"\nUruchamiam epokę uczenia #{update}")
        
        # Rejestry pamięci na to czego agenci się nauczyli
        memories = {ts: {'obs': [], 'actions': [], 'logprobs': [], 'rewards': [], 'values': [], 'dones': []} for ts in ts_ids}
        
        for step in range(ROLLOUT_STEPS):
            actions_dict = {}
            for ts in ts_ids:
                # Agent widzi swój świat
                obs_ts = torch.tensor(obs_dict[ts], dtype=torch.float32)
                
                # Zamiast zgadywać w ciemno, używa naszego Pytorchowego Agenta:
                action, logprob, _, value = agents[ts].get_action_and_value(obs_ts)
                actions_dict[ts] = action.item()
                
                # Zapisujemy pamięć
                memories[ts]['obs'].append(obs_ts)
                memories[ts]['actions'].append(action)
                memories[ts]['logprobs'].append(logprob)
                memories[ts]['values'].append(value)
            
            # Pchanie całego środowiska i sprawdzanie, czy opłacało się zmienić to światło
            res = env.step(actions_dict)
            if len(res) == 5:
                next_obs_dict, rewards_dict, dones_dict, _, _ = res
            else:
                next_obs_dict, rewards_dict, dones_dict, _ = res
                
            for ts in ts_ids:
                memories[ts]['rewards'].append(rewards_dict[ts])
                
                d = dones_dict if not isinstance(dones_dict, dict) else dones_dict[ts]
                memories[ts]['dones'].append(1.0 if d else 0.0)
            
            obs_dict = next_obs_dict
            if isinstance(dones_dict, dict) and all(dones_dict.values()):
                obs_dict = env.reset()
                if isinstance(obs_dict, tuple): obs_dict = obs_dict[0]

        # 3. Odpalamy zaktualizowanie mózgów na tym co zebraliśmy (Algorytm PPO)
        for ts in ts_ids:
            obs_t = torch.stack(memories[ts]['obs'])
            act_t = torch.stack(memories[ts]['actions'])
            logp_t = torch.stack(memories[ts]['logprobs'])
            rew_t = torch.tensor(memories[ts]['rewards'], dtype=torch.float32)
            val_t = torch.stack(memories[ts]['values'])
            don_t = torch.tensor(memories[ts]['dones'], dtype=torch.float32)
            
            # Wliczanie "Korzyści" - Advantage (czy zrobiliśmy lepiej niż się spodziewaliśmy)
            advantages = torch.zeros_like(rew_t)
            lastgaelam = 0
            for t in reversed(range(ROLLOUT_STEPS)):
                if t == ROLLOUT_STEPS - 1:
                    nextnonterminal = 1.0 - don_t[t]
                    with torch.no_grad():
                        next_obs = torch.tensor(obs_dict[ts], dtype=torch.float32)
                        nextvalues = agents[ts].critic(next_obs).squeeze(-1)
                else:
                    nextnonterminal = 1.0 - don_t[t]
                    nextvalues = val_t[t+1]
                
                delta = rew_t[t] + GAMMA * nextvalues * nextnonterminal - val_t[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                
            returns = advantages + val_t
            
            # Właściwa poprawka wag z klipowaniem
            for epoch in range(EPOCHS):
                # Otrzymanie NOWYCH logprob i values dla starych stanów z ODŁĄCZONĄ
                # od przeszłości strukturą. Rozwiązuje to błąd "second time backward" w PyTorchu.
                # Zabezpieczamy historię roboczą (.detach()) żeby liczyło gradient tylko 
                # z wagi nowej pętli aktora po danej epoce
                _, newlogprob, entropy, newvalue = agents[ts].get_action_and_value(obs_t, act_t)
                
                logratio = newlogprob - logp_t.detach()
                ratio = logratio.exp()
                
                # Funkcje PPO - bierzemy mniejszy (zdrowszy) gradient
                adv_detached = advantages.detach()
                pg_loss1 = -adv_detached * ratio
                pg_loss2 = -adv_detached * torch.clamp(ratio, 1 - CLIP_FRAC, 1 + CLIP_FRAC)
                actor_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                critic_loss = 0.5 * ((newvalue - returns.detach()) ** 2).mean()
                entropy_loss = entropy.mean()
                
                loss = actor_loss - 0.01 * entropy_loss + 0.5 * critic_loss
                
                optimizers[ts].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agents[ts].parameters(), 0.5)
                optimizers[ts].step()

        print(f"Średnia nagroda na agenta {ts_ids[0]}: {rew_t.mean().item():.3f}")

    # ===== ZAPISYWANIE MODELU PO ZAKOŃCZENIU TRENINGU =====
    # Zbiera wszystkie nauczone wagi z każdej z niezależnych sieci i zapisuje do jednego pliku .pth
    out_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "ppo_models_weights.pth")
    
    torch.save({ts: agents[ts].state_dict() for ts in ts_ids}, model_path)
    print(f"\nTrening zakończony pomyślnie. Wagi zapisano w {model_path}.")
                     
if __name__ == "__main__":
    print("Mój Dedykowany PPO PyTorch wędruje do pamięci operacyjnej...")
    train()