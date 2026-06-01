from src.Sumo.sumo_rl import SumoEnvironment
from src.params import ENV_CONFIG, NET_FILE, ROUTE_FILE


def create_rl_env(gui=False):
    """Create a SUMO-RL environment configured for the current city_map_2 setup."""
    env_args = dict(ENV_CONFIG)
    env_args["use_gui"] = gui
    env_args["net_file"] = NET_FILE
    env_args["route_file"] = ROUTE_FILE
    return SumoEnvironment(**env_args)


if __name__ == "__main__":
    print("Testowanie utworzonego środowiska RL...")
    env = create_rl_env(gui=False)

    obs = env.reset()
    if isinstance(obs, tuple):
        obs, info = obs
    print("Struktura obserwacji (Stanu):", obs)

    for _ in range(50):
        actions = {ts_id: env.action_spaces(ts_id).sample() for ts_id in env.ts_ids}

        step_result = env.step(actions)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, terminated, info = step_result
            truncated = False

        if isinstance(terminated, dict):
            if all(terminated.values()):
                break
        elif terminated or truncated:
            break

    print("Przykładowa obserwacja po 50 krokach:", obs)
    print("Przykładowa nagroda:", reward)

    env.close()
    print("Środowisko RL zainicjalizowane z sukcesem!")
