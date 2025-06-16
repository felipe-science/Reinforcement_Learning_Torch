import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import count

# Definir dispositivo
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Definir a rede DQN (igual à usada no treinamento)
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# Criar o ambiente para visualização
env = gym.make("CartPole-v1", render_mode='human')
state, info = env.reset()

n_observations = len(state)
n_actions = env.action_space.n

# Instanciar a rede e carregar os pesos treinados
model = DQN(n_observations, n_actions).to(device)
model.load_state_dict(torch.load('dqn_cartpole.pth', map_location=device))
model.eval()  # Coloca o modelo em modo de avaliação

print("Modelo carregado com sucesso. Iniciando simulação...")

num_eval_episodes = 5  # Número de episódios de simulação

for i_episode in range(num_eval_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0

    for t in count():
        with torch.no_grad():
            action = model(state).max(1).indices.view(1, 1)

        observation, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        state = next_state

        if done:
            print(f"Episódio {i_episode + 1} finalizado. Duração: {t + 1}, Recompensa total: {total_reward}")
            break

env.close()
print("Simulação completa.")
