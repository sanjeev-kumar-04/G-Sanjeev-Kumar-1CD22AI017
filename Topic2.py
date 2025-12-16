# =========================================
# Graph-Based Q-Learning
# Autonomous Robot Navigation in Warehouse
# =========================================

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

# -------------------------------
# 1. Warehouse Layout (Graph)
# -------------------------------

edges = [
    (0, 1), (1, 5), (5, 6), (5, 4), (1, 2),
    (1, 3), (9, 10), (2, 4), (0, 6), (6, 7),
    (8, 9), (7, 8), (1, 7), (3, 9)
]

goal = 10            # Delivery location
num_states = 11
gamma = 0.75

# Hazard zones and charging stations
hazards = [2, 4, 5]
charging_stations = [3, 8, 9]

# -------------------------------
# 2. Visualizing Warehouse Graph
# -------------------------------

G = nx.Graph()
G.add_edges_from(edges)
pos = nx.spring_layout(G, seed=42)

plt.figure()
nx.draw(G, pos, with_labels=True, node_size=700)
plt.title("Warehouse Navigation Graph")
plt.show()

# -------------------------------
# 3. Reward Matrix
# -------------------------------

R = -1 * np.ones((num_states, num_states))

for edge in edges:
    R[edge] = 100 if edge[1] == goal else 0
    R[edge[::-1]] = 100 if edge[0] == goal else 0

R[goal, goal] = 100

# -------------------------------
# 4. Q-Matrix & Environment Memory
# -------------------------------

Q = np.zeros((num_states, num_states))
env_hazards = np.zeros((num_states, num_states))
env_charging = np.zeros((num_states, num_states))

# -------------------------------
# 5. Utility Functions
# -------------------------------

def available_actions(state):
    return np.where(R[state] >= 0)[0]

def sample_next_action(actions):
    return int(random.choice(actions))

def observe_environment(action):
    obs = []
    if action in hazards:
        obs.append('h')
    if action in charging_stations:
        obs.append('c')
    return obs

def update_q(state, action):
    max_future = np.max(Q[action])
    Q[state, action] = R[state, action] + gamma * max_future

    env = observe_environment(action)
    if 'h' in env:
        env_hazards[state, action] += 1
    if 'c' in env:
        env_charging[state, action] += 1

    return np.sum(Q)

# -------------------------------
# 6. Training Phase
# -------------------------------

scores = []

for _ in range(1000):
    state = random.randint(0, num_states - 1)
    actions = available_actions(state)
    action = sample_next_action(actions)
    score = update_q(state, action)
    scores.append(score)

# -------------------------------
# 7. Training Convergence Plot
# -------------------------------

plt.figure()
plt.plot(scores)
plt.xlabel("Iterations")
plt.ylabel("Reward")
plt.title("Training Convergence (Warehouse Robot)")
plt.show()

# -------------------------------
# 8. Testing Phase
# -------------------------------

current_state = 0
optimal_path = [current_state]

while current_state != goal:
    next_state = int(np.argmax(Q[current_state]))
    optimal_path.append(next_state)
    current_state = next_state

print("Optimal Robot Path:")
print(optimal_path)

# -------------------------------
# 9. Environment Statistics
# -------------------------------

print("\nHazard Encounter Matrix:")
print(env_hazards)

print("\nCharging Station Encounter Matrix:")
print(env_charging)

# -------------------------------
# 10. Environment-Aware Learning
# -------------------------------

def available_actions_with_env(state):
    actions = available_actions(state)
    q_vals = Q[state, actions]
    safe_actions = actions[q_vals >= 0]
    return safe_actions if len(safe_actions) > 0 else actions

scores_env = []

for _ in range(1000):
    state = random.randint(0, num_states - 1)
    actions = available_actions_with_env(state)
    action = sample_next_action(actions)
    score = update_q(state, action)
    scores_env.append(score)

# -------------------------------
# 11. Environment-Aware Plot
# -------------------------------

plt.figure()
plt.plot(scores_env)
plt.xlabel("Iterations")
plt.ylabel("Reward")
plt.title("Environment-Aware Training Convergence")
plt.show()
