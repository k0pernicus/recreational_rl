import numpy as np

START = "S"
OUTPUT = "O"

WORLD = [
    [START, "", "", "", ""],
    ["", "", "", "", ""],
    ["", "", "", "", ""],
    ["", "", "", "", ""],
    ["", "", "", "", OUTPUT],
]

REWARDS = [
    [0,     0,   0, 0, -50],
    [0,     0,   0, 0, -50],
    [0,     0, -50, 0,   0],
    [0,     0,   0, 0,   0],
    [-50, -50,   0, 0, 10],
]

def policy_evaluation(iterations, gamma):
    values = np.zeros((5, 5), dtype=float)
    values[4][4] = REWARDS[4][4]

    for i in range(iterations):
        new_values = np.copy(values)
        for r in range(len(WORLD)):
            for c in range(len(WORLD[0])):
                if WORLD[r][c] == OUTPUT: continue
                possible_neighbors = [
                    (r-1, c), (r+1, c), (r, c-1), (r, c+1)
                ]
                possible_neighbors = list(filter(lambda c: c[0] >= 0 and c[0] < len(WORLD) and c[1] >= 0 and c[1] < len(WORLD[0]), possible_neighbors))
                neighbor_rewards = list(map(lambda c: values[c[0]][c[1]], possible_neighbors))
                new_values[r][c] = REWARDS[r][c] + gamma * (sum(neighbor_rewards) / len(neighbor_rewards))

        if np.allclose(values, new_values):
            print(f"Converged at iteration {i}")
            break

        values = new_values

    return values

def value_iteration(iterations, gamma):
    values = np.zeros((5, 5), dtype=float)
    values[4][4] = REWARDS[4][4]

    for i in range(iterations):
        new_values = np.copy(values)
        for r in range(len(WORLD)):
            for c in range(len(WORLD[0])):
                if WORLD[r][c] == OUTPUT: continue
                possible_neighbors = [
                    (r-1, c), (r+1, c), (r, c-1), (r, c+1)
                ]
                possible_neighbors = list(filter(lambda c: c[0] >= 0 and c[0] < len(WORLD) and c[1] >= 0 and c[1] < len(WORLD[0]), possible_neighbors))
                neighbor_rewards = list(map(lambda c: values[c[0]][c[1]], possible_neighbors))
                new_values[r][c] = REWARDS[r][c] + gamma * max(neighbor_rewards) # here, we take the best value

        if np.allclose(values, new_values):
            print(f"Converged at iteration {i}")
            break

        values = new_values

    return values

if __name__ == "__main__":
    print(f"POLICY EVALUATION")
    values = policy_evaluation(100, 0.9)
    print(values)
    print(f"=" * 80)
    print(f"VALUE ITERATION")
    values = value_iteration(100, 0.9)
    print(values)
