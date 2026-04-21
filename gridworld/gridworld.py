import numpy as np

START = "S"
OUTPUT = "O"

WORLD = [
    [START, "",  "", "",    "x"],
    ["",    "",  "", "",    "x"],
    ["",    "", "x", "",     ""],
    ["",    "",  "", "",     ""],
    ["x",  "x",  "", "", OUTPUT],
]

REWARDS = [
    [0,     0,   0, 0, -50],
    [0,     0,   0, 0, -50],
    [0,     0, -50, 0,   0],
    [0,     0,   0, 0,   0],
    [-50, -50,   0, 0, 10],
]

WORLD_DIMENSIONS = (len(WORLD), len(WORLD[0]))
_W_H, _W_W = WORLD_DIMENSIONS[0], WORLD_DIMENSIONS[1]

# Basic checks
assert(len(REWARDS) == _W_H) # Ensures the WORLD and REWARDS nb of rows is the same
assert(len(set(map(lambda row: len(row), WORLD))) == 1) # Ensures the WORLD nb of columns is the same per row
assert(len(REWARDS[0]) == _W_W) # Ensures the WORLD and REWARDS nb of columns is the same

def policy_evaluation(iterations, gamma):
    values = np.zeros(WORLD_DIMENSIONS, dtype=float)
    values[_W_H - 1][_W_W - 1] = REWARDS[_W_H - 1][_W_W - 1]

    for i in range(iterations):
        new_values = np.copy(values)
        for r in range(_W_H):
            for c in range(_W_W):
                if WORLD[r][c] == OUTPUT: continue
                possible_neighbors = [
                    (r-1, c), (r+1, c), (r, c-1), (r, c+1)
                ]
                possible_neighbors = list(filter(lambda c: c[0] >= 0 and c[0] < _W_H and c[1] >= 0 and c[1] < _W_W, possible_neighbors))
                neighbor_rewards = list(map(lambda c: values[c[0]][c[1]], possible_neighbors))
                new_values[r][c] = REWARDS[r][c] + gamma * (sum(neighbor_rewards) / len(neighbor_rewards))

        if np.allclose(values, new_values):
            print(f"Converged at iteration {i}")
            break

        values = new_values

    return values

def value_iteration(iterations, gamma):
    values = np.zeros(WORLD_DIMENSIONS, dtype=float)
    values[4][4] = REWARDS[4][4]

    for i in range(iterations):
        new_values = np.copy(values)
        for r in range(_W_H):
            for c in range(_W_W):
                if WORLD[r][c] == OUTPUT: continue
                possible_neighbors = [
                    (r-1, c), (r+1, c), (r, c-1), (r, c+1)
                ]
                possible_neighbors = list(filter(lambda c: c[0] >= 0 and c[0] < _W_H and c[1] >= 0 and c[1] < _W_W, possible_neighbors))
                neighbor_rewards = list(map(lambda c: values[c[0]][c[1]], possible_neighbors))
                new_values[r][c] = REWARDS[r][c] + gamma * max(neighbor_rewards) # here, we take the best value

        if np.allclose(values, new_values):
            print(f"Converged at iteration {i}")
            break

        values = new_values

    return values

def td_lambda(episodes, learning_rate, gamma, lambda_p):
    """
    Evaluating state-values V(s) (Prediction)
    """
    if lambda_p < 0.0 or lambda_p > 1.0:
        raise(f"lambda parameter of td_lambda should be between 0 and 1, got {lambda_p}")

    values = np.zeros(WORLD_DIMENSIONS, dtype=float)

    for i in range(episodes):
        # Start the agent at random place, each time you run a new episode
        r, c = np.random.randint(0, _W_H), np.random.randint(0, _W_W)
        if r == c == (_W_W - 1): c = np.random.randint(0, _W_W - 1) # Do not start with the OUTPUT
        eligibility_traces = np.zeros(WORLD_DIMENSIONS, dtype=float) # keep a trace of all footprints of the agent
        while True:
            if WORLD[r][c] == OUTPUT: break # begin another episode
            # do not check for 'x' as those are not terminal states, only OUTPUT is

            possible_neighbors = [
                (r-1, c), (r+1, c), (r, c-1), (r, c+1)
            ]
            possible_neighbors = list(filter(lambda c: c[0] >= 0 and c[0] < _W_H and c[1] >= 0 and c[1] < _W_W, possible_neighbors))
            np.random.shuffle(possible_neighbors)
            nr, nc = possible_neighbors[0]

            t_reward = REWARDS[nr][nc] # transition reward
            td_error = t_reward + gamma * values[nr][nc] - values[r][c]
            eligibility_traces[r][c] += 1

            for wr in range(_W_H):
                for wc in range(_W_W):
                    values[wr][wc] += learning_rate * td_error * eligibility_traces[wr][wc] # update the value
                    eligibility_traces[wr][wc] *= lambda_p * gamma # fade the footprints

            r, c = nr, nc

    return values

def run_gridworld():
    print(f"POLICY EVALUATION")
    values = policy_evaluation(100, 0.9)
    print(values)
    print(f"=" * 80)
    print(f"VALUE ITERATION")
    values = value_iteration(100, 0.9)
    print(values)
    print(f"=" * 80)
    print(f"TD-Lambda")
    values = td_lambda(5000, learning_rate=0.05, gamma=0.9, lambda_p=0.5)
    print(values)
