import numpy as np
from enum import Enum

START = "S"
END = "E"

WORLD = [
    ["",    "", "", "", "", "", "", "", "",  ""],
    ["",    "", "", "", "", "", "", "", "",  ""],
    ["",    "", "", "", "", "", "", "", "",  ""],
    [START, "", "", "", "", "", "", "", "", END],
    ["",    "", "", "", "", "", "", "", "",  ""],
    ["",    "", "", "", "", "", "", "", "",  ""],
    ["",    "", "", "", "", "", "", "", "",  ""]
]
WORLD_DIMENSIONS = (len(WORLD), len(WORLD[0]))
_W_H, _W_W = WORLD_DIMENSIONS[0], WORLD_DIMENSIONS[1]

WINDY = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0] # 10 different pushes

STEP_REWARD = -1

class Direction(Enum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

START_COORDINATES = (3, 0)

assert(len(WINDY) == len(WORLD[0]))
assert(all(map(lambda r: len(r) == len(WORLD[0]), WORLD)))
assert(WORLD[START_COORDINATES[0]][START_COORDINATES[1]] == START)


def get_action(r, c, q_table, epsilon):
    adirection = (-float(10000), Direction.NORTH)
    if np.random.rand() > epsilon:
        # exploitation
        for direction in Direction:
            if direction == Direction.NORTH:
                if q_table[direction.value][r][c] > adirection[0]: adirection = (q_table[direction.value][r][c], Direction.NORTH)
            if direction == Direction.SOUTH:
                if q_table[direction.value][r][c] > adirection[0]: adirection = (q_table[direction.value][r][c], Direction.SOUTH)
            if direction == Direction.EAST:
                if q_table[direction.value][r][c] > adirection[0]: adirection = (q_table[direction.value][r][c], Direction.EAST)
            if direction == Direction.WEST:
                if q_table[direction.value][r][c] > adirection[0]: adirection = (q_table[direction.value][r][c], Direction.WEST)
        return adirection[1]
    else:
        # exploration
        return np.random.choice(list(Direction))


def sarsa_lambda(episodes, learning_rate, gamma, lambda_p, epsilon = 0.1):
    # SARSA (on-policy)
    # Q(s,a) = Q(s,a) + alpha * [ Rt+1 + gamma * Q(st, qt) - Q(s, a) ]

    # SARSA-lambda
    # E0(s,a) = 0
    # Et(s,a) = y * lambda * E(t-1)(s,a) + 1(St=s, At=a)
    # Q(s,a) = Q(s,a) + alpha * [ Rt+1 + gamma * Q(st, qt) - Q(s, a) ]

    # 4 actions: NORTH, SOUTH, EAST, WEST
    q_table = np.zeros((len(Direction), _W_H, _W_W), dtype=float)

    for episode in range(episodes):
        (r, c) = START_COORDINATES
        eligibility_traces = np.zeros((len(Direction), _W_H, _W_W), dtype=float)
        steps = 0

        # Take the first action using epsilon-greedy
        current_action = get_action(r, c, q_table, epsilon)
        print(f"Choosing first direction for episode {episode}: {current_action}")

        while True:
            # SARSA mathematically requires to know two actions to compute the TD error:
            # the action the agent took to arrive in the new state, and the action
            # the agent is about to take.

            if WORLD[r][c] == END: break # stop as we reached the end

            nr, nc = r, c
            # Apply the CURRENT action's intended movement
            if current_action == Direction.NORTH: nr -= 1
            elif current_action == Direction.SOUTH: nr += 1
            elif current_action == Direction.WEST: nc -= 1
            elif current_action == Direction.EAST: nc += 1

            # Apply the wind based on the column we started in,
            # and clamp the coordinates so the agent cannot fall off the grid boundaries
            nr -= WINDY[c]
            nr = max(0, min(nr, _W_H - 1))
            nc = max(0, min(nc, _W_W - 1))

            # NOW the agent has officially landed in the Next State (S').
            # Look around and pick Next Action (A').
            next_action = get_action(nr, nc, q_table, epsilon)

            # First, add +1 to the trace exactly where the agent is standing right now
            eligibility_traces[current_action.value][r][c] += 1

            # compute theta and q_table update
            theta = STEP_REWARD + gamma * q_table[next_action.value][nr][nc] - q_table[current_action.value][r][c]
            q_table += (theta * learning_rate * eligibility_traces)

            # Then, fade the entire 3D matrix by multiplying it all by y * lambda
            eligibility_traces *= (gamma * lambda_p)

            steps += 1
            r, c = nr, nc

            current_action = next_action

    return q_table

def run_windy():
    sarsa_q_table = sarsa_lambda(1000, 0.05, 0.9, 0.5)
    print(sarsa_q_table)

    # Q-Learning (off-policy)
    # Q(s,a) = Q(s,a) + alpha * ( Rt+1 + argmax(gamma * maxQ(st, qt) - Q(s, a)) )
    pass
