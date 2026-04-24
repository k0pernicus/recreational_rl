import argparse
import numpy as np
from enum import Enum

START = "S"
END = "E"

WORLD = [
    ["",    "", "", "", "", "", "", "",  "",  ""],
    ["",    "", "", "", "", "", "", "",  "",  ""],
    ["",    "", "", "", "", "", "", "",  "",  ""],
    [START, "", "", "", "", "", "", "",  "",  END],
    ["",    "", "", "", "", "", "", "",  "",  ""],
    ["",    "", "", "", "", "", "", "",  "",  ""],
    ["",    "", "", "", "", "", "", "",  "",  ""]
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
    NORTH_EAST = 4
    NORTH_WEST = 5
    SOUTH_EAST = 6
    SOUTH_WEST = 7

START_COORDINATES = (3, 0)

assert(len(WINDY) == len(WORLD[0]))
assert(all(map(lambda r: len(r) == len(WORLD[0]), WORLD)))
assert(WORLD[START_COORDINATES[0]][START_COORDINATES[1]] == START)


def get_action(r, c, q_table, epsilon):
    adirection = (float('-inf'), Direction.NORTH)
    # exploration
    if np.random.rand() <= epsilon: return np.random.choice(list(Direction))
    # exploitation
    for direction in Direction:
        if q_table[direction.value][r][c] > adirection[0]: adirection = (q_table[direction.value][r][c], direction)
    return adirection[1]


def get_action_and_wipe(r, c, q_table, epsilon):
    """
    Used for q_learning, to avoid trace bug (avoid to penalize previous good actions
    is one action was random)
    Returns the direction along with a boolean value to know if we have
    to wipe eligibility traces
    """
    adirection = (float('-inf'), Direction.NORTH)
    # exploration
    if np.random.rand() <= epsilon:
        return np.random.choice(list(Direction)), True
    # exploitation
    for direction in Direction:
        if q_table[direction.value][r][c] > adirection[0]: adirection = (q_table[direction.value][r][c], direction)
    return adirection[1], False


def sarsa_lambda(episodes, learning_rate, gamma, lambda_p, epsilon = 0.1, max_nb_steps=2000):
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

        # epsilon decay
        current_epsilon = max(0.01, 1.0 - (episode / (episodes * 0.8)))

        # Take the first action using epsilon-greedy
        current_action = get_action(r, c, q_table, current_epsilon)

        while True:
            if steps > max_nb_steps: break
            if WORLD[r][c] == END: break # stop as we reached the end

            # SARSA mathematically requires to know two actions to compute the TD error:
            # the action the agent took to arrive in the new state, and the action
            # the agent is about to take.

            nr, nc = r, c
            # Apply the CURRENT action's intended movement
            if current_action == Direction.NORTH: nr -= 1
            elif current_action == Direction.SOUTH: nr += 1
            elif current_action == Direction.WEST: nc -= 1
            elif current_action == Direction.EAST: nc += 1
            elif current_action == Direction.NORTH_EAST: nr -= 1; nc += 1
            elif current_action == Direction.NORTH_WEST: nr -= 1; nc -= 1
            elif current_action == Direction.SOUTH_EAST: nr += 1; nc += 1
            elif current_action == Direction.SOUTH_WEST: nr += 1; nc -= 1
            else: assert(False) # debug

            # Apply the wind based on the column we started in,
            # and clamp the coordinates so the agent cannot fall off the grid boundaries
            nr -= WINDY[c]
            nr = max(0, min(nr, _W_H - 1))
            nc = max(0, min(nc, _W_W - 1))

            # NOW the agent has officially landed in the Next State (S').
            # Look around and pick Next Action (A').
            next_action = get_action(nr, nc, q_table, current_epsilon)

            # First, add +1 to the trace exactly where the agent is standing right now
            eligibility_traces[current_action.value][r][c] += 1.0

            # compute theta and q_table update
            theta = STEP_REWARD + gamma * q_table[next_action.value][nr][nc] - q_table[current_action.value][r][c]
            q_table += (theta * learning_rate * eligibility_traces)

            # Then, fade the entire 3D matrix by multiplying it all by y * lambda
            eligibility_traces *= (gamma * lambda_p)

            steps += 1
            r, c = nr, nc

            current_action = next_action

    return q_table

def q_learning(episodes, learning_rate, gamma, lambda_p, epsilon = 0.1, max_nb_steps=2000):
    # 4 actions: NORTH, SOUTH, EAST, WEST
    q_table = np.zeros((len(Direction), _W_H, _W_W), dtype=float)

    for episode in range(episodes):
        (r, c) = START_COORDINATES
        eligibility_traces = np.zeros((len(Direction), _W_H, _W_W), dtype=float)
        steps = 0

        # epsilon decay
        current_epsilon = max(0.01, 1.0 - (episode / (episodes * 0.8)))

        # Take the first action using epsilon-greedy
        current_action = get_action(r, c, q_table, current_epsilon)

        while True:
            if steps > max_nb_steps: break
            if WORLD[r][c] == END: break # stop as we reached the end

            nr, nc = r, c
            # Apply the CURRENT action's intended movement
            if current_action == Direction.NORTH: nr -= 1
            elif current_action == Direction.SOUTH: nr += 1
            elif current_action == Direction.WEST: nc -= 1
            elif current_action == Direction.EAST: nc += 1
            elif current_action == Direction.NORTH_EAST: nr -= 1; nc += 1
            elif current_action == Direction.NORTH_WEST: nr -= 1; nc -= 1
            elif current_action == Direction.SOUTH_EAST: nr += 1; nc += 1
            elif current_action == Direction.SOUTH_WEST: nr += 1; nc -= 1
            else: assert(False) # debug

            nr -= WINDY[c]
            nr = max(0, min(nr, _W_H - 1))
            nc = max(0, min(nc, _W_W - 1))

            next_action, is_random_action = get_action_and_wipe(nr, nc, q_table, current_epsilon)
            # Watkins trace-cut to prevent random exploratory steps
            # from poisoning good trained data
            if is_random_action: eligibility_traces.fill(0)

            # First, add +1 to the trace exactly where the agent is standing right now
            eligibility_traces[current_action.value][r][c] += 1.0

            # compute theta and q_table update
            # the main difference here is we take the best action among all four
            theta = STEP_REWARD + gamma * np.max(q_table[:,nr,nc]) - q_table[current_action.value][r][c]
            q_table += (theta * learning_rate * eligibility_traces)

            eligibility_traces *= (gamma * lambda_p)

            steps += 1
            r, c = nr, nc

            current_action = next_action

    return q_table

def evaluate_policy(q_table):
    r, c = START_COORDINATES
    path = []
    steps = 0
    visited = set() # infinite loop protection

    print("--- Evaluating Optimal Policy ---")
    while True:
        if (r, c) in visited:
            print("/!\\ Found visited state - stopping")
            break
        path.append((r, c))
        visited.add((r, c))

        if WORLD[r][c] == END:
            break

        # 100% greedy action (epsilon <= 0 guarantees no exploration)
        action = get_action(r, c, q_table, epsilon=-1.0)

        nr, nc = r, c
        # Apply the action's intended movement
        if action == Direction.NORTH: nr -= 1
        elif action == Direction.SOUTH: nr += 1
        elif action == Direction.WEST: nc -= 1
        elif action == Direction.EAST: nc += 1
        elif action == Direction.NORTH_EAST: nr -= 1; nc += 1
        elif action == Direction.NORTH_WEST: nr -= 1; nc -= 1
        elif action == Direction.SOUTH_EAST: nr += 1; nc += 1
        elif action == Direction.SOUTH_WEST: nr += 1; nc -= 1

        # Apply the wind and clamp boundaries
        nr -= WINDY[c]
        nr = max(0, min(nr, _W_H - 1))
        nc = max(0, min(nc, _W_W - 1))

        r, c = nr, nc
        steps += 1

    print(f"Optimal path reached in {len(path) - 1} steps!\n")

    # Visualizing the path on the board
    visual_board = [["." if cell == "" else cell for cell in row] for row in WORLD]

    for (pr, pc) in path:
        if visual_board[pr][pc] == ".": visual_board[pr][pc] = "*"

    # Ensure S and E are still clearly marked
    visual_board[START_COORDINATES[0]][START_COORDINATES[1]] = START

    # Print the board
    for row in visual_board:
        print(" ".join(f"{cell:>2}" for cell in row))

    return path

def run_kings_windy():
    parser = argparse.ArgumentParser(description="Run Kings Move Windy Gridworld")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")

    # SARSA subcommand
    sarsa_parser = subparsers.add_parser("sarsa", help="Run SARSA-lambda")
    sarsa_parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to train")
    sarsa_parser.add_argument("--max-steps-per-episode", type=int, default=2000, help="Maximum number of steps per episode")
    sarsa_parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    sarsa_parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor (gamma)")
    sarsa_parser.add_argument("--lambda", type=float, default=0.5, dest="lambda_p", help="Trace decay (lambda)")
    sarsa_parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate (epsilon)")

    # Q-Learning subcommand
    q_learning_parser = subparsers.add_parser("q-learning", help="Run Q-Learning")
    q_learning_parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to train")
    q_learning_parser.add_argument("--max-steps-per-episode", type=int, default=2000, help="Maximum number of steps per episode")
    q_learning_parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    q_learning_parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor (gamma)")
    q_learning_parser.add_argument("--lambda", type=float, default=0.5, dest="lambda_p", help="Trace decay (lambda)")
    q_learning_parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate (epsilon)")

    args = parser.parse_args()

    if args.command == "sarsa":
        print(f"SARSA")
        # Train the agent
        sarsa_q_table = sarsa_lambda(args.episodes, args.lr, args.gamma, args.lambda_p, args.epsilon, args.max_steps_per_episode)
        # Test the agent
        evaluate_policy(sarsa_q_table)
    elif args.command == "q-learning":
        print(f"Q-Learning")
        # Train the agent
        q_learning_table = q_learning(args.episodes, args.lr, args.gamma, args.lambda_p, args.epsilon, args.max_steps_per_episode)
        # Test the agent
        evaluate_policy(q_learning_table)
    else:
        parser.print_help()

if __name__ == "__main__":
    sarsa_lambda(1000, 0.05, 0.9, 0.5, 0.1)
