"""
This script allows you to play against a fixed model.
You (human player) will be player 0, and the AI model will be player 1.
"""

import textarena as ta 
from agent import HumanAgent
from stars_agent_track2_baseline import StarsAgentTrack2BaseLine

# initialize the agents
agents = {
    0: HumanAgent(),
    1: StarsAgentTrack2BaseLine(),
}

# initialize the environment
env = ta.make(env_id="ColonelBlotto-v0")
# env = ta.make(env_id="Codenames-v0")
# env = ta.make(env_id="ThreePlayerIPD-v0")
env.reset(num_players=len(agents))


if __name__ == "__main__":
    # main game loop
    done = False
    while not done:
      player_id, observation = env.get_observation()
      action = agents[player_id](observation)
      done, step_info = env.step(action=action)
    rewards, game_info = env.close()

    print(f"Rewards: {rewards}")
    print(f"Game Info: {game_info}")