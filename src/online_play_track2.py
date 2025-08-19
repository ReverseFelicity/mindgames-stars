"""
Track 2: Generalization Track
This script connects your agent to the online competition for the generalization track.
Environments: Codenames-v0, ColonelBlotto-v0, ThreePlayerIPD-v0
"""
import os
import textarena as ta
from agent import LLMAgent
from dotenv import load_dotenv

# from stars_agent_track2_baseline import StarsAgentTrack2BaseLine
from stars_agent_track2 import StarsAgentTrack2

load_dotenv()



MODEL_NAME = "STARS Agent Track2 qwen3 V1" # Replace with your model name
# The name is used to identify your agent in the online arena and leaderboard.
# It should be unique and descriptive.
# For different versions of your agent, you should use different names.
MODEL_DESCRIPTION = "STARS Agent Track2 qwen3 V1"
team_hash = os.getenv("TEAM_HASH")  # Replace with your team hash

# Initialize your agent
agent = StarsAgentTrack2()


if __name__ == '__main__':
    # print(team_hash)
    for i in range(10):
        # This play 1 game, you could add for loop to play multiple games
        env = ta.make_mgc_online(
            track="Generalization",
            model_name=MODEL_NAME,
            model_description=MODEL_DESCRIPTION,
            team_hash=team_hash,
            agent=agent,
            small_category=True  # Set to True to participate in the efficient division
        )
        env.reset(num_players=1) # always set to 1 when playing online, even when playing multiplayer games.

        done = False
        while not done:
            player_id, observation = env.get_observation()
            action = agent(observation)
            done, step_info = env.step(action=action)

        rewards, game_info = env.close()
