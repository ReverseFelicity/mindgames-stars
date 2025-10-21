"""
Track 2: Generalization Track
This script connects your agent to the online competition for the generalization track.
Environments: Codenames-v0, ColonelBlotto-v0, ThreePlayerIPD-v0
"""
import os
import textarena as ta
import time
from agent import LLMAgent
from dotenv import load_dotenv

# from stars_agent_track2_baseline import StarsAgentTrack2BaseLine
# from stars_agent_track2_v4 import StarsAgentTrack2V4
from stars_agent_track2_v7 import StarsAgentTrack2V7

load_dotenv()



MODEL_NAME = "STARS Agent Track2 V7" # Replace with your model name
# The name is used to identify your agent in the online arena and leaderboard.
# It should be unique and descriptive.
# For different versions of your agent, you should use different names.
MODEL_DESCRIPTION = "STARS Agent Track2 V7"
team_hash = os.getenv("TEAM_HASH")  # Replace with your team hash

# Initialize your agent
agent = StarsAgentTrack2V7()


def get_game_name(observation: str):
    if "ColonelBlotto" in observation:
        return "ColonelBlotto"
    elif "Iterated Prisoner" in observation:
        return "Iterated Prisoner"
    return "Codenames"


if __name__ == '__main__':
    # print(team_hash)

    valid_game_count = {}
    for i in range(1):
        try:
            print(f"==================================================")
            print(f"====================== {i+1} =====================")
            print(f"==================================================")
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
            game_name = "None"
            while not done:
                player_id, observation = env.get_observation()
                game_name = get_game_name(observation)
                action = agent(observation)
                done, step_info = env.step(action=action)

            rewards, game_info = env.close()
            print(f"[rewards] {rewards}  [game_info]  {game_info}")

            if game_name not in valid_game_count:
                valid_game_count[game_name] = 0
            if len(str(rewards)) > 5:
                valid_game_count[game_name] += 1

            print(f"=================== {valid_game_count} ===================")
            time.sleep(10)
        except BaseException as e:
            print(e)
