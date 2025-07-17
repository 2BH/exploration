import gymnasium as gym
import minigrid.envs
import random
import pdb
import time

# Initialize the environment (you can choose any environment from gym-minigrid)
env = gym.make('MiniGrid-DoorKeyLO-8x8-v0')

# Reset the environment to start a new episode
obs = env.reset()

# Function to choose a random action from available actions
def get_random_action():
    # Available actions are defined in the environment's action space
    action_space = env.action_space
    return action_space.sample()  # Sample a random action from the action space

# Run a loop to take random actions in the environment
# The loop will run for a few steps or until the episode is done
for step in range(100):
    # Take a random action
    action = int(input("action: "))
    # print(f"action: {action}")
    
    # Optionally, you can insert a breakpoint here to debug
    # pdb.set_trace()  # Uncomment this line to start pdb debugging
    
    # Take the action in the environment
    obs, reward, terminated, truncated, info = env.step(action)

    # Render the environment (if desired, for visualization)
    env.render()

    # Print some information at each step
    print(f"Step {step}: Action {action}, Reward: {reward}, Done: {terminated}")

    # Check if the episode is done
    if terminated:
        print("Episode finished!")
        break

    # Sleep to slow down the rendering, so you can debug
    time.sleep(0.5)

# Close the environment when done
env.close()