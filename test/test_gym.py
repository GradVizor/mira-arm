import sys
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym
import mira

print("Creating MIRA/SO101Reach-v0 environment...")
# Default to clean scene; can override with MIRA_SCENE env var (e.g., scene_with_coffee_mug.xml)
scene_name = os.environ.get("MIRA_SCENE", "scene.xml")
scene_path = Path(REPO_ROOT) / "models" / "so101" / scene_name
if not scene_path.exists():
    scene_path = Path(scene_name)
env = gym.make("MIRA/SO101Reach-v0", xml_path=str(scene_path))

print("Resetting environment...")
obs, info = env.reset(seed=42)
print("Obs shape:", obs.shape)
print("Info:", info)

print("\n--- Doing 10 random steps... ---")
for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i+1}: dist={info['distance']:.3f}m | reward={reward:.3f} | done={terminated or truncated}")
    if terminated or truncated:
        print("Episode ended early!")
        obs, info = env.reset(seed=None)
        print("Reset for new episode")
env.close()
print("\nDone. Environment closed.")