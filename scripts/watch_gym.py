import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Problem: Original script created a NEW model and data from scene.xml
# (mujoco.MjModel.from_xml_path + mujoco.MjData), so env.step() updated
# env.data but the viewer showed a different, unrelated data instance.
# As a result, you couldn't see the gym inference - the viewer didn't reflect
# the actions taken by env.step().
#
# Fix: Use the gym environment's model and data (env.unwrapped.model / env.unwrapped.data)
# so the viewer shows exactly what the gym policy/actions produce. Also set
# MUJOCO_GL to 'osmesa' as a safe fallback for headless environments.
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")

import gymnasium as gym
import mujoco
import mujoco.viewer
import mira

# Allow selecting scene (defaults to clean scene.xml; use scene_with_coffee_mug.xml for props)
scene_name = os.environ.get("MIRA_SCENE", "scene.xml")
scene_path = Path(REPO_ROOT) / "models" / "so101" / scene_name
if not scene_path.exists():
    scene_path = Path(scene_name)

env = gym.make("MIRA/SO101Reach-v0", xml_path=str(scene_path))
obs, info = env.reset(seed=123)

env_unwrapped = env.unwrapped
model = env_unwrapped.model
data = env_unwrapped.data

print("Opening MuJoCo viewer...")
print("This will show EXACTLY what gym actions produce.")
print(f"Starting distance to target: {info['distance']:.3f}m")

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running() and step < 200:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        viewer.sync()
        step += 1
        if terminated or truncated:
            print(f"Episode ended at step {step}. Resetting...")
            obs, info = env.reset()
            step = 0
env.close()
print("Done. Environment closed.")
