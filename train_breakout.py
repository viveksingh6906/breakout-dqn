"""
Train a DQN agent on Atari Breakout, unattended, on Kaggle.

Everything the run produces is written to /kaggle/working, which Kaggle keeps
as the "Output" of the run:
  - dqn_breakout_final.zip       trained model
  - checkpoints/                 intermediate models
  - learning_curve.png           score vs. training steps
  - breakout_dqn.mp4             video of the trained agent playing
  - results.json                 evaluation score, training time, steps done
  - logs/*.monitor.csv           raw score of every training game

Requirements on the Kaggle side: GPU accelerator ON and Internet ON
(internet is only needed for the pip install below).
"""
import os
import sys
import time
import json
import glob
import subprocess

# ----------------------------------------------------------------- settings
ENV_ID = "BreakoutNoFrameskip-v4"   # game
TOTAL_TIMESTEPS = 5_000_000         # training length (agent steps)
MAX_TRAIN_MINUTES = 420             # safety stop: training ends after this, even if steps remain
SEED = 42
N_EVAL_EPISODES = 10                # full games used for the final score
VIDEO_MAX_STEPS = 5000              # safety cap for the recorded game
OUT_DIR = "/kaggle/working"         # Kaggle saves this folder as the run's output

CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUT_DIR, "logs")
for d in (OUT_DIR, CKPT_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)


# ----------------------------------------------------------------- install
def install_packages():
    # Needs Internet = ON in the Kaggle kernel settings.
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "stable-baselines3[extra]", "gymnasium[atari]", "ale-py",
        "imageio", "imageio-ffmpeg",
    ])


install_packages()

# Imports come AFTER the install so the new packages are found.
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display on a server: draw to files only
import matplotlib.pyplot as plt
import torch
import imageio
import gymnasium as gym
import ale_py
import stable_baselines3 as sb3
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

gym.register_envs(ale_py)  # makes the Atari game names available


# ----------------------------------------------------------------- helpers
class TimeLimitCallback(BaseCallback):
    """Stops training cleanly once MAX_TRAIN_MINUTES have passed."""

    def __init__(self, max_minutes):
        super().__init__()
        self.max_seconds = max_minutes * 60
        self.t0 = None

    def _on_training_start(self):
        self.t0 = time.time()

    def _on_step(self) -> bool:
        return (time.time() - self.t0) < self.max_seconds  # False = stop training


def check_gpu():
    print("torch", torch.__version__, "| sb3", sb3.__version__,
          "| gymnasium", gym.__version__, "| ale-py", ale_py.__version__)
    if not torch.cuda.is_available():
        # Fail fast: CPU training would be far too slow for this job.
        raise RuntimeError(
            "No GPU found. Turn on a GPU accelerator for this kernel "
            "(and make sure your Kaggle account is phone-verified)."
        )
    print("GPU:", torch.cuda.get_device_name(0))


def make_eval_env(seed, render=False):
    """Environment for testing: real (unclipped) scores, one episode = one full game."""
    env = make_atari_env(
        ENV_ID,
        n_envs=1,
        seed=seed,
        wrapper_kwargs={"terminal_on_life_loss": False, "clip_reward": False},
        env_kwargs={"render_mode": "rgb_array"} if render else None,
    )
    return VecFrameStack(env, n_stack=4)


def plot_learning_curve():
    files = glob.glob(os.path.join(LOG_DIR, "*.monitor.csv"))
    if not files:
        print("No monitor logs found; skipping plot.")
        return
    df = pd.read_csv(files[0], skiprows=1)      # first line is a header comment
    df["steps"] = df["l"].cumsum() / 4          # 'l' counts raw frames; the agent repeats each action 4x
    df["smooth"] = df["r"].rolling(window=20, min_periods=1).mean()
    plt.figure(figsize=(9, 4))
    plt.plot(df["steps"], df["r"], alpha=0.3, label="single game")
    plt.plot(df["steps"], df["smooth"], linewidth=2, label="average of last 20 games")
    plt.xlabel("Training steps")
    plt.ylabel("Score")
    plt.title("DQN on Breakout - learning curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUT_DIR, "learning_curve.png"), dpi=120, bbox_inches="tight")
    plt.close()


def record_video(model, path):
    env = make_eval_env(seed=123, render=True)
    obs = env.reset()
    frames = [env.render()]
    game_score = 0.0
    for _ in range(VIDEO_MAX_STEPS):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, _info = env.step(action)
        frames.append(env.render())
        game_score += float(reward[0])
        if done[0]:
            break
    env.close()
    imageio.mimsave(path, frames, fps=30, macro_block_size=1)
    return game_score, len(frames)


# ----------------------------------------------------------------- main
def main():
    check_gpu()

    # 1) training environment + agent
    train_env = make_atari_env(ENV_ID, n_envs=1, seed=SEED, monitor_dir=LOG_DIR)
    train_env = VecFrameStack(train_env, n_stack=4)

    model = DQN(
        "CnnPolicy",
        train_env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=100_000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        optimize_memory_usage=True,
        replay_buffer_kwargs={"handle_timeout_termination": False},
        verbose=1,
        seed=SEED,
        device="auto",
    )

    # 2) train
    callbacks = [
        CheckpointCallback(save_freq=250_000, save_path=CKPT_DIR, name_prefix="dqn_breakout_ckpt"),
        TimeLimitCallback(MAX_TRAIN_MINUTES),
    ]
    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks,
                log_interval=50, progress_bar=False)  # progress bars make messy logs
    train_minutes = (time.time() - t0) / 60
    steps_done = int(model.num_timesteps)
    print(f"Training stopped after {steps_done} steps ({train_minutes:.1f} min)")

    # 3) save the model
    model_path = os.path.join(OUT_DIR, "dqn_breakout_final")
    model.save(model_path)

    # 4) learning curve
    plot_learning_curve()

    # 5) evaluate on full games (small amount of randomness, standard for Atari)
    model.exploration_rate = 0.05
    eval_env = make_eval_env(seed=SEED + 1)
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=N_EVAL_EPISODES, deterministic=False)
    eval_env.close()
    print(f"Evaluation over {N_EVAL_EPISODES} games: {mean_r:.1f} +/- {std_r:.1f}")

    # 6) video of one full game
    video_path = os.path.join(OUT_DIR, "breakout_dqn.mp4")
    video_score, n_frames = record_video(model, video_path)
    print(f"Video saved ({n_frames} frames, game score {video_score:.0f})")

    # 7) summary file
    results = {
        "env": ENV_ID,
        "steps_trained": steps_done,
        "train_minutes": round(train_minutes, 1),
        "eval_episodes": N_EVAL_EPISODES,
        "eval_mean_score": round(float(mean_r), 2),
        "eval_std_score": round(float(std_r), 2),
        "video_game_score": video_score,
    }
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
