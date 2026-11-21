"""Replay a LIBERO episode's native actions into an mp4 (CPU rendering, no GPU).

LIBERO is deterministic given (suite, task, trial) and the native action sequence, and PEAR's own LiberoEnv restores
episodes the same way, so the scene is built with LiberoEnv itself: same bddl, init state, seed and stabilization.
Frames are rotated 180 degrees as in the model's observations; agentview | wrist side by side, with a header line.

    bash replay.sh --suite libero_10 --task 0 --trial 0 --actions acts.json --out ep.mp4 \
        [--rounds rounds.json] [--success 1]
acts.json: [[7 floats], ...] native actions in execution order.
rounds.json (optional): [{"steps_before": int, "executed": int, "action": {...}}, ...] per model action (round), as in
PEAR's sim/actions_*.jsonl call log; used to label each frame with its round and action.
"""
import argparse
import json

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from pear.envs.libero.env import LiberoEnv

HEADER_PX = 64


def frame(observation, header_lines):
    views = [
        np.ascontiguousarray(observation[LiberoEnv.OBSERVATION_KEY_BY_CAMERA[camera]][::-1, ::-1], dtype=np.uint8)
        for camera in LiberoEnv.CAMERAS
    ]
    body = np.concatenate(views, axis=1)
    canvas = Image.new("RGB", (body.shape[1], body.shape[0] + HEADER_PX), "white")
    canvas.paste(Image.fromarray(body), (0, HEADER_PX))
    draw = ImageDraw.Draw(canvas)
    for i, line in enumerate(header_lines):
        draw.text((8, 4 + 20 * i), line[:170], fill="black")
    return np.asarray(canvas)


def round_of_step(rounds):
    """step index -> (round number, action) from the call log; steps outside any round map to None."""
    by_step = {}
    for r, call in enumerate(rounds, start=1):
        for s in range(call["steps_before"], call["steps_before"] + call["executed"]):
            by_step[s] = (r, call["action"])
    return by_step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--trial", type=int, required=True)
    p.add_argument("--actions", required=True)
    p.add_argument("--rounds")
    p.add_argument("--success", type=int, choices=[0, 1])
    p.add_argument("--out", required=True)
    p.add_argument("--every", type=int, default=2, help="keep every Nth native step (20 Hz sim -> 10 fps at 2)")
    args = p.parse_args()

    actions = np.asarray(json.load(open(args.actions)), dtype=np.float64).reshape(-1, LiberoEnv.ACTION_DIM)
    by_step = round_of_step(json.load(open(args.rounds))) if args.rounds else {}
    env = LiberoEnv(args.suite, args.task, args.trial)
    instruction = str(env._benchmark().get_task(args.task).language)
    max_step_n = LiberoEnv.MAX_STEP_N_BY_SUITE[args.suite]
    fps = 20 // args.every

    writer = imageio.get_writer(args.out, fps=fps, codec="libx264", quality=7, macro_block_size=8)
    try:
        observation = env._open_scene()  # reset, init state, stabilization: the episode's first observation
        writer.append_data(frame(observation, [f"{args.suite} task {args.task} trial {args.trial}: {instruction}",
                                               "initial observation"]))
        for step, native_action in enumerate(actions):
            observation, _, _, _ = env.scene.step(native_action)  # rendered step
            if step % args.every and step != len(actions) - 1:
                continue
            r = by_step.get(step)
            where = f"round {r[0]}  " if r else ""
            act = json.dumps(r[1], separators=(",", ":")) if r else ""
            writer.append_data(frame(observation, [f"{args.suite} task {args.task} trial {args.trial}: {instruction}",
                                                   f"{where}step {step + 1}/{max_step_n}  {act}"]))
        if args.success is not None:
            end = frame(observation, [f"{args.suite} task {args.task} trial {args.trial}: {instruction}",
                                      f"END after {len(actions)} steps: {'SUCCESS' if args.success else 'FAIL'}"])
            for _ in range(fps * 2):  # hold the last frame for 2 s
                writer.append_data(end)
    finally:
        writer.close()
        env._close_scene()
    print(f"wrote {args.out}: {len(actions)} native steps, {fps} fps")


if __name__ == "__main__":
    main()
