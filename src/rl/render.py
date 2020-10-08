"""Display agent / render video"""
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import shutil
from PIL import Image, ImageDraw, ImageFont
from gym import Env
from gym.wrappers import Monitor

from src.agent.base_agent import AbstractAgent
from src.rl.observ_transform import ObservTransformer


def render_agent(
        env_in: Env, agent: AbstractAgent,
        observ_transformer: ObservTransformer, num_steps=500, video_dir="",
        video_name=None):
    """ Watch agent do something

    Args:
        env_in: gym environment
        agent: instance of child of abstractagent (must be able to "act")
        observ_transformer: input transformer (e.g. sincos trick)
        num_steps: max episode steps
        video_dir: where to save the video, if "" no video is done.
        video_name: give a name to the video (optional)

    Returns:
        reward (float)
    """
    # setup video dir
    monitor_dir = Path(video_dir) / video_name
    if video_dir != "":
        # with video
        env = Monitor(env_in, str(monitor_dir), force=True, uid=video_name)
    else:
        # no video
        env = env_in

    # watch an untrained agent
    observ = env.reset()
    observ = observ_transformer.transform_observ(observ)
    j = 0
    sum_reward = 0
    info = []
    for j in range(num_steps):
        print("{:5}/{:5}\r".format(j, num_steps), end="")
        # render
        env.render()
        # save render info
        info.append((j, sum_reward))
        # act / step / collect rewards
        action = agent.act(observ)
        observ, reward, done, _ = env.step(action)
        observ = observ_transformer.transform_observ(observ)
        sum_reward += reward
        if done:
            break
    print("Reward {} after {:5} steps".format(sum_reward, j))

    env.close()
    env_in.close()

    if video_dir == "":
        return sum_reward

    # find video
    video_name += "_r{:.3f}".format(sum_reward)
    files = glob.glob(str(Path(monitor_dir) / "*.mp4"))
    assert len(files) == 1, "more than one/no video was created?! {}".format(
        files)
    vid_file = files[0]
    target_file = str(Path(video_dir) / "{}_raw.mp4".format(video_name))

    # ***** post process (add step, max_step, reward)

    # read video input
    video_in = cv2.VideoCapture(vid_file)
    fps = video_in.get(cv2.CAP_PROP_FPS)
    w = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # create video writer
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_out = cv2.VideoWriter(target_file, fourcc, fps, (w, h))
    counter = 0
    sum_reward_total = sum_reward
    step, sum_reward = 0, 0
    while video_in.isOpened():
        # read frame
        ret, frame = video_in.read()
        if not ret:
            break
        # read step and reward
        if counter <= j:
            step, sum_reward = info[counter]
        # # modify frame
        # # opencv has ugly fonts
        t1 = "S {:4}/{:4}".format(
            step, j + 1)
        t2 = "R {:.2f}/{:.2f}".format(
            sum_reward, sum_reward_total)

        # pillow has better fonts than opencv
        pil_frame = Image.fromarray(frame)

        draw = ImageDraw.Draw(pil_frame)
        monospace = ImageFont.truetype("_misc/cour.ttf", 32)
        t2w, t2h = draw.textsize(t2, font=monospace)
        bordered_text(draw,monospace,t1,5,5)
        bordered_text(draw,monospace,t2, pil_frame.width - t2w - 5, 5)
        # draw.text((5, 5), t1, (0, 0, 0), font=monospace)
        # draw.text((pil_frame.width - t2w - 5, 5), t2, (0, 0, 0),
        #           font=monospace)

        frame = np.array(pil_frame)

        # write video
        video_out.write(frame)
        counter += 1

    video_in.release()
    video_out.release()

    target_file_wa = str(Path(video_dir) / "{}.mp4".format(video_name))

    # make whatsapp ready
    os.system(
        "ffmpeg -i \"{}\" -c:v libx264 -c:a aac -y -preset slow "
        "-crf 5 \"{}\"".format(target_file, target_file_wa))

    # if it worked delete the non-whatsapp version
    if Path(target_file_wa).is_file():
        print("removing raw video file...")
        os.remove(str(target_file))
        print("removing gym monitor directory...")
        shutil.rmtree(str(monitor_dir))
        print("\nWrote video to {}".format(target_file_wa))
    else:
        print("Conversion failed somewhere")

    return sum_reward

def bordered_text(
        draw, font, text, x, y, color = (0,0,0), bgcolor=(255,255,255)):
    # draw.text((x - 1, y), text, font=font, fill=bgcolor)
    # draw.text((x + 1, y), text, font=font, fill=bgcolor)
    # draw.text((x, y - 1), text, font=font, fill=bgcolor)
    # draw.text((x, y + 1), text, font=font, fill=bgcolor)

    # thicker border
    draw.text((x - 1, y - 1), text, font=font, fill=bgcolor)
    draw.text((x + 1, y - 1), text, font=font, fill=bgcolor)
    draw.text((x - 1, y + 1), text, font=font, fill=bgcolor)
    draw.text((x + 1, y + 1), text, font=font, fill=bgcolor)

    # main text
    draw.text((x,y), text, font=font, fill=color)