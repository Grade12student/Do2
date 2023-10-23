import cv2
import os
import numpy as np
import logging

def video_to_frames(video_path, start_frame=0, max_frames=None):
    assert os.path.exists(video_path)

    video_dir, video_filename = os.path.split(video_path)

    logging.info(
        f"Extracting frames from {video_filename}")

    capture = cv2.VideoCapture(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    assert total_frames > start_frame >= 0, "Start-Frame out of range"

    trimmed_total_frames = total_frames - start_frame
    if max_frames is not None:
        required_frames = max_frames
        end = required_frames if trimmed_total_frames > required_frames else trimmed_total_frames
    else:
        end = trimmed_total_frames

    capture.set(1, start_frame)  # Set starting frame
    frame = 0
    while_safety = 0
    frames_mem = []

    while frame < end:
        if while_safety > 500:
            break

        _, image = capture.read()

        if image is None:
            while_safety += 1
            continue

        while_safety = 0

        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames_mem.append(rgb_frame)

        frame += 1

    capture.release()

    frames_mem = np.stack(frames_mem)

    return frames_mem
