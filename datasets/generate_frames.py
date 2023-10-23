import os
import cv2
import numpy as np
import logging

def video_to_frames(video_path, start_frame=0, max_frames=None):
    assert os.path.exists(video_path)

    video_dir, video_filename = os.path.split(video_path)

    logging.info(f"Loading frames from folder {video_filename}")

    frames_mem = []
    frame_count = 0

    for frame_file in sorted(os.listdir(video_path)):
        if max_frames is not None and frame_count >= max_frames:
            break

        frame_path = os.path.join(video_path, frame_file)

        frame = cv2.imread(frame_path)
        if frame is None:
            logging.warning(f"Unable to read frame {frame_path}")
            continue

        frames_mem.append(frame)
        frame_count += 1

    frames_mem = np.stack(frames_mem)

    return frames_mem
