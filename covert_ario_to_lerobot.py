from pathlib import Path
import shutil
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
import os
from tqdm import tqdm
from tools.read_episode_data import get_episode_data


REPO_NAME = "ario"


def convert_one_episode(task, episode, episode_path, dataset):
    (
        joint_position,
        image_high_frames,
        left_wrist_image_frames,
        right_wrist_image_frames,
    ) = get_episode_data(episode_path)

    # print(
    #     f"joint_position shape: {joint_position.shape}, image_high_frames: {len(image_high_frames)}, left_wrist_image_frames: {len(left_wrist_image_frames)}, right_wrist_image_frames: {len(right_wrist_image_frames)}"
    # )
    state_zeros = np.zeros(8)
    task_name = task + "_" + episode
    # for i in tqdm(range(joint_position.shape[0]), desc=f"Processing {task} {episode}"):
    for i in range(joint_position.shape[0]):
        dataset.add_frame(
            {
                "image_high": image_high_frames[i],
                "left_wrist_image": left_wrist_image_frames[i],
                "right_wrist_image": right_wrist_image_frames[i],
                "state": state_zeros,
                "actions": joint_position[i],
                "task": task_name
            }
        )
    dataset.save_episode()


def process_all_episodes():
    origin_data_root_dir = (
        "/home/huanglingyu/data/downloads/ARIO/datasets/collection-Songling copy/series-1"
    )
    tasks = [
        task
        for task in os.listdir(origin_data_root_dir)
        if os.path.isdir(os.path.join(origin_data_root_dir, task))
    ]
    dataset = create_lerobot_dataset()

    for task in tqdm(tasks, desc="Processing tasks"):
        task_path = os.path.join(origin_data_root_dir, task)
        # print(f"Processing task: {task}")

        episodes = [
            episode
            for episode in os.listdir(task_path)
            if os.path.isdir(os.path.join(task_path, episode))
        ]
        episodes.sort(key=lambda x: int(x.split("-")[-1]))  # 按序号排序

        for episode in tqdm(episodes, desc=f"Processing episodes in {task}"):
            episode_path = os.path.join(task_path, episode)
            # print(f"Processing episode: {episode}")
            # print(f"Episode path: {episode_path}")
            convert_one_episode(task, episode, episode_path, dataset)


def process_one_episode():
    episode_path = "/home/huanglingyu/data/downloads/ARIO/datasets/collection-Songling/series-1/task-pick_water_50_4_7th/episode-1"
    convert_one_episode("task-pick_water_50_4_7th", "episode-1", episode_path, create_lerobot_dataset())


def create_lerobot_dataset():
    output_path = Path("datasets/lerobot/conversion")
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="agilex-aloha",
        fps=30,
        root=output_path,
        features={
            "image_high": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "left_wrist_image": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "right_wrist_image": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float64",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float64",
                "shape": (14,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    return dataset


def main():
    # process_one_episode()
    process_all_episodes()


if __name__ == "__main__":
    tyro.cli(main)
