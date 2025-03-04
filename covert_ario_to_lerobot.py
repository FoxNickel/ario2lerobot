from pathlib import Path
import shutil
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
import os
from tqdm import tqdm
import yaml
from tools.read_episode_data import get_episode_data


REPO_NAME = "ario"


def convert_one_episode(instruction, episode_path, dataset):
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
    # for i in tqdm(range(joint_position.shape[0]), desc=f"Processing {task} {episode}"):
    for i in range(joint_position.shape[0]):
        dataset.add_frame(
            {
                "image_high": image_high_frames[i],
                "left_wrist_image": left_wrist_image_frames[i],
                "right_wrist_image": right_wrist_image_frames[i],
                "state": state_zeros,
                "actions": joint_position[i],
                "task": instruction,
            }
        )
    dataset.save_episode()


def parse_instruction(yaml_file_path):
    with open(yaml_file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
        instruction = data.get('instruction_EN')
        return instruction


def process_all_episodes():
    origin_data_root_dir = (
        "/home/huanglingyu/data/downloads/ARIO/datasets/collection-Songling/series-1"
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
        description = [
            description
            for description in os.listdir(task_path)
            if "description" in description
        ]
        instruction = parse_instruction(os.path.join(task_path, description[0]))
        # print(f"task: {task}, instruction: {instruction}")
        
        episodes.sort(key=lambda x: int(x.split("-")[-1]))  # 按序号排序

        for episode in tqdm(episodes, desc=f"Processing episodes in {task}"):
            episode_path = os.path.join(task_path, episode)
            # print(f"Processing episode: {episode}")
            # print(f"Episode path: {episode_path}")
            convert_one_episode(instruction, episode_path, dataset)


def process_one_episode():
    episode_path = "/home/huanglingyu/data/downloads/ARIO/datasets/collection-Songling/series-1/task-pick_water_50_4_7th/episode-1"
    convert_one_episode(
        "instruction task pick", episode_path, create_lerobot_dataset()
    )


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
                "shape": (120, 160, 3),
                "names": ["height", "width", "channel"],
            },
            "left_wrist_image": {
                "dtype": "image",
                "shape": (120, 160, 3),
                "names": ["height", "width", "channel"],
            },
            "right_wrist_image": {
                "dtype": "image",
                "shape": (120, 160, 3),
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
