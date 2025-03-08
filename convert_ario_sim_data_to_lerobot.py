from pathlib import Path
import shutil
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
import os
from tqdm import tqdm
import yaml
from tools.read_episode_data import get_sim_episode_data
# 禁掉lerobot自己的进度条
from datasets import disable_progress_bars
disable_progress_bars()

# 该脚本将ario中虚拟环境的单臂数据转为lerobot数据集, 一次只能转一个文件夹下面的数据


def convert_one_episode(instruction, episode_path, dataset):
    (
        joint_position,
        eef_position,
        image_y_frames,
        image_z_frames,
        right_wrist_image_frames,
    ) = get_sim_episode_data(episode_path)

    # print(
    #     f"joint_position shape: {joint_position.shape}, image_high_frames: {len(image_high_frames)}, left_wrist_image_frames: {len(left_wrist_image_frames)}, right_wrist_image_frames: {len(right_wrist_image_frames)}"
    # )
    state_zeros = np.zeros(8)
    # for i in tqdm(range(joint_position.shape[0]), desc=f"Processing {task} {episode}"):
    for i in range(joint_position.shape[0]):
        dataset.add_frame(
            {
                "image": image_z_frames[i],
                "wrist_image": right_wrist_image_frames[i],
                "state": state_zeros,
                "actions": eef_position[i],
                "joint_actions": joint_position[i],
                "task": instruction,
            }
        )
    dataset.save_episode()


def parse_instruction(yaml_file_path):
    with open(yaml_file_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
        instruction = data.get("instruction_EN")
        return instruction


def process_all_episodes(dataset, input_path, output_path):    
    tasks = [
        task
        for task in os.listdir(input_path)
        if os.path.isdir(os.path.join(input_path, task))
    ]
    tasks.sort(key=lambda x: int(x.split("-")[-1]))  # 按序号排序

    for task in tqdm(tasks, desc="Processing tasks"):
        task_path = os.path.join(input_path, task)
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
        # print(f"task: {task}, episodes: {episodes}")

        for episode in tqdm(episodes, desc=f"Processing episodes in {task}"):
            episode_path = os.path.join(task_path, episode)
            print(f"Processing episode: {episode}")
            # print(f"Episode path: {episode_path}")
            convert_one_episode(instruction, episode_path, dataset)


def process_one_episode():
    episode_path = "/home/huanglingyu/data/downloads/ARIO/datasets/collection-Songling/series-1/task-pick_water_50_4_7th/episode-1"
    convert_one_episode("instruction task pick", episode_path, create_ur5_lerobot_dataset())


def create_ur5_lerobot_dataset(repo_name, output_path):
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="ur5",
        fps=30,
        root=output_path,
        features={
            "image": {
                "dtype": "image",
                "shape": (120, 160, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
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
                "shape": (7,),
                "names": ["actions"],
            },
            "joint_actions": {
                "dtype": "float64",
                "shape": (7,),
                "names": ["joint_actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    return dataset


def main():
    # process_one_episode()
    repo_name = "ario_MuJoCo_UR5"
    input_path = "/home/huanglingyu/data/downloads/ARIO/datasets/collection_PickPlace/series-1"
    output_path = Path("datasets/lerobot/conversion/ario_MuJoCo_UR5_pick_place")
    dataset = create_ur5_lerobot_dataset(repo_name, output_path)
    process_all_episodes(dataset, input_path, output_path)


if __name__ == "__main__":
    tyro.cli(main)
