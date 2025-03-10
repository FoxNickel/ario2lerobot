import pickle
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 禁掉lerobot自己的进度条
from datasets import disable_progress_bars

disable_progress_bars()
from tools.config import bigym_img_height, bigym_img_width, bigym_img_channel
from tqdm import tqdm
import shutil
from pathlib import Path


def convert_one_episode(episode, dataset):
    # 造的数据, 一个episode只有一个trajectory
    # TODO: trajectory_num = len(episode)
    trajectory_num = 1
    state_zeros = np.zeros(8)
    for i in range(trajectory_num):
        state = episode[i][0]
        actions = episode[i][1]
        # print(f"actions: {actions}")
        image = np.transpose(state["rgb_head"], (1, 2, 0)) / 255.0
        wrist_image = np.transpose(state["rgb_left_wrist"], (1, 2, 0)) / 255.0
        addition_wrist_image = np.transpose(state["rgb_right_wrist"], (1, 2, 0)) / 255.0
        action = actions["demo_action"]
        # print(f"action: {action}")
        # TODO: 补0不应该补到最后
        if action.shape[0] < 16:
            action = np.concatenate([action, np.zeros(16 - action.shape[0])])
        dataset.add_frame(
            {
                "image": image,
                "wrist_image": wrist_image,
                "addition_wrist_image": addition_wrist_image,
                "state": state_zeros,
                "actions": action,
                "task": "move_plate",
            }
        )
    dataset.save_episode()


def create_bigym_lerobot_dataset(repo_name, output_path):
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="bigym",
        fps=30,
        root=output_path,
        features={
            "image": {
                "dtype": "image",
                "shape": (bigym_img_height, bigym_img_width, bigym_img_channel),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (bigym_img_height, bigym_img_width, bigym_img_channel),
                "names": ["height", "width", "channel"],
            },
            "addition_wrist_image": {
                "dtype": "image",
                "shape": (bigym_img_height, bigym_img_width, bigym_img_channel),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float64",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float64",
                "shape": (16,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    return dataset


def main():
    repo_name = "bigym"
    output_path = Path("datasets/lerobot/conversion/bigym")

    if output_path.exists():
        shutil.rmtree(output_path)
    print(f"清除{output_path}成功")

    dataset = create_bigym_lerobot_dataset(repo_name, output_path)

    # TODO: 等bigym数据生成完之后再改
    with open("/home/huanglingyu/data/robobase/move_plate_copy.pkl", "rb") as f:
        episodes = pickle.load(f)
        for episode in tqdm(episodes, desc="Processing episodes"):
            convert_one_episode(episode, dataset)


if __name__ == "__main__":
    main()
