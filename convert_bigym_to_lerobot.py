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
import os


def convert_one_episode(episode, dataset, task_name):
    trajectory_num = len(episode)
    for i in range(trajectory_num):
        states = episode[i][0]
        proprioception = states["proprioception"]
        # 分为前30维和后30维, 小于60时补零补到前30维里面的倒数第二维和后30里面的倒数第二维
        if proprioception.shape[0] < 60:
            proprioception = np.insert(proprioception, -1, 0)
            proprioception = np.insert(proprioception, 28, 0)
        proprioception_floating_base = states["proprioception_floating_base"]
        # 补0补到倒数第二位, z的位置
        if proprioception_floating_base.shape[0] < 4:
            proprioception_floating_base = np.insert(
                proprioception_floating_base,
                -1,
                np.zeros(4 - proprioception_floating_base.shape[0]),
            )
        proprioception_grippers = states["proprioception_grippers"]
        state = np.concatenate(
            [proprioception, proprioception_floating_base, proprioception_grippers]
        ).astype(np.float32)
        image = np.transpose(states["rgb_head"], (1, 2, 0)) / 255.0
        wrist_image = np.transpose(states["rgb_left_wrist"], (1, 2, 0)) / 255.0
        addition_wrist_image = (
            np.transpose(states["rgb_right_wrist"], (1, 2, 0)) / 255.0
        )

        # action在最后一个字段里面, 这里可能不止两个字段, 所以用-1取
        actions = episode[i][-1]
        action = actions["demo_action"]
        # 若action长度小于16, 则补0到第三位(应该是15位, 补一个0就好)
        if action.shape[0] < 16:
            action = np.insert(action, 2, np.zeros(16 - action.shape[0]))
        action = action.astype(np.float32)
        dataset.add_frame(
            {
                "image": image,
                "wrist_image": wrist_image,
                "addition_wrist_image": addition_wrist_image,
                "state": state,
                "actions": action,
                "task": task_name,
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
                "dtype": "float32",
                "shape": (66,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (16,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    return dataset


def show_episode_info(episode):
    states = episode[0][0]
    proprioception = states["proprioception"]
    proprioception_floating_base = states["proprioception_floating_base"]
    proprioception_grippers = states["proprioception_grippers"]

    actions = episode[0][-1]
    action = actions["demo_action"]
    print(
        f"proprioception.shape: {len(proprioception)}, proprioception_floating_base.shape: {len(proprioception_floating_base)}, proprioception_grippers.shape: {len(proprioception_grippers)}, action.shape: {len(action)}"
    )


def show_pkl_info(input_path):
    for root, dirs, files in os.walk(input_path):
        # print(f"root: {root}, dirs: {dirs}, files: {files}")
        for file in tqdm(files, desc="Processing files"):
            if file.endswith(".pkl"):
                task_name = file.split(".")[0]
                print(f"Processing task: {task_name}")
                file_path = os.path.join(root, file)
                # print(f"file_path: {file_path}")
                with open(file_path, "rb") as f:
                    episodes = pickle.load(f)
                    show_episode_info(episodes[0])


def main():
    repo_name = "bigym"
    input_path = "/home/huanglingyu/data/robobase/data"
    output_path = Path("datasets/lerobot/conversion/bigym")

    if output_path.exists():
        shutil.rmtree(output_path)
    print(f"清除{output_path}成功")

    dataset = create_bigym_lerobot_dataset(repo_name, output_path)

    for root, dirs, files in os.walk(input_path):
        # print(f"root: {root}, dirs: {dirs}, files: {files}")
        for file in tqdm(files, desc="Processing files"):
            if file.endswith(".pkl"):
                task_name = file.split(".")[0]
                print(f"Processing task: {task_name}")
                file_path = os.path.join(root, file)
                # print(f"file_path: {file_path}")
                with open(file_path, "rb") as f:
                    episodes = pickle.load(f)
                    for episode in tqdm(
                        episodes, desc=f"Processing episodes in task {task_name}"
                    ):
                        convert_one_episode(episode, dataset, task_name)


if __name__ == "__main__":
    main()
    # input_path = "/home/huanglingyu/data/robobase/data"
    # show_pkl_info(input_path)
