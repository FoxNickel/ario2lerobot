import pickle
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import json
from dataclasses import asdict

# 禁掉lerobot自己的进度条
from datasets import disable_progress_bars

disable_progress_bars()
from tqdm import tqdm
import shutil
from pathlib import Path
import os

from demonstrations.demo import Demo

bigym_img_width = 84
bigym_img_height = 84
bigym_img_channel = 3


def convert_one_episode(episode, dataset, task_name):
    trajectorys = episode.timesteps
    # print(f"trajectory len: {len(trajectorys)}")
    metadata = episode.metadata
    uuid = metadata.uuid
    date = metadata.date
    environment_data = metadata.environment_data
    environment_data_dict = asdict(environment_data)
    environment_data_json = json.dumps(environment_data_dict)

    package_versions = metadata.package_versions
    package_versions_json = json.dumps(package_versions)
    seed = str(metadata.seed)
    # i = 0
    for trajectory in trajectorys:
        # if i == 1:
        #     break
        # i = i + 1
        states = trajectory.observation

        actions = trajectory.info

        termination = np.array([trajectory.termination], dtype=np.bool_)

        truncation = np.array([trajectory.truncation], dtype=np.bool_)

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
        action = actions["demo_action"]
        # 若action长度小于16, 则补0到第三位(应该是15位, 补一个0就好)
        if action.shape[0] < 16:
            action = np.insert(action, 2, np.zeros(16 - action.shape[0]))
        action = action.astype(np.float32)
        # print(
        #     f"state: {state}, image: {image}, wrist_image: {wrist_image}, addition_wrist_image: {addition_wrist_image}, action: {action}"
        # )
        # print(f"termination: {termination}, truncation: {truncation}")
        # print(
        #     f"termination type: {type(termination)}, truncation type: {type(truncation)}"
        # )
        # print(
        #     f"uuid: {uuid}, date: {date}, environment_data: {environment_data}, package_versions: {package_versions}, seed: {seed}, task_name: {task_name}, environment_data_json: {environment_data_json}, package_versions_json: {package_versions_json}"
        # )
        # print(
        #     f"state type: {type(state)}, image type: {type(image)}, wrist_image type: {type(wrist_image)}, addition_wrist_image type: {type(addition_wrist_image)}, action type: {type(action)}"
        # )
        # print(
        #     f"uuid type: {type(uuid)}, date type: {type(date)}, environment_data type: {type(environment_data)}, package_versions type: {type(package_versions)}, seed type: {type(seed)}, task_name type: {type(task_name)}"
        # )
        dataset.add_frame(
            {
                "image": image,
                "wrist_image": wrist_image,
                "addition_wrist_image": addition_wrist_image,
                "state": state,
                "actions": action,
                "uuid": uuid,
                "date": date,
                "environment_data": environment_data_json,
                "package_versions": package_versions_json,
                "seed": seed,
                "termination": termination,
                "truncation": truncation,
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
            "uuid": {
                "dtype": "string",
                "shape": (1,),
                "names": ["uuid"],
            },
            "date": {
                "dtype": "string",
                "shape": (1,),
                "names": ["date"],
            },
            "environment_data": {
                "dtype": "string",
                "shape": (1,),
                "names": ["environment_data"],
            },
            "package_versions": {
                "dtype": "string",
                "shape": (1,),
                "names": ["package_versions"],
            },
            "seed": {
                "dtype": "string",
                "shape": (1,),
                "names": ["seed"],
            },
            "termination": {
                "dtype": "bool",
                "shape": (1,),
                "names": ["termination"],
            },
            "truncation": {
                "dtype": "bool",
                "shape": (1,),
                "names": ["truncation"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    return dataset


def main():
    repo_name = "bigym_v2"
    input_path = "/home/huanglingyu/.bigym/demonstrations/0.9.0"
    output_path = Path("datasets/lerobot/conversion/bigym_v2")

    if output_path.exists():
        shutil.rmtree(output_path)
        print(f"清除{output_path}成功")

    dataset = create_bigym_lerobot_dataset(repo_name, output_path)

    # len = 0
    # i = 0
    for root, dirs, files in os.walk(input_path):
        # print(f"root: {root}, dirs: {dirs}, files: {files}")
        if "absolute" in root and "pixel" in root and "hz" in root:
            # print(f"root: {root}, dirs: {dirs}, files: {files}")
            # print(f"root: {root}")
            # if len == 1:
            #     break
            # len = len + 1
            relative_path = os.path.relpath(root, input_path)
            # print(f"relative_path: {relative_path}")
            task_name = relative_path.split(os.sep)[0]
            print(f"task name: {task_name}")
            for file in tqdm(files, desc="Processing files"):
                # print(f"task: {task_name}, file: {file}")
                # 读tensor文件, 然后转, 得看下episode格式
                file_path = os.path.join(root, file)
                try:
                    # if i == 1:
                    #     break
                    # i = i + 1
                    demo = Demo.from_safetensors(file_path)  # 尝试加载 safetensors 文件
                    # print(f"demo: {demo}")
                    convert_one_episode(demo, dataset, task_name)
                except Exception as e:
                    print(f"⚠ 跳过损坏文件: {file}, 错误: {e}")
                # print(f"file_path: {file_path}")
    # print(f"len: {len}")


if __name__ == "__main__":
    main()
    # input_path = "/home/huanglingyu/data/robobase/data"
    # show_pkl_info(input_path)
