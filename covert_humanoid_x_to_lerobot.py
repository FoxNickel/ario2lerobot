"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""
from pathlib import Path
import shutil
import os
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro
from tqdm import tqdm
import multiprocessing
from datasets import disable_progress_bars
disable_progress_bars()

REPO_NAME = "wangjun/Humanoid-X"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    "aist",
    "animation",
    "charades",
    "dance",
    "EgoBody",
    "fitness",  
    "game_motion",  
    "GRAB",  
    "HAA500",  
    "humanml",  
    "humman",  
    "idea400",  
    "kinetics700",  
    "kungfu",  
    "music",  
    "perform",  
    "youtube"
]  # For simplicity we will combine multiple Libero datasets into one training dataset


def extract_description_from_file(txt_file):
    with open(txt_file, 'r') as f:
        content = f.read().strip()  # 读取文件并去掉前后空白字符
    
    # 提取井号前的内容
    description = content.split('#')[0].strip()  # 取井号前的部分，并去掉两边的空白字符
    return description

def main(data_dir: str, *, push_to_hub: bool = False):

    # Clean up any existing dataset in the output directory
    # output_path = LEROBOT_HOME / REPO_NAME
    output_path = Path("datasets/Humanoid-X/train")
    if output_path.exists():
        shutil.rmtree(output_path)

    with open('datasets/Humanoid-X/train.txt', 'r') as f:
        file_paths = [line.strip() for line in f.readlines()]
    file_paths.sort()
    
    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`

    image_zeros = np.zeros((256, 256, 3))
    wrist_image_zeros = np.zeros((256, 256, 3))
    state_zeros = np.zeros(8)

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="H1-2",
        fps=10,
        root=output_path,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (34,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    valid_files_num =0
    for file_path in tqdm(file_paths):

        npy_file_path = os.path.join(data_dir, file_path + '_sample.npy')
        txt_file = npy_file_path.replace('_sample.npy', '.txt').replace('humanoid_action', 'texts')
        if not os.path.exists(npy_file_path):
            continue
        if not os.path.exists(txt_file):
            continue
        
        description = extract_description_from_file(txt_file)
        # print(f"npy_file_path: {npy_file_path}")
        try:
            data = np.load(npy_file_path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading {npy_file_path}: {e}")
            continue

        data_dict = data.item()
        dof_pos = data_dict['dof_pos']  # 形状为 (frame, 27)
        root_trans = data_dict['root_trans']  # 形状为 (frame, 3)
        root_rot = data_dict['root_rot']  # 形状为 (frame, 4)
        
        for i in range(dof_pos.shape[0]):
            combined_frame = np.concatenate((dof_pos[i], root_trans[i], root_rot[i]))
            dataset.add_frame(
                {
                    "image": image_zeros, # <class 'numpy.ndarray'>
                    "wrist_image": wrist_image_zeros, # <class 'numpy.ndarray'>
                    "state": state_zeros, # <class 'numpy.ndarray'>
                    "actions": combined_frame,
                }
            )
        dataset.save_episode(task=description)
        valid_files_num += 1

    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)
    print(f"Processed {valid_files_num} files")

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

if __name__ == "__main__":
    tyro.cli(main)
