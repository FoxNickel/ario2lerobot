# 该脚本将ario中虚拟环境的单臂数据转为lerobot数据集, 将几个UR5的全转到一个lerobot数据集中

from convert_ario_sim_data_to_lerobot import process_all_episodes
from convert_ario_sim_data_to_lerobot import create_ur5_lerobot_dataset
import os
from tqdm import tqdm
from pathlib import Path
import shutil

dataset_path = "/home/huanglingyu/data/downloads/ARIO/datasets/simulation"


def main():
    repo_name = "ario_MuJoCo_UR5"
    output_path = Path("datasets/lerobot/conversion/ario_MuJoCo_UR5")
    dataset = create_ur5_lerobot_dataset(repo_name, output_path)
    
    # 转之前先清空output_path
    if output_path.exists():
        shutil.rmtree(output_path)
    print(f"清除{output_path}成功")

    tasks = [
        task
        for task in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, task))
    ]

    for task in tqdm(tasks, desc="Processing All UR5 Sim Data"):
        print(f"task: {task}")
        input_path = os.path.join(dataset_path, task+"/series-1")
        # print(f"input_path: {input_path}")
        process_all_episodes(dataset, input_path, output_path)
    return


if __name__ == "__main__":
    main()
