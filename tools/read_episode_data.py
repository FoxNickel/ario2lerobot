import tyro
import os
import numpy as np
import cv2

# 该脚本作用是读取一个episode的数据


# 从txt文件读取joint数据, 左臂7维，右臂7维，共14维, 7维包含6个arm的joint和1个gripper的joint, gripper取了最后一维
def read_double_arm_joint_position(episode_path):
    # 获取左右臂各7维的数据
    left_joint_position = parse_single_arm_joint_position("left", episode_path)
    right_joint_position = parse_single_arm_joint_position("right", episode_path)
    # 将左右臂的joint数据拼接成一个14维的数据
    joint_position = np.column_stack((left_joint_position, right_joint_position))
    return joint_position


# 读单臂joint数据
def read_single_arm_joint_position(episode_path):
    right_joint_position = parse_single_arm_joint_position("right", episode_path)
    return right_joint_position


# 解析单臂的joint数据(6个手臂joint position + 1个gripper闭合状态)
def parse_single_arm_joint_position(type, episode_path):
    master_files = []
    # 遍历 episode_path 文件夹
    for root, dirs, files in os.walk(episode_path):
        for file in files:
            # 检查文件名是否包含'left' or 'right' and 'joint' 且后缀为 '.txt'
            if type in file and "master" in file and file.endswith(".txt"):
                master_files.append(os.path.join(root, file))
    master_files = sorted(master_files)

    # 初始化一个空列表来存储每个文件的joint数据
    joint_positions = []
    eef_positions = None

    # 读取每个joint文件的数据
    for master_file in master_files:
        if "joint" in master_file:
            left_arm_data = np.loadtxt(
                master_file, usecols=1
            )  # 只读取第二列数据（joint_position）
            joint_positions.append(left_arm_data)
        elif "gripper" in master_file:
            left_eef_data = np.loadtxt(
                master_file, usecols=7
            )  # 只读取第八列数据（eef_position）
            eef_positions = left_eef_data

    # 将所有joint数据拼接成一个二维数组
    joint_positions = np.column_stack(joint_positions)

    # 将joint数据和eef数据拼接成一个二维数组
    if eef_positions is not None:
        combined_data = np.column_stack((joint_positions, eef_positions))
    else:
        print("No gripper data found.")

    # print("Combined data shape:", combined_data.shape)
    # print("Combined data:", combined_data[0])
    return combined_data


# 解析eef的7维数据(x, y, z, pitch, roll, yaw, gripper闭合), 从gripper.txt文件里面读的
def parse_eef_position(episode_path):
    master_files = []
    # 遍历 episode_path 文件夹
    for root, dirs, files in os.walk(episode_path):
        for file in files:
            # 检查文件名是否包含'gripper' 且后缀为 '.txt'
            if "gripper" in file and file.endswith(".txt"):
                master_files.append(os.path.join(root, file))
    master_files = sorted(master_files)
    # print(master_files)

    # 初始化一个空列表来存储每个文件的eef数据
    eef_positions = None

    # 读取每个joint文件的数据
    for master_file in master_files:
        eef_positions = np.loadtxt(master_file, usecols=[1, 2, 3, 4, 5, 6, 7])

    # print("eef_positions shape:", eef_positions.shape)
    return eef_positions


# 解析双臂eef的各7维数据(x, y, z, pitch, roll, yaw, gripper闭合), 从gripper.txt文件里面读的
def parse_double_arm_eef_position(episode_path):
    master_files = []
    # 遍历 episode_path 文件夹
    for root, dirs, files in os.walk(episode_path):
        for file in files:
            # 检查文件名是否包含'gripper' 且后缀为 '.txt'
            if "gripper" in file and file.endswith(".txt"):
                master_files.append(os.path.join(root, file))
    master_files = sorted(master_files)
    # print(master_files)

    # 初始化一个空列表来存储每个文件的eef数据
    left_eef_positions = None

    # 读取每个joint文件的数据
    for master_file in master_files:
        if "left" in master_file:
            left_eef_positions = np.loadtxt(master_file, usecols=[1, 2, 3, 4, 5, 6, 7])
        if "right" in master_file:
            right_eef_positions = np.loadtxt(master_file, usecols=[1, 2, 3, 4, 5, 6, 7])

    # print("left_eef_positions shape:", left_eef_positions.shape)
    # print("right_eef_positions shape:", right_eef_positions.shape)
    combined_eef_positions = np.column_stack((left_eef_positions, right_eef_positions))
    # print("combined_eef_positions shape:", combined_eef_positions.shape)
    return combined_eef_positions


# 读取视频帧
def read_video_frames(video_path, width=None, height=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if width is not None and height is not None:
            frame = cv2.resize(frame, (width, height))
        frames.append(frame)
    cap.release()
    return frames


# 读取一个来自实机的episode的数据
def get_real_episode_data(episode_path):
    joint_position = read_double_arm_joint_position(episode_path)
    eef_position = parse_double_arm_eef_position(episode_path)

    image_high_frames = read_video_frames(
        episode_path + "/rgbd-cam_high/rgb.mp4", width=160, height=120
    )
    left_wrist_image_frames = read_video_frames(
        episode_path + "/rgbd-cam_left_wrist/rgb.mp4", width=160, height=120
    )
    right_wrist_image_frames = read_video_frames(
        episode_path + "/rgbd-cam_right_wrist/rgb.mp4", width=160, height=120
    )

    return (
        joint_position,
        eef_position,
        image_high_frames,
        left_wrist_image_frames,
        right_wrist_image_frames,
    )


# 读取一个来自sim的episode的数据
def get_sim_episode_data(episode_path):
    joint_position = read_single_arm_joint_position(episode_path)
    eef_position = parse_eef_position(episode_path)

    image_y_frames = read_video_frames(
        episode_path + "/rgbd-1/rgb.mp4", width=160, height=120
    )
    image_z_frames = read_video_frames(
        episode_path + "/rgbd-2/rgb.mp4", width=160, height=120
    )
    right_wrist_image_frames = read_video_frames(
        episode_path + "/rgbd-3/rgb.mp4", width=160, height=120
    )

    return (
        joint_position,
        eef_position,
        image_y_frames,
        image_z_frames,
        right_wrist_image_frames,
    )


# test
def main():
    # episode_path = "/home/huanglingyu/data/downloads/ARIO/datasets/collection_Grasp/series-1/task-1/episode-1"
    # (
    #     joint_position,
    #     image_z_frames,
    #     left_wrist_image_frames,
    #     right_wrist_image_frames,
    # ) = get_real_episode_data(episode_path)
    # print(f"Joint position shape: {joint_position.shape}")
    # print(f"Number of RGB frames: {len(image_z_frames)}")
    # print(f"Number of left wrist image frames: {len(left_wrist_image_frames)}")
    # print(f"Number of right wrist image frames: {len(right_wrist_image_frames)}")
    parse_double_arm_eef_position(
        "/home/huanglingyu/data/downloads/ARIO/datasets/collection-Songling/series-1/task-pick_U_driver_20_4_7th_PCL/episode-1"
    )


if __name__ == "__main__":
    tyro.cli(main)
