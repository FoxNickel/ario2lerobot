from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
import cv2
import numpy as np

def main():
    # Let's take this one for this example
    repo_id = "ario_real_agilex_aloha"
    output_path = "/home/huanglingyu/data/downloads/ARIO/datasets/lerobot/conversion/ario_real_agilex_aloha"
    # We can have a look and fetch its metadata to know more about it:
    ds_meta = LeRobotDatasetMetadata(repo_id, root=output_path)

    # By instantiating just this class, you can quickly access useful information about the content and the
    # structure of the dataset without downloading the actual data yet (only metadata files — which are
    # lightweight).
    # print(f"Total number of episodes: {ds_meta.total_episodes}")
    # print(
    #     f"Average number of frames per episode: {ds_meta.total_frames / ds_meta.total_episodes:.3f}"
    # )
    # print(f"Frames per second used during data collection: {ds_meta.fps}")
    # print(f"Robot type: {ds_meta.robot_type}")
    # print(f"keys to access images from cameras: {ds_meta.camera_keys=}\n")

    # print("Tasks:")
    # print(ds_meta.tasks)
    # print("Features:")
    # print(ds_meta.features)

    # You can also get a short summary by simply printing the object:
    print(ds_meta)

    # You can then load the actual dataset from the hub.
    # Either load any subset of episodes:
    dataset = LeRobotDataset(repo_id, root=output_path, episodes=[0])

    # And see how many frames you have:
    # print(f"Selected episodes: {dataset.episodes}")
    # print(f"Number of episodes selected: {dataset.num_episodes}")
    # print(f"Number of frames selected: {dataset.num_frames}")
    print(f"dataset[0]actions: {dataset[0]['actions']}")
    # print(f"dataset[0]image: {dataset[0]['image'].shape}")
    save_image(dataset, "image")
    save_image(dataset, "wrist_image")
    save_image(dataset, "addition_wrist_image")

def save_image(dataset, img_name):
    img_data = dataset[0][img_name]
    cv_image_data = np.array(img_data)
    cv_image_data = np.transpose(cv_image_data, (1, 2, 0))
    cv_image_data = (cv_image_data * 255).astype(np.uint8)
    # 保存图像到文件
    output_image_path = f"outputs/{img_name}.jpg"
    success = cv2.imwrite(output_image_path, cv_image_data)
    if success:
        print(f"Image saved to {output_image_path}")
    else:
        print(f"Failed to save image to {output_image_path}")

if __name__ == "__main__":
    main()