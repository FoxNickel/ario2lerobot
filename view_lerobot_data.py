from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)

# Let's take this one for this example
repo_id = "ario"
# We can have a look and fetch its metadata to know more about it:
ds_meta = LeRobotDatasetMetadata(repo_id, root="/home/huanglingyu/data/downloads/ARIO/datasets/lerobot/conversion")

# By instantiating just this class, you can quickly access useful information about the content and the
# structure of the dataset without downloading the actual data yet (only metadata files — which are
# lightweight).
print(f"Total number of episodes: {ds_meta.total_episodes}")
print(
    f"Average number of frames per episode: {ds_meta.total_frames / ds_meta.total_episodes:.3f}"
)
print(f"Frames per second used during data collection: {ds_meta.fps}")
print(f"Robot type: {ds_meta.robot_type}")
print(f"keys to access images from cameras: {ds_meta.camera_keys=}\n")

print("Tasks:")
print(ds_meta.tasks)
print("Features:")
print(ds_meta.features)

# You can also get a short summary by simply printing the object:
print(ds_meta)

# You can then load the actual dataset from the hub.
# Either load any subset of episodes:
dataset = LeRobotDataset(repo_id, root="/home/huanglingyu/data/downloads/ARIO/datasets/lerobot/conversion", episodes=[0])

# And see how many frames you have:
print(f"Selected episodes: {dataset.episodes}")
print(f"Number of episodes selected: {dataset.num_episodes}")
print(f"Number of frames selected: {dataset.num_frames}")
print(f"dataset[0]: {dataset[0]['actions']}")
