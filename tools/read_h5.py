import h5py
import os
import cv2
import numpy as np
from collections import defaultdict


class ReadH5Files:
    def __init__(self, robot_infor):
        self.robot_info = robot_infor
        

    def decoder_image(self, camera_rgb_images, camera_depth_images):
        if type(camera_rgb_images[0]) is np.uint8:
            rgb = cv2.imdecode(camera_rgb_images, cv2.IMREAD_COLOR)
            if camera_depth_images is not None:
                depth_array = np.frombuffer(camera_depth_images, dtype=np.uint8)
                depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
            else:
                depth = np.asarray([])
            return rgb, depth
        else:
            rgb_images = []
            depth_images = []
            for idx, camera_rgb_image in enumerate(camera_rgb_images):
                rgb = cv2.imdecode(camera_rgb_image, cv2.IMREAD_COLOR)
                if camera_depth_images is not None:
                    depth_array = np.frombuffer(
                        camera_depth_images[idx], dtype=np.uint8
                    )
                    depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
                else:
                    depth = np.asarray([])
                rgb_images.append(rgb)
                depth_images.append(depth)
            rgb_images = np.asarray(rgb_images)
            depth_images = np.asarray(depth_images)
            return rgb_images, depth_images

    def execute(self, file_path, save_dir, camera_frame=None, control_frame=None):
        with h5py.File(file_path, "r") as root:
            is_sim = root.attrs["sim"]
            is_compress = root.attrs["compress"]
            print("is_compress:", is_compress)
            # select camera frame id
            image_dict = defaultdict(dict)
            for cam_name in self.camera_names:
                if is_compress:
                    if camera_frame is not None:
                        # decode_rgb, decode_depth = self.decoder_image(
                        #     camera_rgb_images=root['observations'][self.camera_sensors[0]][cam_name][camera_frame],
                        #     camera_depth_images=root['observations'][self.camera_sensors[1]][cam_name][camera_frame])
                        decode_rgb, decode_depth = self.decoder_image(
                            camera_rgb_images=root["observations"][
                                self.camera_sensors[0]
                            ][cam_name][camera_frame],
                            camera_depth_images=None,
                        )

                        for i in range(len(decode_rgb)):
                            cv2.imwrite(
                                os.path.join(save_dir, f"{cam_name}_{i}.jpg"),
                                decode_rgb[i],
                            )

                    else:
                        # decode_rgb, decode_depth = self.decoder_image(
                        #     camera_rgb_images=root['observations'][self.camera_sensors[0]][cam_name][:],
                        #     camera_depth_images=root['observations'][self.camera_sensors[1]][cam_name][:])
                        decode_rgb, decode_depth = self.decoder_image(
                            camera_rgb_images=root["observations"][
                                self.camera_sensors[0]
                            ][cam_name][:],
                            camera_depth_images=None,
                        )
                        for i in range(len(decode_rgb)):
                            cv2.imwrite(
                                os.path.join(save_dir, f"{cam_name}_{i}.jpg"),
                                decode_rgb[i],
                            )

                    image_dict[self.camera_sensors[0]][cam_name] = decode_rgb
                    # image_dict[self.camera_sensors[1]][cam_name] = decode_depth

                else:
                    if camera_frame:
                        image_dict[self.camera_sensors[0]][cam_name] = root[
                            "observations"
                        ][self.camera_sensors[0]][cam_name][camera_frame]
                        # image_dict[self.camera_sensors[1]][cam_name] = root[
                        #     'observations'][self.camera_sensors[1]][cam_name][camera_frame]
                    else:
                        image_dict[self.camera_sensors[0]][cam_name] = root[
                            "observations"
                        ][self.camera_sensors[0]][cam_name][:]
                        # image_dict[self.camera_sensors[1]][cam_name] = root[
                        #    'observations'][self.camera_sensors[1]][cam_name][:]

            control_dict = defaultdict(dict)
            for arm_name in self.arms:
                for control in self.robot_infor:
                    if control_frame:
                        control_dict[arm_name][control] = root[arm_name][control][
                            control_frame
                        ]
                    else:
                        control_dict[arm_name][control] = root[arm_name][control][:]
            # print('infor_dict:',infor_dict)
            base_dict = defaultdict(dict)
        # print('control_dict[puppet]:',control_dict['master']['joint_position_left'][0:1])
        return image_dict, control_dict, base_dict, is_sim, is_compress


def read_h5(file_path):
    with h5py.File(file_path, "r") as root:
        print("root keys:", root.keys())
        action = root["action"]
        print("action shape:", action.shape)
        print("action:", action[0:1])
        base_action = root["base_action"]
        print("base_action shape:", base_action.shape)
        print("base_action:", base_action[0:1])
        observation = root["observations"]
        print("observation keys:", observation.keys())
        effort = observation["effort"]
        print("effort shape:", effort.shape)
        print("effort:", effort[0:1])
        images = observation["images"]
        print("image keys:", images.keys())
        images_depth = observation["images_depth"]
        print("image_depth keys:", images_depth.keys())
        qpos = observation["qpos"]
        print("qpos shape:", qpos.shape)
        print("qpos:", qpos[0:1])
        qvel = observation["qvel"]
        print("qvel shape:", qvel.shape)
        print("qvel:", qvel[0:1])

if __name__ == "__main__":
    robot_info = {
        "camera_names": ["camera_top"],
        "camera_sensors": ["rgb_images"],
        "arms": ["puppet", "master"],
        "controls": ["joint_position"],
    }

    file_path = "/home/huanglingyu/data/downloads/ARIO/datasets/pick U driver_20_4_7th_PCL/episode_0.hdf5"

    save_dir = "./test_image/"

    read_h5(file_path)

    # read_h5files = ReadH5Files(robot_infor)
    # image_dict, control_dict, base_dict, is_sim, is_compress = read_h5files.execute(
    #     file_path, save_dir
    # )
    # print(f'control_dict:{control_dict["master"]["joint_position"].shape}')
