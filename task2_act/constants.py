from __future__ import annotations

FPS = 30
IMAGE_SIZE = (200, 200)
STATE_DIM = 15
ACTION_DIM = 7
SEED = 42

STATIC_IMAGE_KEY = "observation.images.static"
GRIPPER_IMAGE_KEY = "observation.images.gripper"
STATE_KEY = "observation.state"
ACTION_KEY = "action"

TASK_DESCRIPTION = "calvin play data"

ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
STATE_NAMES = [
    "tcp_x",
    "tcp_y",
    "tcp_z",
    "tcp_roll",
    "tcp_pitch",
    "tcp_yaw",
    "gripper_width",
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "gripper_action",
]


def lerobot_features(image_size: tuple[int, int] = IMAGE_SIZE) -> dict:
    height, width = image_size
    return {
        STATIC_IMAGE_KEY: {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        GRIPPER_IMAGE_KEY: {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        STATE_KEY: {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": {"axes": STATE_NAMES},
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": {"axes": ACTION_NAMES},
        },
    }
