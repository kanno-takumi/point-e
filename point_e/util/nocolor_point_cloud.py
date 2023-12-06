import random
from dataclasses import dataclass
from typing import BinaryIO, Dict, List, Optional, Union

import numpy as np

from .ply_util import write_ply

@dataclass
class NoColorPointCloud:
    """
    An array of points sampled on a surface. Each point may have zero or more
    channel attributes.

    :param coords: an [N x 3] array of point coordinates.
    :param channels: a dict mapping names to [N] arrays of channel values.
    """

    coords: np.ndarray
    channels: Dict[str, np.ndarray] = None  # Channels is optional now

    @classmethod
    def load(cls, f: Union[str, BinaryIO]) -> "PointCloud":
        """
        Load the point cloud from a .npz file.
        """
        if isinstance(f, str):
            with open(f, "rb") as reader:
                return cls.load(reader)
        else:
            obj = np.load(f)
            keys = list(obj.keys())
            return NoColorPointCloud(
                coords=obj["coords"],
                channels={k: obj[k] for k in keys if k != "coords"} if "coords" in keys else None,
            )

    def combine(self, other: "NoColorPointCloud") -> "NoColorPointCloud":
        assert self.channels == other.channels or self.channels is None or other.channels is None
        return NoColorPointCloud(
            coords=np.concatenate([self.coords, other.coords], axis=0),
            channels=(
                {k: np.concatenate([v, other.channels[k]], axis=0) for k, v in self.channels.items()}
                if self.channels and other.channels
                else None
            ),
        )

# # 利用例:
# # 新しい PointCloud オブジェクトを作成する際に channels を指定しないことで、色情報を含まないオブジェクトを作成できます。
# point_cloud_without_color = NoColorPointCloud(coords=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))

# # もしくは、load メソッドでファイルから読み込む際に、ファイルに channels が含まれていない場合も色情報を含まないオブジェクトが作成されます。
# point_cloud_without_color_from_file = NoColorPointCloud.load("your_file.npz")
