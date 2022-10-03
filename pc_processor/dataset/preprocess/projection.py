import numpy as np


class RangeProjection(object):
    """project 3d point cloud to 2d data with range projection"""

    def __init__(
        self,
        fov_up=3,
        fov_down=-25,
        proj_w=512,
        proj_h=64,
        fov_left=-180,
        fov_right=180,
    ):
        # check params
        assert (
            fov_up >= 0 and fov_down <= 0
        ), "require fov_up >= 0 and fov_down <= 0, while fov_up/fov_down is {}/{}".format(
            fov_up, fov_down
        )
        assert (
            fov_right >= 0 and fov_left <= 0
        ), "require fov_right >= 0 and fov_left <= 0, while fov_right/fov_left is {}/{}".format(
            fov_right, fov_left
        )

        # params of fov angeles
        self.fov_up = fov_up / 180.0 * np.pi
        self.fov_down = fov_down / 180.0 * np.pi
        self.fov_vert = abs(self.fov_up) + abs(self.fov_down)  # 0.4886

        self.fov_left = fov_left / 180.0 * np.pi
        self.fov_right = fov_right / 180.0 * np.pi
        self.fov_hori = abs(self.fov_left) + abs(self.fov_right)  # 1.5707

        # params of proj img size
        self.proj_w = proj_w
        self.proj_h = proj_h

        self.cached_data = {}

    def doProjection(self, pointcloud: np.ndarray, depth: np.ndarray = None):
        self.cached_data = {}
        # get depth of all points
        if depth is None:
            depth = np.linalg.norm(pointcloud[:, :3], 2, axis=1)
        # get point cloud components
        x = pointcloud[:, 0]
        y = pointcloud[:, 1]
        z = pointcloud[:, 2]

        # get angles of all points
        yaw = -np.arctan2(y, x)
        pitch = np.arcsin(z / depth)
        # print('horizontal range, vertical range ', self.fov_hori, self.fov_vert)
        # print('yaw.min(), yaw.max() = ', (yaw + abs(self.fov_left)).min(), (yaw + abs(self.fov_left)).max())
        # print('pitch.min(), pitch.max() = ', (1.0 - (pitch + abs(self.fov_down))).min(),
        #       (1.0 - (pitch + abs(self.fov_down))).max())

        # get projection in image coords
        proj_x = (
            yaw + abs(self.fov_left)
        ) / self.fov_hori  # normalized in [0, 1] # evenly divide in horizon
        # proj_x = 0.5 * (yaw / np.pi + 1.0) # normalized in [0, 1]
        proj_y = (
            1.0 - (pitch + abs(self.fov_down)) / self.fov_vert
        )  # normalized in [0, 1]
        # print('proj_x.min(), proj_x.max() = ', proj_x.min(), proj_x.max())
        # print('proj_y.min(), proj_y.max() =', proj_y.min(), proj_y.max())

        # scale to image size using angular resolution
        proj_x *= self.proj_w
        proj_y *= self.proj_h
        # print('proj_x.min(), proj_x.max() = ', proj_x.min(), proj_x.max())
        # print('proj_y.min(), proj_y.max() =', proj_y.min(), proj_y.max())

        # round and clamp for use as index
        proj_x = np.maximum(np.minimum(self.proj_w - 1, np.floor(proj_x)), 0).astype(
            np.int32
        )

        proj_y = np.maximum(np.minimum(self.proj_h - 1, np.floor(proj_y)), 0).astype(
            np.int32
        )

        self.cached_data["uproj_x_idx"] = proj_x.copy()
        self.cached_data["uproj_y_idx"] = proj_y.copy()
        self.cached_data["uproj_depth"] = depth.copy()

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        # order = np.argsort(depth)
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        pointcloud = pointcloud[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # get projection result
        proj_range = np.full((self.proj_h, self.proj_w), -1, dtype=np.float32)
        proj_range[proj_y, proj_x] = depth

        proj_pointcloud = np.full(
            (self.proj_h, self.proj_w, pointcloud.shape[1]), -1, dtype=np.float32
        )
        proj_pointcloud[proj_y, proj_x] = pointcloud

        proj_idx = np.full((self.proj_h, self.proj_w), -1, dtype=np.int32)
        proj_idx[proj_y, proj_x] = indices

        proj_mask = (proj_idx > 0).astype(np.int32)

        return proj_pointcloud, proj_range, proj_idx, proj_mask
