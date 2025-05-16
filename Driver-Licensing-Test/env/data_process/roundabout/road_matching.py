from env.data_process.roundabout.geo_engine import GeoEngine
import cv2
import json
import numpy as np
import os
import copy


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def parse_config(path_to_json=r'./config.json'):
    with open(path_to_json) as f:
      data = json.load(f)
    return Struct(**data)


class RoadMatcher(GeoEngine):
    """
    Fast road matcher. Given a drivable map, this module maps any coordinate to
    the closest location within the drivable set.
    """

    def __init__(self, map_file_dir, map_height=1024, map_width=1024):
        super(RoadMatcher, self).__init__(map_file_dir, map_height=map_height, map_width=map_width)

        road_map = cv2.imread(map_file_dir, cv2.IMREAD_GRAYSCALE)
        self.road_map = cv2.resize(road_map, (map_width, map_height))

        npz_dir = os.path.splitext(map_file_dir)[0] + '_' + str(map_height) + '_' + str(map_width) + '.npz'
        if os.path.exists(npz_dir):
            maps = np.load(npz_dir)
            self.lookup_lat_map, self.lookup_lon_map = maps['lat_map'], maps['lon_map']
        else:
            self.lookup_lat_map, self.lookup_lon_map = self._create_lookup_maps()
            np.savez(npz_dir, lat_map=self.lookup_lat_map, lon_map=self.lookup_lon_map)

    def _create_lookup_maps(self):
        """
        Look-up table for fast road matching.
        """

        lookup_lat_map = np.zeros([self.h, self.w], np.float64).reshape([-1])
        lookup_lon_map = np.zeros([self.h, self.w], np.float64).reshape([-1])

        y_road, x_road = np.where(self.road_map > 250)

        road_pixel_pts = np.array([x_road, y_road]).T
        road_world_pts = self._pxl2world(road_pixel_pts)

        y_all, x_all = np.where(np.ones_like(self.road_map) > 0)
        all_pixel_pts = np.array([x_all, y_all]).T
        all_world_pts = self._pxl2world(all_pixel_pts)

        # to local coord
        # road_world_pts_norm = np.array(coord_normalization(
        #     road_world_pts[:,0], road_world_pts[:,1], self.f.tl[0], self.f.tl[1])).T.astype(np.float32)
        # all_world_pts_norm = np.array(coord_normalization(
        #     all_world_pts[:, 0], all_world_pts[:, 1], self.f.tl[0], self.f.tl[1])).T.astype(np.float32)
        road_world_pts_norm = np.array((road_world_pts[:,0], road_world_pts[:,1])).T.astype(np.float32)
        all_world_pts_norm = np.array((all_world_pts[:, 0], all_world_pts[:, 1])).T.astype(np.float32)

        # for each pt on map, find the closest road pt
        for i in range(len(all_world_pts)):
            diff_xy = all_world_pts_norm[i, :].reshape([1, 2]) - road_world_pts_norm
            dist = np.sqrt(diff_xy[:, 0]**2 + diff_xy[:, 1]**2)
            min_id = np.argmin(dist)
            lookup_lat_map[i] = road_world_pts[min_id, 0]
            lookup_lon_map[i] = road_world_pts[min_id, 1]
            if i % 50000 == 1:
                print('initializing road matching lookup maps... %.3f %%' % (i/len(all_pixel_pts)*100))

        lookup_lat_map = lookup_lat_map.reshape([self.h, self.w])
        lookup_lon_map = lookup_lon_map.reshape([self.h, self.w])

        print('initializing done.')

        return lookup_lat_map, lookup_lon_map

    def _within_map(self, lat, lon):
        lat_max = self.f.br[0]
        lat_min = self.f.tl[0]
        lon_max = self.f.tl[1]
        lon_min = self.f.br[1]
        if lat < lat_max and lat > lat_min and lon < lon_max and lon > lon_min:
            return True
        else:
            return False

    def _check_boundary(self, x0, y0):
        if x0 < 0:  x0 = 0
        if x0 >= self.w: x0 = self.w - 1
        if y0 < 0:  y0 = 0
        if y0 >= self.h: y0 = self.h - 1
        return x0, y0

    def road_matching(self, vehicle_list):

        vehicle_list_new = []
        for i in range(len(vehicle_list)):
            v = copy.deepcopy(vehicle_list[i])
            lat, lon = vehicle_list[i].location.x, vehicle_list[i].location.y
            pxl_pt = self._world2pxl([lat, lon])
            x0, y0 = pxl_pt[0], pxl_pt[1]
            x0, y0 = self._check_boundary(x0, y0)
            lat_, lon_ = self.lookup_lat_map[y0, x0], self.lookup_lon_map[y0, x0]
            v.location.x, v.location.y = lat_, lon_
            vehicle_list_new.append(v)

        return vehicle_list_new

