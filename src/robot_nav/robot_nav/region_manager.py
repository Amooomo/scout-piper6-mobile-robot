
from pathlib import Path
import math
import yaml

from ament_index_python.packages import (
    get_package_share_directory
)


class RegionManager:

    def __init__(self):
        package_share = get_package_share_directory(
            'robot_nav'
        )

        config_path = (
            Path(package_share) / 'config' / 'regions.yaml'
        )

        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError('Invalid regions.yaml')

        self.frame_id = data.get('frame_id')

        if self.frame_id != 'map':
            raise ValueError('Expected frame_id: map')

        self.regions = data.get('regions')

        if not isinstance(self.regions, dict):
            raise ValueError('Missing regions dictionary')

        # 在启动阶段检查所有区域配置
        for region_id, pose in self.regions.items():
            if not isinstance(region_id, str):
                raise ValueError('Invalid region ID')

            if not isinstance(pose, dict):
                raise ValueError(
                    f'Invalid pose for {region_id}'
                )

            for field in ('x', 'y', 'yaw_deg'):
                value = pose.get(field)

                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(
                        f'Invalid {field} in {region_id}'
                    )

    def resolve(self, region_id):
        return self.regions.get(region_id)
