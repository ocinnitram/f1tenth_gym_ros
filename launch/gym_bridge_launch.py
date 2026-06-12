# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from launch import LaunchDescription
from launch.actions import LogInfo, ExecuteProcess
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory
import os
import subprocess
import yaml

URDF_TMP_PATH = '/tmp/ego_racecar.urdf'

def generate_launch_description():
    ld = LaunchDescription()
    config = os.path.join(
        get_package_share_directory('f1tenth_gym_ros'),
        'config',
        'sim.yaml'
        )
    config_dict = yaml.safe_load(open(config, 'r'))
    has_opp = config_dict['bridge']['ros__parameters']['num_agent'] > 1
    teleop = config_dict['bridge']['ros__parameters']['kb_teleop']

    map_path = config_dict['bridge']['ros__parameters']['map_path']
    if not os.path.isabs(map_path):
        map_path = os.path.join(
            get_package_share_directory('roboracer_data'), 'maps', map_path
        )

    # Generate URDF file synchronously so foxglove_bridge can serve it as an asset
    xacro_path = os.path.join(get_package_share_directory('f1tenth_gym_ros'), 'launch', 'ego_racecar.xacro')
    subprocess.run(['xacro', xacro_path, '-o', URDF_TMP_PATH], check=True)

    bridge_node = Node(
        package='f1tenth_gym_ros',
        executable='gym_bridge',
        name='bridge',
        parameters=[config, {'map_path': map_path}]
    )
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', os.path.join(get_package_share_directory('f1tenth_gym_ros'), 'launch', 'gym_bridge.rviz')]
    )
    foxglove_bridge_node = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        parameters=[{'port': 8765, 'address': '0.0.0.0', 'send_buffer_limit': 10000000,
                     'num_threads': 4}]
    )
    urdf_http_server = ExecuteProcess(
        cmd=['python3', '-m', 'http.server', '8888', '--directory', '/tmp'],
        name='urdf_http_server',
        output='screen'
    )
    foxglove_layout_path = os.path.join(
        get_package_share_directory('f1tenth_gym_ros'),
        'config', 'foxglove', 'gym_bridge_foxglove.json'
    )
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        parameters=[{'yaml_filename': map_path + '.yaml'},
                    {'topic': 'map'},
                    {'frame_id': 'map'},
                    {'output': 'screen'},
                    {'use_sim_time': True}]
    )
    nav_lifecycle_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{'use_sim_time': True},
                    {'autostart': True},
                    {'node_names': ['map_server']}]
    )
    opp_robot_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='opp_robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', os.path.join(get_package_share_directory('f1tenth_gym_ros'), 'launch', 'opp_racecar.xacro')])}],
        remappings=[('/robot_description', 'opp_robot_description')]
    )

    # finalize
    ld.add_action(LogInfo(msg=['Foxglove WebSocket: ws://localhost:8765']))
    ld.add_action(LogInfo(msg=[f'URDF served at: http://localhost:8888/ego_racecar.urdf — use Source:URL in Foxglove URDF layer']))
    ld.add_action(urdf_http_server)
    ld.add_action(LogInfo(msg=['Foxglove layout file: ' + foxglove_layout_path]))
    ld.add_action(foxglove_bridge_node)
    # ld.add_action(rviz_node)  # use Foxglove instead; uncomment to re-enable RViz
    ld.add_action(bridge_node)
    ld.add_action(nav_lifecycle_node)
    ld.add_action(map_server_node)
    if has_opp:
        ld.add_action(opp_robot_publisher)

    return ld