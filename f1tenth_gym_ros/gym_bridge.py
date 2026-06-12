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

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Transform
from geometry_msgs.msg import Quaternion
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
from rcl_interfaces.msg import SetParametersResult
from tf2_ros import TransformBroadcaster

import gym
import numpy as np
import yaml
import os
from transforms3d import euler

class GymBridge(Node):
    def __init__(self):
        super().__init__('gym_bridge')

        # optional sys-ID params path — when set, loads vehicle params and applies actuator patch
        self.declare_parameter('sim_params_path', '')

        self.declare_parameter('ego_namespace')
        self.declare_parameter('ego_odom_topic')
        self.declare_parameter('ego_opp_odom_topic')
        self.declare_parameter('ego_scan_topic')
        self.declare_parameter('ego_drive_topic')
        self.declare_parameter('opp_namespace')
        self.declare_parameter('opp_odom_topic')
        self.declare_parameter('opp_ego_odom_topic')
        self.declare_parameter('opp_scan_topic')
        self.declare_parameter('opp_drive_topic')
        self.declare_parameter('scan_distance_to_base_link')
        self.declare_parameter('scan_fov')
        self.declare_parameter('scan_beams')
        self.declare_parameter('map_path')
        self.declare_parameter('map_img_ext')
        self.declare_parameter('num_agent')
        self.declare_parameter('sx')
        self.declare_parameter('sy')
        self.declare_parameter('stheta')
        self.declare_parameter('sx1')
        self.declare_parameter('sy1')
        self.declare_parameter('stheta1')
        self.declare_parameter('kb_teleop')

        # DR-injectable vehicle params (μ, m, I) — set at runtime via SetParameters
        self.declare_parameter('sim_mu', 0.0)
        self.declare_parameter('sim_m', 0.0)
        self.declare_parameter('sim_I', 0.0)
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # Friction-circle (per-axle lateral force clip at |F_y| <= mu*F_z).
        # When true, sim tires saturate at the friction limit instead of producing
        # unbounded force. Default true: matches real-car physics for slip diagnosis.
        self.declare_parameter('friction_circle_patch', True)
        # Publish wheel/motor rotational velocity (VESC-equivalent) on
        # /ego_racecar/odom.twist.linear.x instead of body velocity. Real PF
        # passes VESC velocity through, so during friction-circle exhaustion
        # the policy sees wheel-spin-inflated velocity even as body decelerates.
        # Requires friction_circle_patch=true (uses RaceCar._v_wheel state).
        self.declare_parameter('publish_wheel_velocity', True)

        # check num_agents
        num_agents = self.get_parameter('num_agent').value
        if num_agents < 1 or num_agents > 2:
            raise ValueError('num_agents should be either 1 or 2.')
        elif type(num_agents) != int:
            raise ValueError('num_agents should be an int.')

        # load sys-ID vehicle params if path provided, then apply actuator patch
        sim_params_path = self.get_parameter('sim_params_path').value
        self._vehicle_params = None
        if sim_params_path:
            if not os.path.isabs(sim_params_path):
                self.get_logger().warn(f'sim_params_path is relative: {sim_params_path}')
            with open(sim_params_path, 'r') as f:
                raw = yaml.safe_load(f)
            self._vehicle_params = raw['simulator_params']
            # Actuator patch FIRST (delays + better_pid). Friction patch SECOND
            # so its update_pose override wins; it inlines the actuator logic
            # and routes the integrator through the lagged-saturating dynamics.
            from f1tenth_gym_ros.actuator_patch import apply_custom_actuator_patch
            apply_custom_actuator_patch()
            self.get_logger().info(f'Loaded sys-ID params from {sim_params_path}; actuator patch applied.')
            if self.get_parameter('friction_circle_patch').value:
                from f1tenth_gym_ros.vehicle_model_patch import (
                    apply_friction_circle_patch, SIGMA_FY, LAT_PEAK_SCALE,
                )
                apply_friction_circle_patch()
                self.get_logger().info(
                    f'Friction patch applied: Pacejka/AWD/combined-slip single '
                    f'track, peak {LAT_PEAK_SCALE}x sliding mu, tire relaxation '
                    f'length {SIGMA_FY} m.'
                )

        # env backend
        make_kwargs = dict(
            map=self.get_parameter('map_path').value,
            map_ext=self.get_parameter('map_img_ext').value,
            num_agents=num_agents,
            lidar_dist=self.get_parameter("scan_distance_to_base_link").value,
        )
        if self._vehicle_params is not None:
            make_kwargs['params'] = self._vehicle_params
        self.env = gym.make('f110_gym:f110-v0', **make_kwargs)

        # Override gym's lap-count termination: lap_counter handles laps externally,
        # bridge only needs collision-driven done. Without this, env returns done=True
        # after 2 laps (toggle_list >= 4), which the bridge interprets as a crash.
        _base = self.env.unwrapped
        _orig_check_done = _base._check_done
        def _check_done_collision_only(_self=_base, _orig=_orig_check_done):
            _, toggle_list = _orig()  # preserve toggle_list / lap_counts side-effects
            return bool(_self.collisions[_self.ego_idx]), toggle_list
        _base._check_done = _check_done_collision_only

        sx = self.get_parameter('sx').value
        sy = self.get_parameter('sy').value
        stheta = self.get_parameter('stheta').value
        self.ego_pose = [sx, sy, stheta]
        self.ego_speed = [0.0, 0.0, 0.0]
        # Dead-reckoned VESC-equivalent odometry (wheel speed + gyro yaw rate,
        # never ground truth). Published on <ns>/wheel_odom for the sim PF's
        # motion model so the PF lags during slides exactly like the real one
        # (the real PF eats VESC wheel odom, which under-reports while the rear
        # is braked/slipping). Origin is arbitrary — the PF only uses deltas.
        self.wheel_odom_pose = [sx, sy, stheta]
        self._wheel_odom_last_t = None
        self.ego_requested_speed = 0.0
        self.ego_steer = 0.0
        self.ego_collision = False
        ego_scan_topic = self.get_parameter('ego_scan_topic').value
        ego_drive_topic = self.get_parameter('ego_drive_topic').value
        scan_fov = self.get_parameter('scan_fov').value
        scan_beams = self.get_parameter('scan_beams').value
        self.angle_min = -scan_fov / 2.
        self.angle_max = scan_fov / 2.
        self.angle_inc = scan_fov / scan_beams
        self.ego_namespace = self.get_parameter('ego_namespace').value
        ego_odom_topic = self.ego_namespace + '/' + self.get_parameter('ego_odom_topic').value
        self.scan_distance_to_base_link = self.get_parameter('scan_distance_to_base_link').value
        
        if num_agents == 2:
            self.has_opp = True
            self.opp_namespace = self.get_parameter('opp_namespace').value
            sx1 = self.get_parameter('sx1').value
            sy1 = self.get_parameter('sy1').value
            stheta1 = self.get_parameter('stheta1').value
            self.opp_pose = [sx1, sy1, stheta1]
            self.opp_speed = [0.0, 0.0, 0.0]
            self.opp_requested_speed = 0.0
            self.opp_steer = 0.0
            self.opp_collision = False
            self.obs, _ , self.done, _ = self.env.reset(np.array([[sx, sy, stheta], [sx1, sy1, stheta1]]))
            self.ego_scan = list(self.obs['scans'][0])
            self.opp_scan = list(self.obs['scans'][1])

            opp_scan_topic = self.get_parameter('opp_scan_topic').value
            opp_odom_topic = self.opp_namespace + '/' + self.get_parameter('opp_odom_topic').value
            opp_drive_topic = self.get_parameter('opp_drive_topic').value

            ego_opp_odom_topic = self.ego_namespace + '/' + self.get_parameter('ego_opp_odom_topic').value
            opp_ego_odom_topic = self.opp_namespace + '/' + self.get_parameter('opp_ego_odom_topic').value
        else:
            self.has_opp = False
            self.obs, _ , self.done, _ = self.env.reset(np.array([[sx, sy, stheta]]))
            self.ego_scan = list(self.obs['scans'][0])

        # latched collision publisher — lap_runner subscribes to detect run termination
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.collision_pub = self.create_publisher(Bool, '/sim/collision', latched_qos)
        self._collision_published = False

        # sim physical step timer — keep fast so dynamics stay smooth
        self.drive_timer = self.create_timer(0.01, self.drive_timer_callback)
        # sensor publish timers — match real-car rates (Sick LiDAR 15Hz, VESC odom 50Hz)
        self.scan_timer = self.create_timer(1.0 / 15.0, self.scan_timer_callback)
        self.odom_timer = self.create_timer(1.0 / 50.0, self.odom_timer_callback)
        # TF at 100 Hz — matches physics step so Foxglove position is never stale
        self.tf_timer = self.create_timer(0.01, self.tf_timer_callback)

        # transform broadcaster
        self.br = TransformBroadcaster(self)

        # publishers
        self.ego_scan_pub = self.create_publisher(LaserScan, ego_scan_topic, 10)
        self.ego_odom_pub = self.create_publisher(Odometry, ego_odom_topic, 10)
        self.wheel_odom_pub = self.create_publisher(
            Odometry, self.ego_namespace + '/wheel_odom', 10)
        self.ego_drive_published = False
        if num_agents == 2:
            self.opp_scan_pub = self.create_publisher(LaserScan, opp_scan_topic, 10)
            self.ego_opp_odom_pub = self.create_publisher(Odometry, ego_opp_odom_topic, 10)
            self.opp_odom_pub = self.create_publisher(Odometry, opp_odom_topic, 10)
            self.opp_ego_odom_pub = self.create_publisher(Odometry, opp_ego_odom_topic, 10)
            self.opp_drive_published = False

        # subscribers
        self.ego_drive_sub = self.create_subscription(
            AckermannDriveStamped,
            ego_drive_topic,
            self.drive_callback,
            10)
        self.ego_reset_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.ego_reset_callback,
            10)
        if num_agents == 2:
            self.opp_drive_sub = self.create_subscription(
                AckermannDriveStamped,
                opp_drive_topic,
                self.opp_drive_callback,
                10)
            self.opp_reset_sub = self.create_subscription(
                PoseStamped,
                '/goal_pose',
                self.opp_reset_callback,
                10)

        if self.get_parameter('kb_teleop').value:
            self.teleop_sub = self.create_subscription(
                Twist,
                '/cmd_vel',
                self.teleop_callback,
                10)


    def _on_set_parameters(self, params):
        """Apply DR param updates (sim_mu, sim_m, sim_I) to the live gym environment."""
        changed = {p.name: p.value for p in params if p.name in ('sim_mu', 'sim_m', 'sim_I')}
        if changed and self._vehicle_params is not None:
            new_params = dict(self._vehicle_params)
            if 'sim_mu' in changed and changed['sim_mu'] > 0.0:
                new_params['mu'] = changed['sim_mu']
            if 'sim_m' in changed and changed['sim_m'] > 0.0:
                new_params['m'] = changed['sim_m']
            if 'sim_I' in changed and changed['sim_I'] > 0.0:
                new_params['I'] = changed['sim_I']
            self._vehicle_params = new_params
            self.env.update_params(new_params)
            self.get_logger().info(
                f'DR params updated: mu={new_params["mu"]:.4f} m={new_params["m"]:.4f} I={new_params["I"]:.5f}'
            )
        return SetParametersResult(successful=True)

    def drive_callback(self, drive_msg):
        self.ego_requested_speed = drive_msg.drive.speed
        self.ego_steer = drive_msg.drive.steering_angle
        self.ego_drive_published = True

    def opp_drive_callback(self, drive_msg):
        self.opp_requested_speed = drive_msg.drive.speed
        self.opp_steer = drive_msg.drive.steering_angle
        self.opp_drive_published = True

    def ego_reset_callback(self, pose_msg):
        rx = pose_msg.pose.pose.position.x
        ry = pose_msg.pose.pose.position.y
        rqx = pose_msg.pose.pose.orientation.x
        rqy = pose_msg.pose.pose.orientation.y
        rqz = pose_msg.pose.pose.orientation.z
        rqw = pose_msg.pose.pose.orientation.w
        _, _, rtheta = euler.quat2euler([rqw, rqx, rqy, rqz], axes='sxyz')
        if self.has_opp:
            opp_pose = [self.obs['poses_x'][1], self.obs['poses_y'][1], self.obs['poses_theta'][1]]
            self.obs, _ , self.done, _ = self.env.reset(np.array([[rx, ry, rtheta], opp_pose]))
        else:
            self.obs, _ , self.done, _ = self.env.reset(np.array([[rx, ry, rtheta]]))
        self._collision_published = False
        # Re-seat the dead-reckoned wheel odom at the reset pose. The PF sees
        # one large pose delta from this jump, but /initialpose (this same
        # message) reinitialises its particles anyway.
        self.wheel_odom_pose = [rx, ry, rtheta]
        self._update_sim_state()

    def opp_reset_callback(self, pose_msg):
        if self.has_opp:
            rx = pose_msg.pose.position.x
            ry = pose_msg.pose.position.y
            rqx = pose_msg.pose.orientation.x
            rqy = pose_msg.pose.orientation.y
            rqz = pose_msg.pose.orientation.z
            rqw = pose_msg.pose.orientation.w
            _, _, rtheta = euler.quat2euler([rqw, rqx, rqy, rqz], axes='sxyz')
            self.obs, _ , self.done, _ = self.env.reset(np.array([list(self.ego_pose), [rx, ry, rtheta]]))
            self._update_sim_state()

    def teleop_callback(self, twist_msg):
        if not self.ego_drive_published:
            self.ego_drive_published = True

        self.ego_requested_speed = twist_msg.linear.x

        if twist_msg.angular.z > 0.0:
            self.ego_steer = 0.3
        elif twist_msg.angular.z < 0.0:
            self.ego_steer = -0.3
        else:
            self.ego_steer = 0.0

    def drive_timer_callback(self):
        if self.done:
            if not self._collision_published:
                msg = Bool()
                msg.data = True
                self.collision_pub.publish(msg)
                self._collision_published = True
            return
        if self.ego_drive_published and not self.has_opp:
            self.obs, _, self.done, _ = self.env.step(np.array([[self.ego_steer, self.ego_requested_speed]]))
        elif self.ego_drive_published and self.has_opp and self.opp_drive_published:
            self.obs, _, self.done, _ = self.env.step(np.array([[self.ego_steer, self.ego_requested_speed], [self.opp_steer, self.opp_requested_speed]]))
        if self.done and not self._collision_published:
            msg = Bool()
            msg.data = True
            self.collision_pub.publish(msg)
            self._collision_published = True
        self._update_sim_state()

    def scan_timer_callback(self):
        ts = self.get_clock().now().to_msg()

        scan = LaserScan()
        scan.header.stamp = ts
        scan.header.frame_id = self.ego_namespace + '/laser'
        scan.angle_min = self.angle_min
        scan.angle_max = self.angle_max
        scan.angle_increment = self.angle_inc
        scan.range_min = 0.
        scan.range_max = 30.
        scan.ranges = self.ego_scan
        self.ego_scan_pub.publish(scan)

        if self.has_opp:
            opp_scan = LaserScan()
            opp_scan.header.stamp = ts
            opp_scan.header.frame_id = self.opp_namespace + '/laser'
            opp_scan.angle_min = self.angle_min
            opp_scan.angle_max = self.angle_max
            opp_scan.angle_increment = self.angle_inc
            opp_scan.range_min = 0.
            opp_scan.range_max = 30.
            opp_scan.ranges = self.opp_scan
            self.opp_scan_pub.publish(opp_scan)

    def odom_timer_callback(self):
        ts = self.get_clock().now().to_msg()
        self._publish_odom(ts)
        self._publish_wheel_odom(ts)

    def _wheel_velocity(self):
        """VESC-equivalent wheel speed: RaceCar._v_wheel when the friction
        patch is active, longitudinal body velocity otherwise."""
        try:
            car = self.env.unwrapped.sim.agents[0]
            if hasattr(car, '_v_wheel'):
                return float(car._v_wheel)
        except Exception:
            pass
        return self.ego_speed[0]

    def _publish_wheel_odom(self, ts):
        # Dead-reckon the x/y advance from the WHEEL speed, so during
        # friction-circle slip this pose under-reports motion like the real
        # VESC odom and the PF lags behind the true car. The ORIENTATION is
        # the true yaw (gyro equivalent): the real PF replaces the VESC's
        # steering-derived heading with the integrated IMU gyro, which tracks
        # true heading closely. Integrating yaw rate here at 50 Hz instead
        # aliased badly during slide events (several rad/s swings), rotated
        # the particle cloud faster than scan matching could correct, and the
        # PF diverged unrecoverably (14 m behind by run end, 2026-06-10).
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.02 if self._wheel_odom_last_t is None else now - self._wheel_odom_last_t
        self._wheel_odom_last_t = now
        if not (0.0 < dt < 0.5):
            dt = 0.02

        v_w = self._wheel_velocity()
        yaw_rate = self.ego_speed[2]
        theta = self.ego_pose[2]
        self.wheel_odom_pose[0] += v_w * np.cos(theta) * dt
        self.wheel_odom_pose[1] += v_w * np.sin(theta) * dt
        self.wheel_odom_pose[2] = theta

        odom = Odometry()
        odom.header.stamp = ts
        odom.header.frame_id = 'odom'
        odom.child_frame_id = self.ego_namespace + '/base_link'
        odom.pose.pose.position.x = self.wheel_odom_pose[0]
        odom.pose.pose.position.y = self.wheel_odom_pose[1]
        quat = euler.euler2quat(0., 0., self.wheel_odom_pose[2], axes='sxyz')
        odom.pose.pose.orientation.x = quat[1]
        odom.pose.pose.orientation.y = quat[2]
        odom.pose.pose.orientation.z = quat[3]
        odom.pose.pose.orientation.w = quat[0]
        odom.twist.twist.linear.x = v_w
        odom.twist.twist.angular.z = yaw_rate
        self.wheel_odom_pub.publish(odom)

    def tf_timer_callback(self):
        ts = self.get_clock().now().to_msg()
        self._publish_transforms(ts)
        self._publish_laser_transforms(ts)
        self._publish_wheel_transforms(ts)

    def _update_sim_state(self):
        self.ego_scan = list(self.obs['scans'][0])
        if self.has_opp:
            self.opp_scan = list(self.obs['scans'][1])
            self.opp_pose[0] = self.obs['poses_x'][1]
            self.opp_pose[1] = self.obs['poses_y'][1]
            self.opp_pose[2] = self.obs['poses_theta'][1]
            self.opp_speed[0] = self.obs['linear_vels_x'][1]
            self.opp_speed[1] = self.obs['linear_vels_y'][1]
            self.opp_speed[2] = self.obs['ang_vels_z'][1]

        self.ego_pose[0] = self.obs['poses_x'][0]
        self.ego_pose[1] = self.obs['poses_y'][0]
        self.ego_pose[2] = self.obs['poses_theta'][0]
        self.ego_speed[0] = self.obs['linear_vels_x'][0]
        self.ego_speed[1] = self.obs['linear_vels_y'][0]
        self.ego_speed[2] = self.obs['ang_vels_z'][0]

        

    def _publish_odom(self, ts):
        ego_odom = Odometry()
        ego_odom.header.stamp = ts
        ego_odom.header.frame_id = 'map'
        ego_odom.child_frame_id = self.ego_namespace + '/base_link'
        ego_odom.pose.pose.position.x = self.ego_pose[0]
        ego_odom.pose.pose.position.y = self.ego_pose[1]
        ego_quat = euler.euler2quat(0., 0., self.ego_pose[2], axes='sxyz')
        ego_odom.pose.pose.orientation.x = ego_quat[1]
        ego_odom.pose.pose.orientation.y = ego_quat[2]
        ego_odom.pose.pose.orientation.z = ego_quat[3]
        ego_odom.pose.pose.orientation.w = ego_quat[0]
        # Publish either body velocity (stock behavior) or wheel/motor rotational
        # velocity (VESC-equivalent — vehicle_model_patch tracks self._v_wheel on the
        # RaceCar instance, which decouples from body during friction-circle
        # exhaustion).
        v_publish = self.ego_speed[0]
        if self.get_parameter('publish_wheel_velocity').value:
            try:
                car = self.env.unwrapped.sim.agents[0]
                if hasattr(car, '_v_wheel'):
                    v_publish = float(car._v_wheel)
            except Exception:
                pass
        ego_odom.twist.twist.linear.x = v_publish
        ego_odom.twist.twist.linear.y = self.ego_speed[1]
        ego_odom.twist.twist.angular.z = self.ego_speed[2]
        self.ego_odom_pub.publish(ego_odom)

        if self.has_opp:
            opp_odom = Odometry()
            opp_odom.header.stamp = ts
            opp_odom.header.frame_id = 'map'
            opp_odom.child_frame_id = self.opp_namespace + '/base_link'
            opp_odom.pose.pose.position.x = self.opp_pose[0]
            opp_odom.pose.pose.position.y = self.opp_pose[1]
            opp_quat = euler.euler2quat(0., 0., self.opp_pose[2], axes='sxyz')
            opp_odom.pose.pose.orientation.x = opp_quat[1]
            opp_odom.pose.pose.orientation.y = opp_quat[2]
            opp_odom.pose.pose.orientation.z = opp_quat[3]
            opp_odom.pose.pose.orientation.w = opp_quat[0]
            opp_odom.twist.twist.linear.x = self.opp_speed[0]
            opp_odom.twist.twist.linear.y = self.opp_speed[1]
            opp_odom.twist.twist.angular.z = self.opp_speed[2]
            self.opp_odom_pub.publish(opp_odom)
            self.opp_ego_odom_pub.publish(ego_odom)
            self.ego_opp_odom_pub.publish(opp_odom)

    def _publish_transforms(self, ts):
        ego_t = Transform()
        ego_t.translation.x = self.ego_pose[0]
        ego_t.translation.y = self.ego_pose[1]
        ego_t.translation.z = 0.0
        ego_quat = euler.euler2quat(0.0, 0.0, self.ego_pose[2], axes='sxyz')
        ego_t.rotation.x = ego_quat[1]
        ego_t.rotation.y = ego_quat[2]
        ego_t.rotation.z = ego_quat[3]
        ego_t.rotation.w = ego_quat[0]

        ego_ts = TransformStamped()
        ego_ts.transform = ego_t
        ego_ts.header.stamp = ts
        ego_ts.header.frame_id = 'map'
        ego_ts.child_frame_id = self.ego_namespace + '/base_link'
        self.br.sendTransform(ego_ts)

        if self.has_opp:
            opp_t = Transform()
            opp_t.translation.x = self.opp_pose[0]
            opp_t.translation.y = self.opp_pose[1]
            opp_t.translation.z = 0.0
            opp_quat = euler.euler2quat(0.0, 0.0, self.opp_pose[2], axes='sxyz')
            opp_t.rotation.x = opp_quat[1]
            opp_t.rotation.y = opp_quat[2]
            opp_t.rotation.z = opp_quat[3]
            opp_t.rotation.w = opp_quat[0]

            opp_ts = TransformStamped()
            opp_ts.transform = opp_t
            opp_ts.header.stamp = ts
            opp_ts.header.frame_id = 'map'
            opp_ts.child_frame_id = self.opp_namespace + '/base_link'
            self.br.sendTransform(opp_ts)

    def _publish_wheel_transforms(self, ts):
        ego_wheel_ts = TransformStamped()
        ego_wheel_quat = euler.euler2quat(0., 0., self.ego_steer, axes='sxyz')
        ego_wheel_ts.transform.rotation.x = ego_wheel_quat[1]
        ego_wheel_ts.transform.rotation.y = ego_wheel_quat[2]
        ego_wheel_ts.transform.rotation.z = ego_wheel_quat[3]
        ego_wheel_ts.transform.rotation.w = ego_wheel_quat[0]
        ego_wheel_ts.header.stamp = ts
        ego_wheel_ts.header.frame_id = self.ego_namespace + '/front_left_hinge'
        ego_wheel_ts.child_frame_id = self.ego_namespace + '/front_left_wheel'
        self.br.sendTransform(ego_wheel_ts)
        ego_wheel_ts.header.frame_id = self.ego_namespace + '/front_right_hinge'
        ego_wheel_ts.child_frame_id = self.ego_namespace + '/front_right_wheel'
        self.br.sendTransform(ego_wheel_ts)

        if self.has_opp:
            opp_wheel_ts = TransformStamped()
            opp_wheel_quat = euler.euler2quat(0., 0., self.opp_steer, axes='sxyz')
            opp_wheel_ts.transform.rotation.x = opp_wheel_quat[1]
            opp_wheel_ts.transform.rotation.y = opp_wheel_quat[2]
            opp_wheel_ts.transform.rotation.z = opp_wheel_quat[3]
            opp_wheel_ts.transform.rotation.w = opp_wheel_quat[0]
            opp_wheel_ts.header.stamp = ts
            opp_wheel_ts.header.frame_id = self.opp_namespace + '/front_left_hinge'
            opp_wheel_ts.child_frame_id = self.opp_namespace + '/front_left_wheel'
            self.br.sendTransform(opp_wheel_ts)
            opp_wheel_ts.header.frame_id = self.opp_namespace + '/front_right_hinge'
            opp_wheel_ts.child_frame_id = self.opp_namespace + '/front_right_wheel'
            self.br.sendTransform(opp_wheel_ts)

    def _publish_laser_transforms(self, ts):
        ego_scan_ts = TransformStamped()
        ego_scan_ts.transform.translation.x = self.scan_distance_to_base_link
        # ego_scan_ts.transform.translation.z = 0.04+0.1+0.025
        ego_scan_ts.transform.rotation.w = 1.
        ego_scan_ts.header.stamp = ts
        ego_scan_ts.header.frame_id = self.ego_namespace + '/base_link'
        ego_scan_ts.child_frame_id = self.ego_namespace + '/laser'
        self.br.sendTransform(ego_scan_ts)

        if self.has_opp:
            opp_scan_ts = TransformStamped()
            opp_scan_ts.transform.translation.x = self.scan_distance_to_base_link
            opp_scan_ts.transform.rotation.w = 1.
            opp_scan_ts.header.stamp = ts
            opp_scan_ts.header.frame_id = self.opp_namespace + '/base_link'
            opp_scan_ts.child_frame_id = self.opp_namespace + '/laser'
            self.br.sendTransform(opp_scan_ts)

def main(args=None):
    rclpy.init(args=args)
    gym_bridge = GymBridge()
    rclpy.spin(gym_bridge)

if __name__ == '__main__':
    main()