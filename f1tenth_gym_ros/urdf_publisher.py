import subprocess
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String


class UrdfPublisher(Node):
    def __init__(self):
        super().__init__('urdf_publisher')
        self.declare_parameter('xacro_path', '')
        xacro_path = self.get_parameter('xacro_path').value

        result = subprocess.run(['xacro', xacro_path], capture_output=True, text=True)
        if result.returncode != 0:
            self.get_logger().error(f'xacro failed: {result.stderr}')
            self._urdf = ''
        else:
            self._urdf = result.stdout

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._pub = self.create_publisher(String, 'robot_description', qos)
        self.create_timer(1.0, self._publish)
        self._publish()

    def _publish(self):
        if self._urdf:
            msg = String()
            msg.data = self._urdf
            self._pub.publish(msg)


def main():
    rclpy.init()
    node = UrdfPublisher()
    rclpy.spin(node)
    rclpy.shutdown()
