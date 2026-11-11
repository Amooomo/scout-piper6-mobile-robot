import argparse
import math
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.utilities import remove_ros_args

from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from std_srvs.srv import Trigger

from enum import Enum, auto


class NavState(Enum):
    IDLE = auto()
    WAIT_ACCEPT = auto()
    NAVIGATING = auto()
    CANCELING = auto()
    SUCCEEDED = auto()
    ABORTED = auto()
    CANCELED = auto()
    REJECTED = auto()
    ERROR = auto()


class NavClient(Node):
    def __init__(self):
        super().__init__("nav_client")
        # 声明 Action Server 名称参数
        self.declare_parameter(
            'action_name',
            '/navigate_to_pose'
        )

        # 读取实际配置
        action_name = self.get_parameter(
            'action_name'
        ).value

        # 创建 Action Client
        self.client = ActionClient(
            self,
            NavigateToPose,
            action_name
        )

        self.get_logger().info(
            f'Connecting to Action Server: {action_name}'
        )
        
        self.cancel_service = self.create_service(
            Trigger,
            '/robot_nav/cancel',
            self.cancel_service_callback
        )

        self.last_feedback_time = 0.0

        self.state = NavState.IDLE
        self.goal_handle = None
        self.cancel_requested = False
        self.last_result = None
        self.cancel_result = None
        self.cancel_sent = False

    def send_goal(self, x, y, yaw) -> bool:
        if self.state not in (
            NavState.IDLE,
            NavState.SUCCEEDED,
            NavState.ABORTED,
            NavState.CANCELED,
            NavState.REJECTED
        ):
            self.get_logger().warn(
                f'Cannot send goal in state {self.state.name}'
            )
            return False

        self.state = NavState.WAIT_ACCEPT
        self.goal_handle = None
        self.cancel_requested = False
        self.cancel_sent = False
        self.last_result = None

        goal = NavigateToPose.Goal()

        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()

        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.position.z = 0.0

        yaw_rad = math.radians(yaw)

        goal.pose.pose.orientation.z = math.sin(yaw_rad / 2)
        goal.pose.pose.orientation.w = math.cos(yaw_rad / 2)

        goal.behavior_tree = ''

        # 2. 异步发送 Goal
        future = self.client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback
        )

        # 3. 注册目标响应回调
        future.add_done_callback(
            self.goal_response_callback
        )

        self.get_logger().info(
            f'Goal sent: x={x}, y={y}, yaw={yaw} deg'
        )

        return True

    def goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f'Send goal failed: {exc}')
            self.state = NavState.ERROR
            return

        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            self.state = NavState.REJECTED
            self.cancel_requested = False
            return

        self.get_logger().info('Goal accepted')
        self.goal_handle = goal_handle
        self.state = NavState.NAVIGATING

        # 4. 异步等待导航最终结果
        result_future = self.goal_handle.get_result_async()

        result_future.add_done_callback(
            self.result_callback
        )

        # 如果接受 Goal 前用户已经请求取消
        if self.cancel_requested:
            self.request_cancel()

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback

        # 限制日志频率，避免每条反馈都打印
        now = time.monotonic()

        if now - self.last_feedback_time >= 2.0:
            self.last_feedback_time = now

            self.get_logger().info(
                f'Distance remaining: '
                f'{feedback.distance_remaining:.2f} m'
            )

    def result_callback(self, future):
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f'Get result failed: {exc}')
            self.state = NavState.ERROR
            return

        status = response.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.state = NavState.SUCCEEDED
            self.get_logger().info('Navigation succeeded')

        elif status == GoalStatus.STATUS_CANCELED:
            self.state = NavState.CANCELED
            self.get_logger().warn('Navigation canceled')

        elif status == GoalStatus.STATUS_ABORTED:
            self.state = NavState.ABORTED
            self.get_logger().error('Navigation aborted')

        else:
            self.state = NavState.ERROR
            self.get_logger().warn(
                f'Unexpected goal status: {status}'
            )

        self.last_result = status
        self.goal_handle = None
        self.cancel_requested = False
        self.cancel_sent = False

    def finish(self):
        if rclpy.ok():
            rclpy.shutdown()

    def cancel_service_callback(self, request, response):
        if self.state not in (
            NavState.WAIT_ACCEPT,
            NavState.NAVIGATING,
            NavState.CANCELING
        ):
            response.success = False
            response.message = 'No active navigation goal'
            return response

        self.cancel_requested = True

        if self.state == NavState.NAVIGATING:
            self.request_cancel()

        response.success = True
        response.message = (
            'Cancel requested; awaiting final action result'
        )
        return response

    def request_cancel(self):
        if self.goal_handle is None:
            self.get_logger().warn(
                'Goal handle not available'
            )
            return

        if self.cancel_sent:
            self.get_logger().warn(
                'Cancel already requested'
            )
            return

        self.cancel_requested = True
        self.cancel_sent = True
        self.state = NavState.CANCELING

        try:
            cancel_future = self.goal_handle.cancel_goal_async()
        except Exception as exc:
            self.state = NavState.ERROR
            self.get_logger().error(
                f'Cannot send cancel request: {exc}'
            )
            return

        cancel_future.add_done_callback(
            self.cancel_response_callback
        )

    def cancel_response_callback(self, future):
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(
                f'Cancel response failed: {exc}'
            )
            self.state = NavState.ERROR
            return

        if response.goals_canceling:
            self.get_logger().info(
                'Cancel accepted, waiting for final result'
            )
        else:
            self.get_logger().warn(
                f'Cancel not accepted, code={response.return_code}'
            )

            # 可能已经正常完成，也可能仍在导航
            # 不应该直接设为 CANCELED
            if self.state == NavState.CANCELING:
                self.state = NavState.NAVIGATING

            self.cancel_requested = False
            self.cancel_sent = False



def main(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('x', type=float)
    parser.add_argument('y', type=float)
    parser.add_argument('yaw', type=float)

    cli_args = remove_ros_args(args=sys.argv)[1:]
    target = parser.parse_args(cli_args)

    rclpy.init(args=args)
    node = NavClient()

    try:
        # 启动阶段最多等待 5 秒
        if not node.client.wait_for_server(timeout_sec=5.0):
            node.get_logger().error(
                'NavigateToPose server unavailable'
            )
            return

        node.send_goal(
            target.x,
            target.y,
            target.yaw
        )

        node.get_logger().info("Entering rclpy.spin()")

        # Executor 处理 Goal、Feedback、Result 回调
        rclpy.spin(node)

    except (KeyboardInterrupt, ExternalShutdownException):
        # 注意：退出 Client 不等于取消 Nav2 Goal
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()