
import math
from threading import Lock

import rclpy
from rclpy.node import Node
from rclpy.action import (
    ActionServer,
    ActionClient,
    GoalResponse,
    CancelResponse,
)
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from robot_interfaces.action import NavigateToRegion

from robot_nav.region_manager import RegionManager


class NavigateSkillServer(Node):

    def __init__(self):
        super().__init__('navigate_skill_server')

        self.region_manager = RegionManager()

        # 默认仅连接 Mock Server，避免误操作实机
        self.declare_parameter(
            'nav2_action_name',
            '/mock_navigate_to_pose'
        )
        nav2_action_name = (
            self.get_parameter('nav2_action_name').value
        )

        self.callback_group = ReentrantCallbackGroup()

        # 防止多个上层任务同时控制底层导航
        self.busy_lock = Lock()
        self.busy = False

        # 任务取消标志
        self.downstream_goal_handle = None
        self.cancel_requested = False
        self.cancel_sent = False

        # 下层：我们是 NavigateToPose Client
        self.nav2_client = ActionClient(
            self,
            NavigateToPose,
            nav2_action_name,
            callback_group=self.callback_group
        )

        # 上层：我们是 NavigateToRegion Server
        self.skill_server = ActionServer(
            self,
            NavigateToRegion,
            '/robot_nav/navigate_to_region',
            goal_callback=self.goal_callback,
            execute_callback=self.execute_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group
        )

        self.get_logger().info(
            f'NavigateSkill ready -> {nav2_action_name}'
        )

    def goal_callback(self, goal_request):
        region_id = goal_request.region_id

        target = self.region_manager.resolve(region_id)

        if target is None:
            self.get_logger().warn(
                f'Unknown region: {region_id}'
            )
            return GoalResponse.REJECT

        # 当前只允许一个导航任务
        with self.busy_lock:
            if self.busy:
                self.get_logger().warn('Navigation is busy')
                return GoalResponse.REJECT

            # 在接受 Goal 时提前占用任务槽
            self.busy = True
            self.downstream_goal_handle = None
            self.cancel_requested = False
            self.cancel_sent = False

        self.get_logger().info(
            f'Accepted region: {region_id}'
        )
        return GoalResponse.ACCEPT

    def build_nav2_goal(self, target):
        nav_goal = NavigateToPose.Goal()

        nav_goal.pose.header.frame_id = (
            self.region_manager.frame_id
        )
        nav_goal.pose.header.stamp = (
            self.get_clock().now().to_msg()
        )

        nav_goal.pose.pose.position.x = float(target['x'])
        nav_goal.pose.pose.position.y = float(target['y'])
        nav_goal.pose.pose.position.z = 0.0

        yaw = math.radians(target['yaw_deg'])

        nav_goal.pose.pose.orientation.z = math.sin(yaw / 2)
        nav_goal.pose.pose.orientation.w = math.cos(yaw / 2)

        nav_goal.behavior_tree = ''

        return nav_goal

    async def execute_callback(self, goal_handle):
        result = NavigateToRegion.Result()
        result.success = False

        # 只有确认下层没有活动任务，才允许释放 busy
        downstream_settled = True

        try:
            region_id = goal_handle.request.region_id
            target = self.region_manager.resolve(region_id)

            if target is None:
                result.message = 'Unknown region'
                goal_handle.abort()
                return result

            # 上层在执行开始前就完成了取消请求
            if goal_handle.is_cancel_requested:
                result.message = 'Canceled before Nav2 dispatch'
                goal_handle.canceled()
                return result

            if not self.nav2_client.server_is_ready():
                result.message = 'Nav2 Action Server unavailable'
                goal_handle.abort()
                return result

            nav_goal = self.build_nav2_goal(target)

            self.get_logger().info(
                f'Sending downstream goal: {region_id}'
            )

            # 发送请求后、收到最终结果前，
            # 下层是否仍在运行需要由 Action Result 确认。
            downstream_settled = False

            nav_handle = await self.nav2_client.send_goal_async(
                nav_goal,
                feedback_callback=lambda msg:
                    self.forward_feedback(goal_handle, msg)
            )

            if not nav_handle.accepted:
                downstream_settled = True
                result.message = 'Downstream goal rejected'
                goal_handle.abort()
                return result

            # 保存下层 GoalHandle，供取消回调使用
            with self.busy_lock:
                self.downstream_goal_handle = nav_handle

            # 先取得 Result Future
            result_future = nav_handle.get_result_async()

            # 可能在等待 Goal Accepted 时，
            # 上层已经请求过取消。
            self.forward_cancel_if_ready()

            # 异步等待下层最终结束
            nav_response = await result_future
            downstream_settled = True

            status = nav_response.status

            if status == GoalStatus.STATUS_SUCCEEDED:
                result.success = True
                result.message = f'Arrived at {region_id}'
                goal_handle.succeed()

            elif status == GoalStatus.STATUS_CANCELED:
                result.success = False

                if goal_handle.is_cancel_requested:
                    result.message = 'Navigation canceled'
                    goal_handle.canceled()
                else:
                    # 下层被其他来源取消，不应假装
                    # 上层也经历了正常的取消流程
                    result.message = 'Downstream canceled unexpectedly'
                    goal_handle.abort()

            elif status == GoalStatus.STATUS_ABORTED:
                result.message = 'Downstream navigation aborted'
                goal_handle.abort()

            else:
                result.message = f'Unexpected downstream status: {status}'
                goal_handle.abort()

        except Exception as exc:
            self.get_logger().error(
                f'NavigateSkill error: {exc}'
            )

            result.message = f'Navigation error: {exc}'

            if goal_handle.is_active:
                goal_handle.abort()

        finally:
            with self.busy_lock:
                if downstream_settled:
                    self.busy = False
                    self.downstream_goal_handle = None
                    self.cancel_requested = False
                    self.cancel_sent = False

            if not downstream_settled:
                self.get_logger().error(
                    'Downstream outcome unknown; '
                    'keeping BUSY until manual inspection'
                )

        return result

    def forward_feedback(self, goal_handle, feedback_msg):
        if not goal_handle.is_active:
            return

        nav_feedback = feedback_msg.feedback

        skill_feedback = NavigateToRegion.Feedback()
        skill_feedback.phase = 'NAVIGATING'
        skill_feedback.distance_remaining = (
            nav_feedback.distance_remaining
        )

        goal_handle.publish_feedback(skill_feedback)

    def cancel_callback(self, goal_handle):
        with self.busy_lock:
            if not self.busy:
                return CancelResponse.REJECT

            self.cancel_requested = True

        self.get_logger().warn(
            'Upstream cancel not implemented yet'
        )

        self.forward_cancel_if_ready()

        return CancelResponse.ACCEPT

    def forward_cancel_if_ready(self):
        with self.busy_lock:
            if (
                not self.cancel_requested
                or self.cancel_sent
                or self.downstream_goal_handle is None
            ):
                return

            nav_handle = self.downstream_goal_handle
            self.cancel_sent = True

        try:
            future = nav_handle.cancel_goal_async()
            future.add_done_callback(
                self.downstream_cancel_response_callback
            )
        except Exception as exc:
            # 取消发送是否生效无法确认，不报告已取消
            self.get_logger().error(
                f'Failed to send downstream cancel: {exc}'
            )

    def downstream_cancel_response_callback(self, future):
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(
                f'Downstream cancel response error: {exc}'
            )
            return

        if response.goals_canceling:
            self.get_logger().info(
                'Downstream cancel accepted; awaiting final result'
            )
        else:
            self.get_logger().warn(
                f'Downstream cancel not accepted, '
                f'return_code={response.return_code}'
            )


def main(args=None):
    rclpy.init(args=args)

    node = NavigateSkillServer()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
