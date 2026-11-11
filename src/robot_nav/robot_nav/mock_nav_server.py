
import time
import rclpy

from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.action import GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from nav2_msgs.action import NavigateToPose


class MockNavServer(Node):

    def __init__(self):
        super().__init__('mock_nav_server')

        self.declare_parameter('scenario', 'success')
        self.scenario = (
            self.get_parameter('scenario').value
        )

        self.callback_group = ReentrantCallbackGroup()

        self.server = ActionServer(
            self,
            NavigateToPose,
            '/mock_navigate_to_pose',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group
        )

        self.get_logger().info(
            f'Mock Server ready, scenario={self.scenario}'
        )

    def goal_callback(self, goal_request):
        if self.scenario == 'reject':
            self.get_logger().warn('Rejecting goal')
            return GoalResponse.REJECT

        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Cancel requested')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        result = NavigateToPose.Result()

        # 模拟持续执行10秒
        for i in range(20):

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return result

            # 模拟导航中途失败
            if self.scenario == 'abort' and i == 6:
                goal_handle.abort()
                self.get_logger().warn('Goal aborted')
                return result

            feedback = NavigateToPose.Feedback()
            feedback.distance_remaining = float(20 - i) * 0.1

            goal_handle.publish_feedback(feedback)

            time.sleep(0.5)

        goal_handle.succeed()
        self.get_logger().info('Goal succeeded')

        return result


def main(args=None):
    rclpy.init(args=args)
    node = MockNavServer()

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
