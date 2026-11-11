
import time

import rclpy
import py_trees

from rclpy.node import Node
from rclpy.action import ActionClient

from action_msgs.msg import GoalStatus
from robot_interfaces.action import NavigateToRegion

class NavigateRegion(py_trees.behaviour.Behaviour):

    def __init__(self, node, region_id):
        super().__init__(name="NavigateToRegion")

        self.node = node
        self.region_id = region_id

        self.client = ActionClient(
            node,
            NavigateToRegion,
            '/robot_nav/navigate_to_region'
        )

        self.finished = False
        self.succeeded = False
        self.goal_sent = False

        # 用于取消的标志位
        self.goal_handle = None
        self.cancel_pending = False
        self.cancel_sent = False
        self.action_settled = True

    def initialise(self):
        # 本次行为开始时重置状态
        self.finished = False
        self.succeeded = False
        self.goal_sent = False
        self.start_time = time.monotonic()
        self.last_feedback_time = 0.0
        self.goal_handle = None
        self.cancel_pending = False
        self.cancel_sent = False

    def update(self):
        # 已经获得 Action Result
        if self.finished:
            if self.succeeded:
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.FAILURE

        # 尚未发送 Goal
        if not self.goal_sent:

            # 非阻塞地等待 Skill Server
            if not self.client.server_is_ready():
                if time.monotonic() - self.start_time > 5.0:
                    self.node.get_logger().error(
                        'NavigateSkill unavailable'
                    )
                    return py_trees.common.Status.FAILURE

                return py_trees.common.Status.RUNNING

            goal = NavigateToRegion.Goal()
            goal.region_id = self.region_id

            # 从这一刻起，存在尚未确认结束的导航请求
            self.action_settled = False

            future = self.client.send_goal_async(
                goal,
                feedback_callback=self.feedback_callback
            )

            self.goal_sent = True

            future.add_done_callback(
                self.goal_response_callback
            )

        # Goal 已发送，但执行尚未结束
        return py_trees.common.Status.RUNNING

    def goal_response_callback(self, future):
        try:
            goal_handle = future.result()

            if not goal_handle.accepted:
                self.node.get_logger().error(
                    'NavigateToRegion goal rejected'
                )
                self.action_settled = True
                self.finished = True
                return

            self.node.get_logger().info(
                'NavigateToRegion goal accepted'
            )

            self.goal_handle = goal_handle

            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(
                self.result_callback
            )

            self.send_cancel_if_ready()

        except Exception as exc:
            self.node.get_logger().error(str(exc))
            self.finished = True

    def feedback_callback(self, msg):
        now = time.monotonic()

        if now - self.last_feedback_time < 2.0:
            return

        self.last_feedback_time = now
        feedback = msg.feedback

        self.node.get_logger().info(
            f'{feedback.phase}: '
            f'{feedback.distance_remaining:.2f} m'
        )

    def result_callback(self, future):
        try:
            response = future.result()

        except Exception as exc:
            self.succeeded = False
            self.finished = True

            self.node.get_logger().error(
                f'Failed to get Action Result: {exc}'
            )

            # 注意：没有拿到最终 Result
            # 不能确认下层 Action 是否已经结束
            # 因此不修改 action_settled 和 goal_handle
            return

        status = response.status

        terminal_statuses = (
            GoalStatus.STATUS_SUCCEEDED,
            GoalStatus.STATUS_ABORTED,
            GoalStatus.STATUS_CANCELED,
        )

        if status not in terminal_statuses:
            self.succeeded = False
            self.finished = True

            self.node.get_logger().error(
                f'Unexpected Action status: {status}'
            )
            return

        # 已经获得明确的终止状态
        self.action_settled = True
        self.goal_handle = None

        self.succeeded = (
            status == GoalStatus.STATUS_SUCCEEDED
            and response.result.success
        )

        self.finished = True

        self.node.get_logger().info(
            f'Skill result: {response.result.message}, '
            f'status={status}'
        )

    def terminate(self, new_status):
        if (
            self.status == py_trees.common.Status.RUNNING
            and new_status == py_trees.common.Status.INVALID
            and self.goal_sent
            and not self.action_settled
        ):
            self.node.get_logger().warn(
                "BT interrupted: request Action cancel"
            )

            self.cancel_pending = True
            self.send_cancel_if_ready()

    def send_cancel_if_ready(self):
        if not self.cancel_pending or self.cancel_sent or self.goal_handle is None:
            return

        self.cancel_sent = True

        try:
            future = self.goal_handle.cancel_goal_async()
            future.add_done_callback(
                self.cancel_response_callback
            )
        except Exception as exc:
            self.node.get_logger().error(
                f"Cancel request failed: {exc}"
            )


    def cancel_response_callback(self, future):
        try:
            response = future.result()

            if response.goals_canceling:
                self.node.get_logger().info(
                    "Cancel accepted, waiting for final result"
                )
            else:
                self.node.get_logger().warn(
                    f"Cancel not accepted: {response.return_code}"
                )

        except Exception as exc:
            self.node.get_logger().error(
                f"Cancel response error: {exc}"
            )

class TaskBTNode(Node):

    def __init__(self):
        super().__init__('task_bt')

        self.declare_parameter(
            'region_id',
            'experiment_table_A'
        )
        region_id = self.get_parameter('region_id').value

        self.declare_parameter(
            'test_interrupt_after_sec', 0.0
        )

        delay = self.get_parameter(
            'test_interrupt_after_sec'
        ).value

        if delay > 0.0:
            self.test_interrupt_timer = self.create_timer(
                delay,
                self.test_interrupt_once
            )

        # 创建任务级行为树
        root = py_trees.composites.Sequence(
            name='MobileTask',
            memory=True,
            children=[
                NavigateRegion(self, region_id),

                # 测试占位节点：不执行真实操作
                py_trees.behaviours.Success(
                    name='NextStepPlaceholder'
                )
            ]
        )

        self.tree = py_trees.trees.BehaviourTree(root)

        # 每 0.2 秒 Tick 一次
        self.timer = self.create_timer(
            0.2,
            self.tick_tree
        )

    def tick_tree(self):
        self.tree.tick()

        status = self.tree.root.status

        if status == py_trees.common.Status.SUCCESS:
            self.get_logger().info('Task BT succeeded')
            self.timer.cancel()

        elif status == py_trees.common.Status.FAILURE:
            self.get_logger().error('Task BT failed')
            self.timer.cancel()

    def test_interrupt_once(self):
        self.test_interrupt_timer.cancel()

        if self.tree.root.status != py_trees.common.Status.RUNNING:
            self.get_logger().warn(
                'BT is not running'
            )
            return

        self.get_logger().warn(
            'Test: interrupting BT'
        )

        # 停止后续 Tick，避免立即重新开始任务
        self.timer.cancel()

        # 触发运行中的叶子节点 terminate(INVALID)
        self.tree.root.stop(
            py_trees.common.Status.INVALID
        )

def main(args=None):
    rclpy.init(args=args)
    node = TaskBTNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()