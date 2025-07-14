from pyniryo import NiryoRobot, JointsPosition

robot = NiryoRobot('10.10.10.10')

robot.calibrate_auto()
robot.update_tool()

tool = robot.get_current_tool_id()
print("Current tool:", tool)

robot.release_with_tool()
robot.move(JointsPosition(1.658, -0.282, -0.637, 0.032, -0.644, 0.167))
robot.grasp_with_tool()

robot.move(JointsPosition(1.658, -0.529, -0.637, 0.032, -0.411, 0.167))
robot.release_with_tool()

robot.close_connection()