import time
import math

import mujoco
import mujoco.viewer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "model.xml"


# ============================================================
# LOAD MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)


# ============================================================
# HELPER: LOOK UP IDs
# ============================================================

def actuator_id(name):
    return mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name
    )


def sensor_id(name):
    return mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SENSOR,
        name
    )


# ============================================================
# ACTUATOR IDS
# ============================================================

wheel_fr_id = actuator_id("wheel_motor_FR")
wheel_fl_id = actuator_id("wheel_motor_FL")
wheel_br_id = actuator_id("wheel_motor_BR")
wheel_bl_id = actuator_id("wheel_motor_BL")

hip_fr_id = actuator_id("hip_motor_FR")
hip_fl_id = actuator_id("hip_motor_FL")
hip_br_id = actuator_id("hip_motor_BR")
hip_bl_id = actuator_id("hip_motor_BL")


# ============================================================
# SENSOR IDS
# ============================================================

orientation_sensor_id = sensor_id("imu_orientation")
gyro_sensor_id = sensor_id("imu_gyro")
accel_sensor_id = sensor_id("imu_accel")
position_sensor_id = sensor_id("robot_position")


# ============================================================
# SENSOR READING HELPER
# ============================================================

def read_sensor(sensor_index):

    start = model.sensor_adr[sensor_index]
    dimension = model.sensor_dim[sensor_index]

    return data.sensordata[start:start + dimension].copy()


# ============================================================
# ORIENTATION
# ============================================================

def quaternion_to_euler(quat):
    """
    MuJoCo quaternion:
        [w, x, y, z]

    Returns:
        roll, pitch, yaw
    in radians.
    """

    w, x, y, z = quat

    # Roll
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)

    roll = math.atan2(
        sinr_cosp,
        cosr_cosp
    )

    # Pitch
    sinp = 2.0 * (w * y - z * x)

    sinp = max(-1.0, min(1.0, sinp))

    pitch = math.asin(sinp)

    # Yaw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

    yaw = math.atan2(
        siny_cosp,
        cosy_cosp
    )

    return roll, pitch, yaw


def get_orientation():

    quat = read_sensor(orientation_sensor_id)

    return quaternion_to_euler(quat)


def get_yaw():

    _, _, yaw = get_orientation()

    return yaw


def get_position():

    return read_sensor(position_sensor_id)


def get_gyro():

    return read_sensor(gyro_sensor_id)


def get_acceleration():

    return read_sensor(accel_sensor_id)


# ============================================================
# ANGLE UTILITIES
# ============================================================

def wrap_angle(angle):
    """
    Wrap an angle into the range:

        -pi <= angle <= pi
    """

    return math.atan2(
        math.sin(angle),
        math.cos(angle)
    )


# ============================================================
# HIP CONTROL
# ============================================================

def set_hips(fr=0.0, fl=0.0, br=0.0, bl=0.0):

    data.ctrl[hip_fr_id] = fr
    data.ctrl[hip_fl_id] = fl
    data.ctrl[hip_br_id] = br
    data.ctrl[hip_bl_id] = bl


# ============================================================
# WHEEL CONTROL
# ============================================================

def set_wheels(fr, fl, br, bl):

    data.ctrl[wheel_fr_id] = fr
    data.ctrl[wheel_fl_id] = fl
    data.ctrl[wheel_br_id] = br
    data.ctrl[wheel_bl_id] = bl


def drive(speed):
    """
    Positive = forward.
    Negative = backward.

    Your experimentally verified convention:

        negative wheel velocity = forward.
    """

    wheel_speed = -speed

    set_wheels(
        wheel_speed,
        wheel_speed,
        wheel_speed,
        wheel_speed
    )


def turn_left(speed):

    set_wheels(
        -speed,   # FR forward
        speed,    # FL backward
        -speed,   # BR forward
        speed     # BL backward
    )


def turn_right(speed):

    set_wheels(
        speed,
        -speed,
        speed,
        -speed
    )


def stop():

    set_wheels(
        0.0,
        0.0,
        0.0,
        0.0
    )


# ============================================================
# SIMULATION STEP
# ============================================================

def step_sim(viewer):

    step_start = time.time()

    mujoco.mj_step(model, data)

    viewer.sync()

    elapsed = time.time() - step_start
    remaining = model.opt.timestep - elapsed

    if remaining > 0:
        time.sleep(remaining)


def simulate_for(viewer, duration):

    start_time = data.time

    while (
        viewer.is_running()
        and data.time - start_time < duration
    ):

        step_sim(viewer)


# ============================================================
# CLOSED-LOOP TURN
# ============================================================

def turn_to_relative_angle(
    viewer,
    angle_degrees,
    max_speed=2.0,
    kp=2.0,
    tolerance_degrees=2.0
):
    """
    Turn the robot by a requested relative yaw angle.

    Positive angle = left.
    Negative angle = right.
    """

    start_yaw = get_yaw()

    target_yaw = wrap_angle(
        start_yaw
        + math.radians(angle_degrees)
    )

    tolerance = math.radians(
        tolerance_degrees
    )

    print(
        f"Turning {angle_degrees:.1f} degrees..."
    )


    while viewer.is_running():

        current_yaw = get_yaw()

        error = wrap_angle(
            target_yaw - current_yaw
        )

        # Finished
        if abs(error) < tolerance:

            stop()

            print(
                "Target reached. "
                f"Final error = "
                f"{math.degrees(error):.2f} degrees"
            )

            break


        # Proportional controller
        speed = kp * abs(error)

        # Clamp wheel speed
        speed = min(
            max_speed,
            max(0.3, speed)
        )


        if error > 0:

            turn_left(speed)

        else:

            turn_right(speed)


        step_sim(viewer)


    stop()


# ============================================================
# DEMONSTRATION
# ============================================================

def run_demo(viewer):

    print()
    print("======================================")
    print(" Closed-Loop Robot Demonstration")
    print("======================================")
    print()


    # Neutral hips
    set_hips(
        fr=0.0,
        fl=0.0,
        br=0.0,
        bl=0.0
    )

    stop()


    # --------------------------------------------------------
    # Settle
    # --------------------------------------------------------

    print("Settling...")

    simulate_for(
        viewer,
        2.0
    )


    # --------------------------------------------------------
    # Sensor check
    # --------------------------------------------------------

    position = get_position()

    roll, pitch, yaw = get_orientation()

    print()
    print("Initial sensor readings:")

    print(
        f"Position: "
        f"x={position[0]:.3f}, "
        f"y={position[1]:.3f}, "
        f"z={position[2]:.3f} m"
    )

    print(
        f"Orientation: "
        f"roll={math.degrees(roll):.2f}, "
        f"pitch={math.degrees(pitch):.2f}, "
        f"yaw={math.degrees(yaw):.2f} deg"
    )

    print()


    # --------------------------------------------------------
    # Drive forward briefly
    # --------------------------------------------------------

    print("Driving forward...")

    drive(2.0)

    simulate_for(
        viewer,
        2.0
    )

    stop()

    simulate_for(
        viewer,
        1.0
    )


    # --------------------------------------------------------
    # Closed-loop 90 degree turn
    # --------------------------------------------------------

    turn_to_relative_angle(
        viewer,
        angle_degrees=90,
        max_speed=2.0,
        kp=2.0,
        tolerance_degrees=2.0
    )


    simulate_for(
        viewer,
        1.0
    )


    # --------------------------------------------------------
    # Print final state
    # --------------------------------------------------------

    position = get_position()

    roll, pitch, yaw = get_orientation()

    print()

    print(
        f"Final position: "
        f"x={position[0]:.3f}, "
        f"y={position[1]:.3f}, "
        f"z={position[2]:.3f} m"
    )

    print(
        f"Final yaw: "
        f"{math.degrees(yaw):.2f} degrees"
    )

    print()

    print("Demo complete.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading MuJoCo robot...")

    with mujoco.viewer.launch_passive(
        model,
        data
    ) as viewer:

        run_demo(viewer)

        print()
        print(
            "Close the MuJoCo viewer "
            "to exit."
        )


        while viewer.is_running():

            step_sim(viewer)


if __name__ == "__main__":
    main()