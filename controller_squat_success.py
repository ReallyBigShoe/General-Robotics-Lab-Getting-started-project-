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
# ID HELPERS
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
position_sensor_id = sensor_id("robot_position")


# ============================================================
# SENSOR HELPERS
# ============================================================

def read_sensor(sensor_index):

    start = model.sensor_adr[sensor_index]
    dimension = model.sensor_dim[sensor_index]

    return data.sensordata[
        start:start + dimension
    ].copy()


def get_position():

    return read_sensor(position_sensor_id)


def quaternion_to_euler(quat):

    w, x, y, z = quat

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)

    roll = math.atan2(
        sinr_cosp,
        cosr_cosp
    )

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))

    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

    yaw = math.atan2(
        siny_cosp,
        cosy_cosp
    )

    return roll, pitch, yaw


def get_orientation():

    quat = read_sensor(
        orientation_sensor_id
    )

    return quaternion_to_euler(quat)


# ============================================================
# WHEEL CONTROL
# ============================================================

def set_wheels(fr, fl, br, bl):

    data.ctrl[wheel_fr_id] = fr
    data.ctrl[wheel_fl_id] = fl
    data.ctrl[wheel_br_id] = br
    data.ctrl[wheel_bl_id] = bl


def stop_wheels():

    set_wheels(
        0.0,
        0.0,
        0.0,
        0.0
    )


# ============================================================
# HIP CONTROL
# ============================================================

def set_hips(fr, fl, br, bl):
    """
    Set hip position targets in radians.
    """

    data.ctrl[hip_fr_id] = fr
    data.ctrl[hip_fl_id] = fl
    data.ctrl[hip_br_id] = br
    data.ctrl[hip_bl_id] = bl


def neutral_stance():

    set_hips(
        fr=0.0,
        fl=0.0,
        br=0.0,
        bl=0.0
    )


def symmetric_stance(angle_degrees):
    """
    Move all four legs symmetrically.

    Because the front and back legs point in opposite
    directions in the physical geometry, the signs are
    opposite front-to-back.

    Positive angle_degrees increases the amount of
    symmetric leg articulation.
    """

    angle = math.radians(
        angle_degrees
    )

    set_hips(
        fr=angle,
        fl=angle,
        br=-angle,
        bl=-angle
    )


# ============================================================
# SIMULATION
# ============================================================

def step_sim(viewer):

    start = time.time()

    mujoco.mj_step(
        model,
        data
    )

    viewer.sync()

    elapsed = time.time() - start

    remaining = (
        model.opt.timestep
        - elapsed
    )

    if remaining > 0:

        time.sleep(
            remaining
        )


def simulate_for(viewer, duration):

    start_time = data.time

    while (
        viewer.is_running()
        and data.time - start_time < duration
    ):

        step_sim(viewer)


# ============================================================
# SMOOTH HIP MOVEMENT
# ============================================================

def move_hips_smoothly(
    viewer,
    start_angle_deg,
    end_angle_deg,
    duration
):
    """
    Interpolate the symmetric hip angle gradually.

    This prevents us from suddenly commanding a large
    position jump and kicking the robot.
    """

    start_time = data.time

    while viewer.is_running():

        elapsed = (
            data.time
            - start_time
        )

        if elapsed >= duration:
            break

        fraction = (
            elapsed
            / duration
        )

        angle = (
            start_angle_deg
            + fraction
            * (
                end_angle_deg
                - start_angle_deg
            )
        )

        symmetric_stance(
            angle
        )

        step_sim(viewer)

    symmetric_stance(
        end_angle_deg
    )


# ============================================================
# REPORT ROBOT STATE
# ============================================================

def report_state(label):

    position = get_position()

    roll, pitch, yaw = (
        get_orientation()
    )

    print()
    print(label)

    print(
        f"Height: "
        f"{position[2] * 1000:.1f} mm"
    )

    print(
        f"Roll: "
        f"{math.degrees(roll):.2f} deg"
    )

    print(
        f"Pitch: "
        f"{math.degrees(pitch):.2f} deg"
    )

    print(
        f"Yaw: "
        f"{math.degrees(yaw):.2f} deg"
    )


# ============================================================
# LEG EXPERIMENT
# ============================================================

def run_leg_experiment(viewer):

    print()
    print(
        "======================================"
    )
    print(
        " Wheel-Leg Morphology Test"
    )
    print(
        "======================================"
    )
    print()


    # --------------------------------------------------------
    # Neutral starting configuration
    # --------------------------------------------------------

    stop_wheels()
    neutral_stance()

    print(
        "Settling in neutral stance..."
    )

    simulate_for(
        viewer,
        2.0
    )

    report_state(
        "Neutral stance:"
    )


    # --------------------------------------------------------
    # First conservative test: 10 degrees
    # --------------------------------------------------------

    print()
    print(
        "Moving hips to 10 degrees..."
    )

    move_hips_smoothly(
        viewer,
        start_angle_deg=0,
        end_angle_deg=10,
        duration=2.0
    )

    simulate_for(
        viewer,
        1.0
    )

    report_state(
        "10-degree stance:"
    )


    # --------------------------------------------------------
    # Increase to 20 degrees
    # --------------------------------------------------------

    print()
    print(
        "Moving hips to 20 degrees..."
    )

    move_hips_smoothly(
        viewer,
        start_angle_deg=10,
        end_angle_deg=20,
        duration=2.0
    )

    simulate_for(
        viewer,
        1.0
    )

    report_state(
        "20-degree stance:"
    )


    # --------------------------------------------------------
    # Return to neutral
    # --------------------------------------------------------

    print()
    print(
        "Returning to neutral..."
    )

    move_hips_smoothly(
        viewer,
        start_angle_deg=20,
        end_angle_deg=0,
        duration=2.0
    )

    simulate_for(
        viewer,
        1.0
    )

    report_state(
        "Returned neutral stance:"
    )


    print()
    print(
        "Leg experiment complete."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Loading wheel-leg robot..."
    )

    with mujoco.viewer.launch_passive(
        model,
        data
    ) as viewer:

        run_leg_experiment(
            viewer
        )

        print()
        print(
            "Close the MuJoCo viewer "
            "to exit."
        )

        while viewer.is_running():

            step_sim(viewer)


if __name__ == "__main__":

    main()