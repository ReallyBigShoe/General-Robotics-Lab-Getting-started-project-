import time

import mujoco
import mujoco.viewer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "model.xml"

# Wheel actuator names from model.xml
WHEEL_FR = "wheel_motor_FR"
WHEEL_FL = "wheel_motor_FL"
WHEEL_BR = "wheel_motor_BR"
WHEEL_BL = "wheel_motor_BL"

# Hip actuator names
HIP_FR = "hip_motor_FR"
HIP_FL = "hip_motor_FL"
HIP_BR = "hip_motor_BR"
HIP_BL = "hip_motor_BL"


# ============================================================
# LOAD MUJOCO MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)


# ============================================================
# FIND ACTUATOR IDS
# ============================================================

wheel_fr_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, WHEEL_FR
)

wheel_fl_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, WHEEL_FL
)

wheel_br_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, WHEEL_BR
)

wheel_bl_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, WHEEL_BL
)

hip_fr_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, HIP_FR
)

hip_fl_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, HIP_FL
)

hip_br_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, HIP_BR
)

hip_bl_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, HIP_BL
)


# ============================================================
# HIP CONTROL
# ============================================================

def set_hips(fr=0.0, fl=0.0, br=0.0, bl=0.0):
    """
    Set the four hip target positions in radians.
    """

    data.ctrl[hip_fr_id] = fr
    data.ctrl[hip_fl_id] = fl
    data.ctrl[hip_br_id] = br
    data.ctrl[hip_bl_id] = bl


# ============================================================
# LOW-LEVEL WHEEL CONTROL
# ============================================================

def set_wheels(fr, fl, br, bl):
    """
    Directly command the four wheel angular velocities.

    Positive/negative values correspond to MuJoCo joint
    velocity commands in rad/s.
    """

    data.ctrl[wheel_fr_id] = fr
    data.ctrl[wheel_fl_id] = fl
    data.ctrl[wheel_br_id] = br
    data.ctrl[wheel_bl_id] = bl


# ============================================================
# HIGH-LEVEL DRIVE COMMANDS
# ============================================================

def drive(speed):
    """
    Drive straight.

    Positive speed = forward
    Negative speed = backward

    The physical wheel convention discovered experimentally is:
        negative MuJoCo wheel velocity = robot forward

    Therefore the command is inverted here.
    """

    wheel_speed = -speed

    set_wheels(
        wheel_speed,
        wheel_speed,
        wheel_speed,
        wheel_speed
    )


def turn_left(speed):
    """
    Rotate counterclockwise / left approximately in place.

    Right wheels drive forward.
    Left wheels drive backward.
    """

    set_wheels(
        -speed,   # FR forward
        speed,    # FL backward
        -speed,   # BR forward
        speed     # BL backward
    )


def turn_right(speed):
    """
    Rotate clockwise / right approximately in place.

    Left wheels drive forward.
    Right wheels drive backward.
    """

    set_wheels(
        speed,    # FR backward
        -speed,   # FL forward
        speed,    # BR backward
        -speed    # BL forward
    )


def stop():
    """
    Command zero wheel velocity.
    """

    set_wheels(0.0, 0.0, 0.0, 0.0)


# ============================================================
# SIMULATION HELPER
# ============================================================

def simulate_for(viewer, duration):
    """
    Advance the simulation for a specified number of
    real-time seconds while keeping the viewer synchronized.
    """

    start_sim_time = data.time

    while viewer.is_running() and data.time - start_sim_time < duration:

        step_start = time.time()

        mujoco.mj_step(model, data)

        viewer.sync()

        # Attempt to keep simulation approximately real-time.
        elapsed = time.time() - step_start
        remaining = model.opt.timestep - elapsed

        if remaining > 0:
            time.sleep(remaining)


# ============================================================
# AUTOMATIC DEMONSTRATION
# ============================================================

def run_demo(viewer):

    print()
    print("======================================")
    print(" General Robotics Robot Simulation")
    print("======================================")
    print()

    # --------------------------------------------------------
    # Neutral leg configuration
    # --------------------------------------------------------

    set_hips(
        fr=0.0,
        fl=0.0,
        br=0.0,
        bl=0.0
    )

    stop()

    print("Settling...")
    simulate_for(viewer, 2.0)


    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    print("Driving forward...")

    drive(2.0)
    simulate_for(viewer, 3.0)

    stop()
    simulate_for(viewer, 1.0)


    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    print("Driving backward...")

    drive(-2.0)
    simulate_for(viewer, 3.0)

    stop()
    simulate_for(viewer, 1.0)


    # --------------------------------------------------------
    # Left turn
    # --------------------------------------------------------

    print("Turning left...")

    turn_left(2.0)
    simulate_for(viewer, 2.0)

    stop()
    simulate_for(viewer, 1.0)


    # --------------------------------------------------------
    # Right turn
    # --------------------------------------------------------

    print("Turning right...")

    turn_right(2.0)
    simulate_for(viewer, 2.0)

    stop()
    simulate_for(viewer, 1.0)


    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("Demo complete.")
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading MuJoCo robot...")

    with mujoco.viewer.launch_passive(model, data) as viewer:

        run_demo(viewer)

        print("Simulation finished.")
        print("Close the MuJoCo window to exit.")

        # Leave the viewer open after the demonstration.
        while viewer.is_running():

            step_start = time.time()

            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.time() - step_start
            remaining = model.opt.timestep - elapsed

            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()