import time
import math

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "model.xml"

# Search range for the two independent body-lean components.
SEARCH_LIMIT_DEG = 10.0
SEARCH_STEP_DEG = 2.0

# How long to let each candidate stance settle.
SETTLE_TIME = 0.7

# How long to average the contact force after settling.
MEASUREMENT_TIME = 0.2

# Minimum FR force fraction considered "unloaded enough".
UNLOAD_FRACTION = 0.08

# Conservative FR lift attempt.
LIFT_TEST_DEG = 12.0


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


def geom_id(name):
    return mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
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

wheel_fr_act = actuator_id("wheel_motor_FR")
wheel_fl_act = actuator_id("wheel_motor_FL")
wheel_br_act = actuator_id("wheel_motor_BR")
wheel_bl_act = actuator_id("wheel_motor_BL")

hip_fr_act = actuator_id("hip_motor_FR")
hip_fl_act = actuator_id("hip_motor_FL")
hip_br_act = actuator_id("hip_motor_BR")
hip_bl_act = actuator_id("hip_motor_BL")


# ============================================================
# COLLISION GEOM IDS
# ============================================================

wheel_geom_ids = {
    "FR": geom_id("wheel_FR_collision"),
    "FL": geom_id("wheel_FL_collision"),
    "BR": geom_id("wheel_BR_collision"),
    "BL": geom_id("wheel_BL_collision"),
}

floor_geom_id = geom_id("floor")


# ============================================================
# SENSOR IDS
# ============================================================

orientation_sensor_id = sensor_id("imu_orientation")
position_sensor_id = sensor_id("robot_position")


# ============================================================
# SENSOR READING
# ============================================================

def read_sensor(sensor_index):

    start = model.sensor_adr[sensor_index]
    dimension = model.sensor_dim[sensor_index]

    return data.sensordata[
        start:start + dimension
    ].copy()


def get_position():

    return read_sensor(
        position_sensor_id
    )


def quaternion_to_euler(quat):

    w, x, y, z = quat

    sinr_cosp = 2.0 * (
        w * x + y * z
    )

    cosr_cosp = 1.0 - 2.0 * (
        x * x + y * y
    )

    roll = math.atan2(
        sinr_cosp,
        cosr_cosp
    )

    sinp = 2.0 * (
        w * y - z * x
    )

    sinp = max(
        -1.0,
        min(1.0, sinp)
    )

    pitch = math.asin(
        sinp
    )

    siny_cosp = 2.0 * (
        w * z + x * y
    )

    cosy_cosp = 1.0 - 2.0 * (
        y * y + z * z
    )

    yaw = math.atan2(
        siny_cosp,
        cosy_cosp
    )

    return roll, pitch, yaw


def get_orientation():

    quat = read_sensor(
        orientation_sensor_id
    )

    return quaternion_to_euler(
        quat
    )


# ============================================================
# BASIC CONTROL
# ============================================================

def set_wheels(fr, fl, br, bl):

    data.ctrl[wheel_fr_act] = fr
    data.ctrl[wheel_fl_act] = fl
    data.ctrl[wheel_br_act] = br
    data.ctrl[wheel_bl_act] = bl


def stop_wheels():

    set_wheels(
        0.0,
        0.0,
        0.0,
        0.0
    )


def set_hips(fr, fl, br, bl):

    data.ctrl[hip_fr_act] = fr
    data.ctrl[hip_fl_act] = fl
    data.ctrl[hip_br_act] = br
    data.ctrl[hip_bl_act] = bl


def neutral_stance():

    set_hips(
        0.0,
        0.0,
        0.0,
        0.0
    )


# ============================================================
# COMBINED LEAN MODEL
# ============================================================

def combined_lean_targets(
    lateral_deg,
    longitudinal_deg
):
    """
    Construct a combined body-lean stance.

    Lateral component:
        FR +L
        FL -L
        BR -L
        BL +L

    Longitudinal component:
        FR +P
        FL +P
        BR -P
        BL -P

    Combined:
        FR = L + P
        FL = -L + P
        BR = -L - P
        BL = L - P
    """

    lateral = math.radians(
        lateral_deg
    )

    longitudinal = math.radians(
        longitudinal_deg
    )

    fr = lateral + longitudinal
    fl = -lateral + longitudinal
    br = -lateral - longitudinal
    bl = lateral - longitudinal

    return (
        fr,
        fl,
        br,
        bl
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


def simulate_for(
    viewer,
    duration
):

    start_time = data.time

    while (
        viewer.is_running()
        and data.time - start_time < duration
    ):

        step_sim(viewer)


# ============================================================
# HIP INTERPOLATION
# ============================================================

def move_hips_smoothly(
    viewer,
    start_targets,
    end_targets,
    duration
):

    start_time = data.time

    while viewer.is_running():

        elapsed = (
            data.time
            - start_time
        )

        if elapsed >= duration:
            break

        fraction = (
            elapsed / duration
        )

        targets = [
            start_targets[i]
            + fraction
            * (
                end_targets[i]
                - start_targets[i]
            )
            for i in range(4)
        ]

        set_hips(
            targets[0],
            targets[1],
            targets[2],
            targets[3]
        )

        step_sim(viewer)

    set_hips(
        end_targets[0],
        end_targets[1],
        end_targets[2],
        end_targets[3]
    )


def go_to_stance(
    viewer,
    lateral_deg,
    longitudinal_deg,
    duration=0.5
):

    target = combined_lean_targets(
        lateral_deg,
        longitudinal_deg
    )

    current = (
        data.ctrl[hip_fr_act],
        data.ctrl[hip_fl_act],
        data.ctrl[hip_br_act],
        data.ctrl[hip_bl_act]
    )

    move_hips_smoothly(
        viewer,
        current,
        target,
        duration
    )


# ============================================================
# CONTACT FORCE
# ============================================================

def get_wheel_normal_forces():

    forces = {
        "FR": 0.0,
        "FL": 0.0,
        "BR": 0.0,
        "BL": 0.0,
    }

    contact_force = np.zeros(6)

    for i in range(data.ncon):

        contact = data.contact[i]

        geom1 = contact.geom1
        geom2 = contact.geom2

        if geom1 == floor_geom_id:

            other_geom = geom2

        elif geom2 == floor_geom_id:

            other_geom = geom1

        else:

            continue

        mujoco.mj_contactForce(
            model,
            data,
            i,
            contact_force
        )

        normal_force = abs(
            contact_force[0]
        )

        for wheel_name, wheel_id in wheel_geom_ids.items():

            if other_geom == wheel_id:

                forces[wheel_name] += (
                    normal_force
                )

                break

    return forces


def average_wheel_forces(
    viewer,
    duration
):

    sums = {
        "FR": 0.0,
        "FL": 0.0,
        "BR": 0.0,
        "BL": 0.0,
    }

    samples = 0

    start_time = data.time

    while (
        viewer.is_running()
        and data.time - start_time < duration
    ):

        forces = get_wheel_normal_forces()

        for wheel in sums:

            sums[wheel] += forces[wheel]

        samples += 1

        step_sim(viewer)

    if samples == 0:

        return sums

    return {
        wheel: sums[wheel] / samples
        for wheel in sums
    }


# ============================================================
# SEARCH
# ============================================================

def search_unloading_stances(viewer):

    print()
    print(
        "=========================================="
    )
    print(
        " FR Unloading Stance Search"
    )
    print(
        "=========================================="
    )
    print()

    neutral_stance()
    stop_wheels()

    print(
        "Settling at neutral..."
    )

    simulate_for(
        viewer,
        2.0
    )

    baseline = average_wheel_forces(
        viewer,
        0.2
    )

    total = sum(
        baseline.values()
    )

    print()
    print(
        "Baseline wheel forces:"
    )

    for wheel in [
        "FL",
        "FR",
        "BL",
        "BR"
    ]:

        print(
            f"{wheel}: "
            f"{baseline[wheel]:.3f} N"
        )

    print()

    print(
        "Searching stance space..."
    )

    results = []

    values = np.arange(
        -SEARCH_LIMIT_DEG,
        SEARCH_LIMIT_DEG + 0.1,
        SEARCH_STEP_DEG
    )

    total_candidates = (
        len(values) * len(values)
    )

    candidate_number = 0

    for lateral in values:

        for longitudinal in values:

            candidate_number += 1

            # Return to neutral before each experiment.
            go_to_stance(
                viewer,
                0.0,
                0.0,
                duration=0.15
            )

            # Move into candidate stance.
            go_to_stance(
                viewer,
                float(lateral),
                float(longitudinal),
                duration=0.25
            )

            # Allow dynamics to settle.
            simulate_for(
                viewer,
                SETTLE_TIME
            )

            forces = average_wheel_forces(
                viewer,
                MEASUREMENT_TIME
            )

            results.append(
                (
                    forces["FR"],
                    float(lateral),
                    float(longitudinal),
                    forces
                )
            )

            print(
                f"[{candidate_number:3d}/"
                f"{total_candidates}] "
                f"L={lateral:+5.1f} "
                f"P={longitudinal:+5.1f} "
                f"FR={forces['FR']:6.3f} N"
            )

    # Sort by lowest FR load.
    results.sort(
        key=lambda item: item[0]
    )

    # Return to neutral.
    go_to_stance(
        viewer,
        0.0,
        0.0,
        duration=0.5
    )

    simulate_for(
        viewer,
        0.8
    )

    return baseline, results


# ============================================================
# REPORT SEARCH RESULTS
# ============================================================

def report_best_stances(
    results,
    number=10
):

    print()
    print(
        "=========================================="
    )
    print(
        " Best FR-Unloading Configurations"
    )
    print(
        "=========================================="
    )
    print()

    for i, result in enumerate(
        results[:number],
        start=1
    ):

        fr_force, lateral, longitudinal, forces = result

        print(
            f"{i:2d}. "
            f"L={lateral:+5.1f} deg, "
            f"P={longitudinal:+5.1f} deg | "
            f"FR={forces['FR']:6.3f} N | "
            f"FL={forces['FL']:6.3f} N | "
            f"BL={forces['BL']:6.3f} N | "
            f"BR={forces['BR']:6.3f} N"
        )


# ============================================================
# SINGLE LEG LIFT TEST
# ============================================================

def test_fr_lift(
    viewer,
    lateral_deg,
    longitudinal_deg,
    lift_delta_deg
):

    print()
    print(
        "=========================================="
    )
    print(
        " FR Lift Test"
    )
    print(
        "=========================================="
    )
    print()

    print(
        f"Using stance: "
        f"L={lateral_deg:+.1f} deg, "
        f"P={longitudinal_deg:+.1f} deg"
    )

    # Go to optimal unloading stance.
    go_to_stance(
        viewer,
        lateral_deg,
        longitudinal_deg,
        duration=1.0
    )

    simulate_for(
        viewer,
        1.5
    )

    before = average_wheel_forces(
        viewer,
        0.2
    )

    print()
    print(
        "Wheel loads before lift:"
    )

    for wheel in [
        "FL",
        "FR",
        "BL",
        "BR"
    ]:

        print(
            f"{wheel}: "
            f"{before[wheel]:.3f} N"
        )


    # --------------------------------------------------------
    # Test requested lift direction.
    # --------------------------------------------------------

    base = combined_lean_targets(
        lateral_deg,
        longitudinal_deg
    )

    lift_targets = list(base)

    # FR is target 0.
    lift_targets[0] += math.radians(
        lift_delta_deg
    )

    print()
    print(
        f"Testing FR change of "
        f"{lift_delta_deg:+.1f} degrees..."
    )

    current = (
        data.ctrl[hip_fr_act],
        data.ctrl[hip_fl_act],
        data.ctrl[hip_br_act],
        data.ctrl[hip_bl_act]
    )

    move_hips_smoothly(
        viewer,
        current,
        tuple(lift_targets),
        duration=1.5
    )

    simulate_for(
        viewer,
        1.5
    )

    after = average_wheel_forces(
        viewer,
        0.4
    )

    print()
    print(
        "Wheel loads after lift command:"
    )

    for wheel in [
        "FL",
        "FR",
        "BL",
        "BR"
    ]:

        print(
            f"{wheel}: "
            f"{after[wheel]:.3f} N"
        )

    # Restore the best stance.
    go_to_stance(
        viewer,
        lateral_deg,
        longitudinal_deg,
        duration=1.0
    )

    simulate_for(
        viewer,
        0.8
    )

    return before, after


# ============================================================
# TEST BOTH POSSIBLE FR DIRECTIONS
# ============================================================

def test_both_lift_directions(
    viewer,
    lateral_deg,
    longitudinal_deg
):

    print()
    print(
        "=========================================="
    )
    print(
        " Testing Both FR Motion Directions"
    )
    print(
        "=========================================="
    )

    positive_before, positive_after = (
        test_fr_lift(
            viewer,
            lateral_deg,
            longitudinal_deg,
            +LIFT_TEST_DEG
        )
    )

    negative_before, negative_after = (
        test_fr_lift(
            viewer,
            lateral_deg,
            longitudinal_deg,
            -LIFT_TEST_DEG
        )
    )

    positive_force = positive_after["FR"]
    negative_force = negative_after["FR"]

    print()
    print(
        "=========================================="
    )
    print(
        " Lift Direction Comparison"
    )
    print(
        "=========================================="
    )
    print()

    print(
        f"+{LIFT_TEST_DEG:.1f} deg -> "
        f"FR load = "
        f"{positive_force:.3f} N"
    )

    print(
        f"-{LIFT_TEST_DEG:.1f} deg -> "
        f"FR load = "
        f"{negative_force:.3f} N"
    )

    if positive_force < negative_force:

        best_direction = +1

        best_force = positive_force

    else:

        best_direction = -1

        best_force = negative_force

    print()

    print(
        f"Better unloading direction: "
        f"{best_direction:+d}"
    )

    print(
        f"Lowest observed FR load: "
        f"{best_force:.3f} N"
    )

    return best_direction, best_force


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment(viewer):

    baseline, results = (
        search_unloading_stances(
            viewer
        )
    )

    report_best_stances(
        results,
        number=10
    )

    best_fr_force = results[0][0]
    best_lateral = results[0][1]
    best_longitudinal = results[0][2]

    total_baseline = sum(
        baseline.values()
    )

    threshold = (
        UNLOAD_FRACTION
        * total_baseline
    )

    print()
    print(
        "Best configuration:"
    )

    print(
        f"Lateral: "
        f"{best_lateral:+.1f} deg"
    )

    print(
        f"Longitudinal: "
        f"{best_longitudinal:+.1f} deg"
    )

    print(
        f"FR load: "
        f"{best_fr_force:.3f} N"
    )

    print(
        f"Unload threshold: "
        f"{threshold:.3f} N"
    )

    # --------------------------------------------------------
    # If the search already found a very low FR load,
    # test lifting.
    # --------------------------------------------------------

    best_direction, best_force = (
        test_both_lift_directions(
            viewer,
            best_lateral,
            best_longitudinal
        )
    )

    # --------------------------------------------------------
    # Attempt a larger lift in the better direction.
    # --------------------------------------------------------

    if best_force <= threshold:

        print()
        print(
            "FR is sufficiently unloaded."
        )

        print(
            "Attempting a larger controlled lift..."
        )

        base = combined_lean_targets(
            best_lateral,
            best_longitudinal
        )

        lift_targets = list(base)

        lift_targets[0] += (
            best_direction
            * math.radians(
                LIFT_TEST_DEG
            )
        )

        current = (
            data.ctrl[hip_fr_act],
            data.ctrl[hip_fl_act],
            data.ctrl[hip_br_act],
            data.ctrl[hip_bl_act]
        )

        move_hips_smoothly(
            viewer,
            current,
            tuple(lift_targets),
            duration=2.0
        )

        simulate_for(
            viewer,
            2.0
        )

        final_forces = (
            average_wheel_forces(
                viewer,
                0.5
            )
        )

        print()
        print(
            "Final three-point-support test:"
        )

        for wheel in [
            "FL",
            "FR",
            "BL",
            "BR"
        ]:

            print(
                f"{wheel}: "
                f"{final_forces[wheel]:.3f} N"
            )

        position = get_position()

        roll, pitch, yaw = (
            get_orientation()
        )

        print()
        print(
            f"Body height: "
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

        # Safely restore neutral.
        go_to_stance(
            viewer,
            0.0,
            0.0,
            duration=2.0
        )

        simulate_for(
            viewer,
            1.0
        )

    else:

        print()
        print(
            "FR was not sufficiently unloaded "
            "by the searched stance."
        )

        print(
            "We should analyze the best configuration "
            "before attempting a physical lift."
        )

        go_to_stance(
            viewer,
            0.0,
            0.0,
            duration=1.0
        )

        simulate_for(
            viewer,
            1.0
        )

    print()
    print(
        "=========================================="
    )
    print(
        " Experiment complete."
    )
    print(
        "=========================================="
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

        run_experiment(
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