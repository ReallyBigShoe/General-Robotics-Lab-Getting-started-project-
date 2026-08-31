import time
import math

import numpy as np
import mujoco
import mujoco.viewer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "model.xml"


# ============================================================
# SUPPORTING LEG POSTURE
# ============================================================

FL_SUPPORT_DEG = -35.0
BR_SUPPORT_DEG = -2.4
BL_SUPPORT_DEG = -5.4


# ============================================================
# FR WALKING STRUT
# ============================================================
#
# This is the exact motion that visually worked:
#
#     +70 deg
#        |
#        |  THRUST
#        v
#     -40 deg
#        |
#        |  FAST RESET
#        v
#     +70 deg
#
# Then repeat continuously.
# ============================================================

FR_FORWARD_DEG = 70.0

FR_PUSH_END_DEG = -40.0


# ============================================================
# TIMING
# ============================================================

INITIAL_SETTLE_TIME = 0.5

SUPPORT_SETUP_TIME = 0.8

# Initial move into +70 position
FR_INITIAL_PLACEMENT_TIME = 0.35

# Exact successful thrust timing
FR_PUSH_TIME = 0.50

# Exact successful reset timing
FR_RESET_TIME = 0.35

# Very short pause before beginning the next thrust.
CYCLE_PAUSE_TIME = 0.08


# ============================================================
# WHEEL BRAKE
# ============================================================

BRAKE_KP = 1.5
BRAKE_KD = 0.12

BRAKE_MAX_TORQUE = 0.50


# ============================================================
# LOAD MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(
    MODEL_PATH
)

data = mujoco.MjData(
    model
)


# ============================================================
# ID HELPERS
# ============================================================

def actuator_id(name):

    return mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name
    )


def joint_id(name):

    return mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
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
# ACTUATORS
# ============================================================

hip_act = {

    "FR": actuator_id(
        "hip_motor_FR"
    ),

    "FL": actuator_id(
        "hip_motor_FL"
    ),

    "BR": actuator_id(
        "hip_motor_BR"
    ),

    "BL": actuator_id(
        "hip_motor_BL"
    ),
}


wheel_act = {

    "FR": actuator_id(
        "wheel_motor_FR"
    ),

    "FL": actuator_id(
        "wheel_motor_FL"
    ),

    "BR": actuator_id(
        "wheel_motor_BR"
    ),

    "BL": actuator_id(
        "wheel_motor_BL"
    ),
}


# ============================================================
# HIP JOINTS
# ============================================================

hip_joint = {

    "FR": joint_id(
        "hip_FR"
    ),

    "FL": joint_id(
        "hip_FL"
    ),

    "BR": joint_id(
        "hip_BR"
    ),

    "BL": joint_id(
        "hip_BL"
    ),
}


hip_qpos_adr = {

    leg: model.jnt_qposadr[jid]

    for leg, jid
    in hip_joint.items()
}


# ============================================================
# WHEEL JOINTS
# ============================================================

wheel_joint = {

    "FR": joint_id(
        "wheel_FR"
    ),

    "FL": joint_id(
        "wheel_FL"
    ),

    "BR": joint_id(
        "wheel_BR"
    ),

    "BL": joint_id(
        "wheel_BL"
    ),
}


wheel_qpos_adr = {

    wheel: model.jnt_qposadr[jid]

    for wheel, jid
    in wheel_joint.items()
}


wheel_dof_adr = {

    wheel: model.jnt_dofadr[jid]

    for wheel, jid
    in wheel_joint.items()
}


# ============================================================
# CONTACT GEOMETRY
# ============================================================

floor_geom = geom_id(
    "floor"
)


wheel_geom = {

    "FR": geom_id(
        "wheel_FR_collision"
    ),

    "FL": geom_id(
        "wheel_FL_collision"
    ),

    "BR": geom_id(
        "wheel_BR_collision"
    ),

    "BL": geom_id(
        "wheel_BL_collision"
    ),
}


# ============================================================
# SENSORS
# ============================================================

position_sensor = sensor_id(
    "robot_position"
)

orientation_sensor = sensor_id(
    "imu_orientation"
)


# ============================================================
# WHEEL HOLD STATE
# ============================================================

wheel_hold_angle = {

    "FR": 0.0,
    "FL": 0.0,
    "BR": 0.0,
    "BL": 0.0,
}


# ============================================================
# HIP CONTROL
# ============================================================

def set_hips(
    fr,
    fl,
    br,
    bl
):

    data.ctrl[
        hip_act["FR"]
    ] = fr

    data.ctrl[
        hip_act["FL"]
    ] = fl

    data.ctrl[
        hip_act["BR"]
    ] = br

    data.ctrl[
        hip_act["BL"]
    ] = bl


# ============================================================
# WHEEL STATE
# ============================================================

def get_wheel_angle(wheel):

    return data.qpos[
        wheel_qpos_adr[wheel]
    ]


def get_wheel_velocity(wheel):

    return data.qvel[
        wheel_dof_adr[wheel]
    ]


# ============================================================
# ACTIVE WHEEL BRAKES
# ============================================================

def capture_wheel_hold_angles():

    for wheel in wheel_hold_angle:

        wheel_hold_angle[
            wheel
        ] = get_wheel_angle(
            wheel
        )


def apply_wheel_brakes():

    for wheel in wheel_hold_angle:

        angle = get_wheel_angle(
            wheel
        )

        velocity = get_wheel_velocity(
            wheel
        )

        error = (
            wheel_hold_angle[
                wheel
            ]
            - angle
        )

        torque = (
            BRAKE_KP
            * error
            - BRAKE_KD
            * velocity
        )

        torque = float(
            np.clip(
                torque,
                -BRAKE_MAX_TORQUE,
                BRAKE_MAX_TORQUE
            )
        )

        data.ctrl[
            wheel_act[
                wheel
            ]
        ] = torque


# ============================================================
# SENSOR HELPERS
# ============================================================

def read_sensor(index):

    start = model.sensor_adr[
        index
    ]

    size = model.sensor_dim[
        index
    ]

    return data.sensordata[
        start:start + size
    ].copy()


def get_position():

    return read_sensor(
        position_sensor
    )


def quaternion_to_euler(quat):

    w, x, y, z = quat


    sinr_cosp = 2.0 * (
        w * x
        + y * z
    )

    cosr_cosp = 1.0 - 2.0 * (
        x * x
        + y * y
    )

    roll = math.atan2(
        sinr_cosp,
        cosr_cosp
    )


    sinp = 2.0 * (
        w * y
        - z * x
    )

    sinp = max(
        -1.0,
        min(
            1.0,
            sinp
        )
    )

    pitch = math.asin(
        sinp
    )


    siny_cosp = 2.0 * (
        w * z
        + x * y
    )

    cosy_cosp = 1.0 - 2.0 * (
        y * y
        + z * z
    )

    yaw = math.atan2(
        siny_cosp,
        cosy_cosp
    )


    return (
        roll,
        pitch,
        yaw
    )


def get_orientation():

    return quaternion_to_euler(
        read_sensor(
            orientation_sensor
        )
    )


# ============================================================
# CONTACT FORCES
# ============================================================

def get_wheel_forces():

    forces = {

        "FR": 0.0,
        "FL": 0.0,
        "BR": 0.0,
        "BL": 0.0,
    }


    wrench = np.zeros(
        6,
        dtype=np.float64
    )


    for i in range(
        data.ncon
    ):

        contact = (
            data.contact[i]
        )


        if (
            contact.geom1
            == floor_geom
        ):

            other = (
                contact.geom2
            )


        elif (
            contact.geom2
            == floor_geom
        ):

            other = (
                contact.geom1
            )


        else:

            continue


        wrench.fill(
            0.0
        )


        mujoco.mj_contactForce(
            model,
            data,
            i,
            wrench
        )


        normal_force = abs(
            wrench[0]
        )


        for (
            wheel,
            geom
        ) in wheel_geom.items():

            if other == geom:

                forces[
                    wheel
                ] += normal_force

                break


    return forces


# ============================================================
# SIMULATION
# ============================================================

def step_sim(viewer):

    apply_wheel_brakes()


    start = time.time()


    mujoco.mj_step(
        model,
        data
    )


    viewer.sync()


    elapsed = (
        time.time()
        - start
    )


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

    start_time = (
        data.time
    )


    while (
        viewer.is_running()

        and

        data.time
        - start_time
        < duration
    ):

        step_sim(
            viewer
        )


# ============================================================
# SMOOTH HIP MOTION
# ============================================================

def move_hips(
    viewer,
    start,
    end,
    duration
):

    start_time = (
        data.time
    )


    while viewer.is_running():

        elapsed = (
            data.time
            - start_time
        )


        if elapsed >= duration:

            break


        t = (
            elapsed
            / duration
        )


        # Same smoothstep interpolation as the successful test.

        t = (
            t
            * t
            * (
                3.0
                - 2.0 * t
            )
        )


        targets = [

            start[i]
            + t
            * (
                end[i]
                - start[i]
            )

            for i in range(4)
        ]


        set_hips(
            targets[0],
            targets[1],
            targets[2],
            targets[3]
        )


        step_sim(
            viewer
        )


    set_hips(
        end[0],
        end[1],
        end[2],
        end[3]
    )


# ============================================================
# REPORT
# ============================================================

def report_cycle(
    cycle_number,
    start_position,
    end_position
):

    delta = (
        end_position
        - start_position
    )


    forces = (
        get_wheel_forces()
    )


    roll, pitch, yaw = (
        get_orientation()
    )


    print()
    print(
        f"Cycle {cycle_number}:"
    )


    print(
        f"  displacement: "
        f"dX={delta[0]*1000:+.2f} mm, "
        f"dY={delta[1]*1000:+.2f} mm"
    )


    print(
        f"  attitude: "
        f"roll={math.degrees(roll):+.2f} deg, "
        f"pitch={math.degrees(pitch):+.2f} deg, "
        f"yaw={math.degrees(yaw):+.2f} deg"
    )


    print(
        f"  loads: "
        f"FL={forces['FL']:.2f} N, "
        f"FR={forces['FR']:.2f} N, "
        f"BL={forces['BL']:.2f} N, "
        f"BR={forces['BR']:.2f} N"
    )


# ============================================================
# CONTINUOUS FR WALKING STRUT
# ============================================================

def run_continuous_fr_strut(
    viewer
):

    print()
    print(
        "=========================================="
    )

    print(
        " CONTINUOUS FR WALKING STRUT"
    )

    print(
        "=========================================="
    )

    print()


    # ========================================================
    # 1. START NEUTRAL
    # ========================================================

    neutral = (
        0.0,
        0.0,
        0.0,
        0.0
    )


    set_hips(
        *neutral
    )


    simulate_for(
        viewer,
        INITIAL_SETTLE_TIME
    )


    # ========================================================
    # 2. SUPPORT POSTURE
    # ========================================================

    support = (

        0.0,

        math.radians(
            FL_SUPPORT_DEG
        ),

        math.radians(
            BR_SUPPORT_DEG
        ),

        math.radians(
            BL_SUPPORT_DEG
        )
    )


    print(
        "Establishing support posture..."
    )


    move_hips(
        viewer,
        neutral,
        support,
        SUPPORT_SETUP_TIME
    )


    simulate_for(
        viewer,
        0.15
    )


    # ========================================================
    # 3. INITIAL FR PLACEMENT
    # ========================================================

    forward = list(
        support
    )


    forward[0] = (
        math.radians(
            FR_FORWARD_DEG
        )
    )


    forward = tuple(
        forward
    )


    print(
        f"Placing FR at "
        f"+{FR_FORWARD_DEG:.1f} deg..."
    )


    move_hips(
        viewer,
        support,
        forward,
        FR_INITIAL_PLACEMENT_TIME
    )


    simulate_for(
        viewer,
        CYCLE_PAUSE_TIME
    )


    # ========================================================
    # 4. DEFINE THRUST POSITION
    # ========================================================

    pushed = list(
        forward
    )


    pushed[0] = (
        math.radians(
            FR_PUSH_END_DEG
        )
    )


    pushed = tuple(
        pushed
    )


    # ========================================================
    # 5. CONTINUOUS LOOP
    # ========================================================

    print()
    print(
        "Continuous stepping started."
    )

    print(
        "Close the MuJoCo window whenever "
        "you want to stop."
    )

    print()


    cycle_number = 0

    total_start_position = (
        get_position()
    )


    while viewer.is_running():

        cycle_number += 1


        cycle_start_position = (
            get_position()
        )


        # ----------------------------------------------------
        # PROPULSION
        #
        # +70 -> -40 in 0.50 s
        # ----------------------------------------------------

        move_hips(
            viewer,
            forward,
            pushed,
            FR_PUSH_TIME
        )


        if not viewer.is_running():
            break


        # ----------------------------------------------------
        # RESET
        #
        # -40 -> +70 in 0.35 s
        # ----------------------------------------------------

        move_hips(
            viewer,
            pushed,
            forward,
            FR_RESET_TIME
        )


        if not viewer.is_running():
            break


        # ----------------------------------------------------
        # SHORT PAUSE
        # ----------------------------------------------------

        simulate_for(
            viewer,
            CYCLE_PAUSE_TIME
        )


        # ----------------------------------------------------
        # REPORT CYCLE
        # ----------------------------------------------------

        cycle_end_position = (
            get_position()
        )


        report_cycle(
            cycle_number,
            cycle_start_position,
            cycle_end_position
        )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    total_end_position = (
        get_position()
    )


    total_delta = (
        total_end_position
        - total_start_position
    )


    print()
    print(
        "=========================================="
    )

    print(
        " CONTINUOUS RUN RESULT"
    )

    print(
        "=========================================="
    )


    print(
        f"Completed cycles: "
        f"{cycle_number}"
    )


    print(
        f"Total dX: "
        f"{total_delta[0]*1000:+.2f} mm"
    )


    print(
        f"Total dY: "
        f"{total_delta[1]*1000:+.2f} mm"
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


        # ----------------------------------------------------
        # Passive initial settling
        # ----------------------------------------------------

        for wheel in wheel_act:

            data.ctrl[
                wheel_act[wheel]
            ] = 0.0


        for _ in range(
            1000
        ):

            mujoco.mj_step(
                model,
                data
            )


        # ----------------------------------------------------
        # Lock the wheels at their settled orientations.
        # ----------------------------------------------------

        capture_wheel_hold_angles()


        # ----------------------------------------------------
        # GO
        # ----------------------------------------------------

        run_continuous_fr_strut(
            viewer
        )


if __name__ == "__main__":

    main()