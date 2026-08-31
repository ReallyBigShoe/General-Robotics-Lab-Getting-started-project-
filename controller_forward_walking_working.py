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
# FRONT-LEG GAIT
# ============================================================
#
# FR is kept very close to the experimentally successful
# single-leg gait.
#
# FL is deliberately made slightly less aggressive on this
# iteration because the previous alternating gait produced
# enormous lateral excursions.
#
# We will use the per-half-cycle measurements to tune this.
# ============================================================


# -------------------------
# FRONT RIGHT
# -------------------------

FR_OUT_DEG = +70.0
FR_IN_DEG = -44.0

FR_THRUST_TIME = 0.50
FR_RECOVERY_TIME = 0.35


# -------------------------
# FRONT LEFT
# -------------------------
#
# FL has the opposite physical coordinate convention.
#
# Start slightly less aggressive than FR.
# If the resulting motion is clean, we can enlarge this later.
# -------------------------

FL_OUT_DEG = +36.0
FL_IN_DEG = -65.0

FL_THRUST_TIME = 0.50
FL_RECOVERY_TIME = 0.35


# ============================================================
# REAR LEG STABILIZATION
# ============================================================
#
# The previous gait left the rear legs essentially frozen.
#
# Here they are given small synchronized stabilizing motions.
#
# These are NOT intended to create propulsion.
#
# Their job is to oppose the large lateral body motion created
# when one front leg transitions from thrust to recovery.
# ============================================================

BR_BASE_DEG = -2.4
BL_BASE_DEG = -5.4

# Small stabilization amplitudes.
BR_STABILIZE_DEG = 4.0
BL_STABILIZE_DEG = 4.0


# ============================================================
# TIMING
# ============================================================

INITIAL_SETTLE_TIME = 0.50
INITIAL_SETUP_TIME = 0.60

HALF_CYCLE_TIME = 0.50

FINAL_SETTLE_TIME = 1.0


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

    wheel:
        model.jnt_qposadr[jid]

    for wheel, jid
    in wheel_joint.items()
}


wheel_dof_adr = {

    wheel:
        model.jnt_dofadr[jid]

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
# WHEEL BRAKE STATE
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

def get_wheel_angle(
    wheel
):

    return data.qpos[
        wheel_qpos_adr[wheel]
    ]


def get_wheel_velocity(
    wheel
):

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
            BRAKE_KP * error
            - BRAKE_KD * velocity
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


def quaternion_to_euler(
    quat
):

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


def get_yaw():

    _, _, yaw = (
        get_orientation()
    )

    return yaw


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

def step_sim(
    viewer
):

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
# SMOOTHSTEP
# ============================================================

def smoothstep(t):

    t = max(
        0.0,
        min(
            1.0,
            t
        )
    )

    return (
        t
        * t
        * (
            3.0
            - 2.0 * t
        )
    )


# ============================================================
# SIMULTANEOUS HIP MOTION
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


        t = smoothstep(
            t
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
# ROBOT-FRAME DISPLACEMENT
# ============================================================

def robot_frame_displacement(
    start_position,
    end_position,
    start_yaw
):

    dx = (
        end_position[0]
        - start_position[0]
    )

    dy = (
        end_position[1]
        - start_position[1]
    )


    c = math.cos(
        start_yaw
    )

    s = math.sin(
        start_yaw
    )


    local_x = (
        c * dx
        + s * dy
    )

    local_y = (
        -s * dx
        + c * dy
    )


    return (
        local_y,
        local_x
    )


# ============================================================
# STATE REPORT
# ============================================================

def report_half_cycle(
    label,
    start_position,
    end_position,
    start_yaw,
    end_yaw
):

    forward, lateral = (
        robot_frame_displacement(
            start_position,
            end_position,
            start_yaw
        )
    )


    yaw_delta = math.atan2(
        math.sin(
            end_yaw
            - start_yaw
        ),
        math.cos(
            end_yaw
            - start_yaw
        )
    )


    forces = (
        get_wheel_forces()
    )


    print()
    print(
        "------------------------------------------"
    )

    print(
        label
    )

    print(
        f"Forward: "
        f"{forward*1000:+.2f} mm"
    )

    print(
        f"Lateral: "
        f"{lateral*1000:+.2f} mm"
    )

    print(
        f"Yaw: "
        f"{math.degrees(yaw_delta):+.2f} deg"
    )

    print(
        f"Loads: "
        f"FL={forces['FL']:.2f} N, "
        f"FR={forces['FR']:.2f} N, "
        f"BL={forces['BL']:.2f} N, "
        f"BR={forces['BR']:.2f} N"
    )

    print(
        "------------------------------------------"
    )


# ============================================================
# CONTINUOUS ALTERNATING GAIT
# ============================================================

def run_gait(
    viewer
):

    print()
    print(
        "=========================================="
    )

    print(
        " OPTIMIZED ALTERNATING WALK"
    )

    print(
        "=========================================="
    )

    print()

    print(
        f"FR: "
        f"+{FR_OUT_DEG:.0f} -> "
        f"{FR_IN_DEG:.0f} deg"
    )

    print(
        f"FL: "
        f"{FL_OUT_DEG:.0f} -> "
        f"{FL_IN_DEG:.0f} deg"
    )

    print()

    print(
        "FR thrust + FL recovery"
    )

    print(
        "FL thrust + FR recovery"
    )

    print()


    # ========================================================
    # INITIAL STATE
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
    # INITIAL GAIT STATE
    # ========================================================

    initial_state = (

        math.radians(
            FR_OUT_DEG
        ),

        math.radians(
            FL_IN_DEG
        ),

        math.radians(
            BR_BASE_DEG
        ),

        math.radians(
            BL_BASE_DEG
        )
    )


    move_hips(
        viewer,
        neutral,
        initial_state,
        INITIAL_SETUP_TIME
    )


    simulate_for(
        viewer,
        0.15
    )


    total_start = (
        get_position()
    )


    # ========================================================
    # GAIT LOOP
    # ========================================================

    half_cycle = 0


    while viewer.is_running():

        # ====================================================
        # HALF-CYCLE A
        #
        # FR THRUSTS
        # FL RECOVERS
        #
        # REAR LEGS:
        # BL moves slightly inward while FR thrusts,
        # giving the body a small stabilizing reaction.
        # ====================================================

        half_cycle += 1


        start_position = (
            get_position()
        )

        start_yaw = (
            get_yaw()
        )


        start_state = (
            math.radians(
                FR_OUT_DEG
            ),

            math.radians(
                FL_IN_DEG
            ),

            math.radians(
                BR_BASE_DEG
            ),

            math.radians(
                BL_BASE_DEG
            )
        )


        end_state = (
            math.radians(
                FR_IN_DEG
            ),

            math.radians(
                FL_OUT_DEG
            ),

            math.radians(
                BR_BASE_DEG
                + BR_STABILIZE_DEG
            ),

            math.radians(
                BL_BASE_DEG
            )
        )


        move_hips(
            viewer,
            start_state,
            end_state,
            HALF_CYCLE_TIME
        )


        if not viewer.is_running():

            break


        end_position = (
            get_position()
        )

        end_yaw = (
            get_yaw()
        )


        report_half_cycle(
            "HALF-CYCLE "
            f"{half_cycle}: FR THRUST",
            start_position,
            end_position,
            start_yaw,
            end_yaw
        )


        # ====================================================
        # HALF-CYCLE B
        #
        # FL THRUSTS
        # FR RECOVERS
        #
        # Opposite rear-leg stabilization.
        # ====================================================

        half_cycle += 1


        start_position = (
            get_position()
        )

        start_yaw = (
            get_yaw()
        )


        start_state = (
            math.radians(
                FR_IN_DEG
            ),

            math.radians(
                FL_OUT_DEG
            ),

            math.radians(
                BR_BASE_DEG
                + BR_STABILIZE_DEG
            ),

            math.radians(
                BL_BASE_DEG
            )
        )


        end_state = (
            math.radians(
                FR_OUT_DEG
            ),

            math.radians(
                FL_IN_DEG
            ),

            math.radians(
                BR_BASE_DEG
            ),

            math.radians(
                BL_BASE_DEG
                - BL_STABILIZE_DEG
            )
        )


        move_hips(
            viewer,
            start_state,
            end_state,
            HALF_CYCLE_TIME
        )


        if not viewer.is_running():

            break


        end_position = (
            get_position()
        )

        end_yaw = (
            get_yaw()
        )


        report_half_cycle(
            "HALF-CYCLE "
            f"{half_cycle}: FL THRUST",
            start_position,
            end_position,
            start_yaw,
            end_yaw
        )


        # ====================================================
        # OCCASIONAL SUMMARY
        # ====================================================

        if (
            half_cycle % 10
            == 0
        ):

            current = (
                get_position()
            )

            delta = (
                current
                - total_start
            )

            _, _, yaw = (
                get_orientation()
            )


            print()
            print(
                "=========================================="
            )

            print(
                f"After "
                f"{half_cycle // 2} full cycles:"
            )

            print(
                f"World dX = "
                f"{delta[0]*1000:+.1f} mm"
            )

            print(
                f"World dY = "
                f"{delta[1]*1000:+.1f} mm"
            )

            print(
                f"Heading = "
                f"{math.degrees(yaw):+.1f} deg"
            )

            print(
                "=========================================="
            )


    # ========================================================
    # FINAL
    # ========================================================

    final = (
        get_position()
    )


    delta = (
        final
        - total_start
    )


    print()
    print(
        "=========================================="
    )

    print(
        " GAIT STOPPED"
    )

    print(
        "=========================================="
    )


    print(
        f"World dX = "
        f"{delta[0]*1000:+.2f} mm"
    )

    print(
        f"World dY = "
        f"{delta[1]*1000:+.2f} mm"
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
        # Passive settle
        # ----------------------------------------------------

        for wheel in wheel_act:

            data.ctrl[
                wheel_act[
                    wheel
                ]
            ] = 0.0


        for _ in range(
            1000
        ):

            mujoco.mj_step(
                model,
                data
            )


        # ----------------------------------------------------
        # Capture wheel angles and engage brakes
        # ----------------------------------------------------

        capture_wheel_hold_angles()


        # ----------------------------------------------------
        # Begin walking
        # ----------------------------------------------------

        run_gait(
            viewer
        )


if __name__ == "__main__":

    main()