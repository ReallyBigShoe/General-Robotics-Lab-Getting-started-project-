import time
import math

import numpy as np
import mujoco
import mujoco.viewer


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "model.xml"


# ============================================================
# BACKWARD GAIT
# ============================================================
#
# Diagnostic data established:
#
# BR single stroke:
#     -40 -> +70
#
# BL single stroke:
#     -40 -> +70
#
# Both produced backward motion.
#
# Their yaw effects were approximately opposite, which is
# exactly what we want from an alternating gait.
#
# ============================================================

BR_START_DEG = -40.0
BR_END_DEG = +70.0

BL_START_DEG = -40.0
BL_END_DEG = +70.0


# ============================================================
# FRONT SUPPORT
# ============================================================

FR_SUPPORT_DEG = 0.0
FL_SUPPORT_DEG = 0.0


# ============================================================
# TIMING
# ============================================================
#
# Same propulsion timing that worked in the forward gait.
#
# One half-cycle:
#
#     0.50 s
#
# Recovery occurs simultaneously and is completed faster.
# ============================================================

THRUST_TIME = 0.50
RECOVERY_TIME = 0.35

HALF_CYCLE_TIME = 0.50


# ============================================================
# STARTUP
# ============================================================

INITIAL_SETTLE_TIME = 0.40
INITIAL_SETUP_TIME = 0.40


# ============================================================
# WHEEL BRAKING
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

def get_wheel_angle(
    wheel
):

    return data.qpos[
        wheel_qpos_adr[
            wheel
        ]
    ]


def get_wheel_velocity(
    wheel
):

    return data.qvel[
        wheel_dof_adr[
            wheel
        ]
    ]


# ============================================================
# WHEEL BRAKES
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
            wheel_hold_angle[wheel]
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

def read_sensor(
    index
):

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

        contact = data.contact[
            i
        ]


        if contact.geom1 == floor_geom:

            other = contact.geom2


        elif contact.geom2 == floor_geom:

            other = contact.geom1


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

    start_time = data.time


    while (
        viewer.is_running()
        and
        data.time - start_time < duration
    ):

        step_sim(
            viewer
        )


# ============================================================
# INTERPOLATION
# ============================================================

def smoothstep(
    t
):

    t = max(
        0.0,
        min(
            1.0,
            t
        )
    )


    return (
        t * t
        * (
            3.0
            - 2.0 * t
        )
    )


def interpolate(
    start,
    end,
    fraction
):

    return (
        start
        + (
            end
            - start
        )
        * smoothstep(
            fraction
        )
    )


# ============================================================
# SETUP
# ============================================================

def move_to_start(
    viewer
):

    print(
        "Moving into initial backward gait..."
    )


    # --------------------------------------------------------
    # Start:
    #
    # BR at its experimentally validated pre-thrust angle.
    #
    # BL at its opposite-phase starting angle.
    #
    # Front legs support.
    # --------------------------------------------------------

    target = (

        math.radians(
            FR_SUPPORT_DEG
        ),

        math.radians(
            FL_SUPPORT_DEG
        ),

        math.radians(
            BR_START_DEG
        ),

        math.radians(
            BL_END_DEG
        )
    )


    start = (
        data.ctrl[
            hip_act["FR"]
        ],

        data.ctrl[
            hip_act["FL"]
        ],

        data.ctrl[
            hip_act["BR"]
        ],

        data.ctrl[
            hip_act["BL"]
        ]
    )


    t0 = data.time


    while viewer.is_running():

        elapsed = (
            data.time
            - t0
        )


        if elapsed >= INITIAL_SETUP_TIME:

            break


        fraction = smoothstep(
            elapsed
            / INITIAL_SETUP_TIME
        )


        values = [

            start[i]
            + fraction
            * (
                target[i]
                - start[i]
            )

            for i in range(4)
        ]


        set_hips(
            values[0],
            values[1],
            values[2],
            values[3]
        )


        step_sim(
            viewer
        )


    set_hips(
        *target
    )


# ============================================================
# BR THRUST / BL RECOVERY
# ============================================================

def br_thrust_bl_recover(
    viewer
):

    t0 = data.time


    br_start = math.radians(
        BR_START_DEG
    )

    br_end = math.radians(
        BR_END_DEG
    )


    bl_start = math.radians(
        BL_END_DEG
    )

    bl_end = math.radians(
        BL_START_DEG
    )


    fr = math.radians(
        FR_SUPPORT_DEG
    )

    fl = math.radians(
        FL_SUPPORT_DEG
    )


    while viewer.is_running():

        elapsed = (
            data.time
            - t0
        )


        if elapsed >= HALF_CYCLE_TIME:

            break


        br = interpolate(
            br_start,
            br_end,
            elapsed
            / THRUST_TIME
        )


        bl = interpolate(
            bl_start,
            bl_end,
            elapsed
            / RECOVERY_TIME
        )


        set_hips(
            fr,
            fl,
            br,
            bl
        )


        step_sim(
            viewer
        )


    set_hips(
        fr,
        fl,
        br_end,
        bl_end
    )


# ============================================================
# BL THRUST / BR RECOVERY
# ============================================================

def bl_thrust_br_recover(
    viewer
):

    t0 = data.time


    # --------------------------------------------------------
    # BL now performs the reverse of its experimentally
    # verified stroke:
    #
    # -40 -> +70
    #
    # --------------------------------------------------------

    bl_start = math.radians(
        BL_START_DEG
    )

    bl_end = math.radians(
        BL_END_DEG
    )


    # --------------------------------------------------------
    # BR simultaneously recovers:
    #
    # +70 -> -40
    # --------------------------------------------------------

    br_start = math.radians(
        BR_END_DEG
    )

    br_end = math.radians(
        BR_START_DEG
    )


    fr = math.radians(
        FR_SUPPORT_DEG
    )

    fl = math.radians(
        FL_SUPPORT_DEG
    )


    while viewer.is_running():

        elapsed = (
            data.time
            - t0
        )


        if elapsed >= HALF_CYCLE_TIME:

            break


        bl = interpolate(
            bl_start,
            bl_end,
            elapsed
            / THRUST_TIME
        )


        br = interpolate(
            br_start,
            br_end,
            elapsed
            / RECOVERY_TIME
        )


        set_hips(
            fr,
            fl,
            br,
            bl
        )


        step_sim(
            viewer
        )


    set_hips(
        fr,
        fl,
        br_end,
        bl_end
    )


# ============================================================
# REPORT
# ============================================================

def report(
    number,
    label,
    start_position,
    end_position,
    start_yaw,
    end_yaw
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


    forward = (
        -s * dx
        + c * dy
    )


    lateral = (
        c * dx
        + s * dy
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


    print()
    print(
        "------------------------------------------"
    )


    print(
        f"HALF-CYCLE {number}: {label}"
    )


    print(
        f"Backward progress: "
        f"{-forward*1000:+.2f} mm"
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
        "------------------------------------------"
    )


# ============================================================
# CONTINUOUS BACKWARD WALK
# ============================================================

def run_backward_walk(
    viewer
):

    print()
    print(
        "=========================================="
    )

    print(
        " CONTINUOUS BACKWARD WALK"
    )

    print(
        "=========================================="
    )

    print()


    print(
        "BR thrust + BL recovery"
    )

    print(
        "BL thrust + BR recovery"
    )

    print()


    # --------------------------------------------------------
    # Set initial gait phase.
    # --------------------------------------------------------

    set_hips(
        0.0,
        0.0,
        0.0,
        0.0
    )


    simulate_for(
        viewer,
        INITIAL_SETTLE_TIME
    )


    move_to_start(
        viewer
    )


    print()
    print(
        "BACKWARD GAIT STARTED."
    )

    print(
        "Close MuJoCo to stop."
    )

    print()


    half_cycle = 0


    total_start = (
        get_position()
    )


    # ========================================================
    # CONTINUOUS LOOP
    # ========================================================

    while viewer.is_running():


        # ====================================================
        # BR THRUST
        # BL RECOVERY
        # ====================================================

        half_cycle += 1


        start_position = (
            get_position()
        )

        start_yaw = (
            get_yaw()
        )


        br_thrust_bl_recover(
            viewer
        )


        if not viewer.is_running():

            break


        end_position = (
            get_position()
        )

        end_yaw = (
            get_yaw()
        )


        report(
            half_cycle,
            "BR THRUST / BL RECOVERY",
            start_position,
            end_position,
            start_yaw,
            end_yaw
        )


        # ====================================================
        # BL THRUST
        # BR RECOVERY
        # ====================================================

        half_cycle += 1


        start_position = (
            get_position()
        )

        start_yaw = (
            get_yaw()
        )


        bl_thrust_br_recover(
            viewer
        )


        if not viewer.is_running():

            break


        end_position = (
            get_position()
        )

        end_yaw = (
            get_yaw()
        )


        report(
            half_cycle,
            "BL THRUST / BR RECOVERY",
            start_position,
            end_position,
            start_yaw,
            end_yaw
        )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    final_position = (
        get_position()
    )


    delta = (
        final_position
        - total_start
    )


    print()
    print(
        "=========================================="
    )

    print(
        " BACKWARD WALK STOPPED"
    )

    print(
        "=========================================="
    )


    print(
        f"Half-cycles completed: "
        f"{half_cycle}"
    )


    print(
        f"World dX: "
        f"{delta[0]*1000:+.2f} mm"
    )


    print(
        f"World dY: "
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
        # Passive settling before wheel braking.
        # ----------------------------------------------------

        for wheel in wheel_act:

            data.ctrl[
                wheel_act[
                    wheel
                ]
            ] = 0.0


        for _ in range(1000):

            mujoco.mj_step(
                model,
                data
            )


        # ----------------------------------------------------
        # Capture wheel positions.
        # ----------------------------------------------------

        capture_wheel_hold_angles()


        # ----------------------------------------------------
        # Run continuous backward gait.
        # ----------------------------------------------------

        run_backward_walk(
            viewer
        )


if __name__ == "__main__":

    main()