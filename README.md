# General-Robotics-Lab-Getting-started-project-

# Wheel-Legged Robot Simulation

This repository contains the CAD-derived MuJoCo simulation and control experiments for a wheel-legged mobile robot I designed as a getting-started project for the Duke General Robotics Lab.

The robot combines four independently actuated wheels with four independently articulated legs. The idea is to retain the efficiency and simplicity of wheeled locomotion on ordinary terrain while gaining the ability to manipulate the robot's stance, redistribute its weight, and use its legs for locomotion when rolling alone may not be sufficient.

The robot was designed in Autodesk Fusion and translated into a physics-based MuJoCo model using physical and geometric properties extracted from the CAD assembly. I then used the simulation to progressively develop and test wheel control, body pose manipulation, load transfer, individual wheel lifting, and ultimately continuous forward and backward legged locomotion.


## Current Capabilities

The simulation currently demonstrates:

- Four-wheel forward and backward driving
- Differential wheel-based turning and in-place rotation
- Independent control of all four leg joints
- Squatting and controlled body leaning
- Wheel-ground contact force measurement
- Intentional weight transfer and wheel unloading
- Stable three-point support
- Controlled lifting, movement, and replanting of an individual wheel
- Continuous forward legged locomotion using the front legs
- Continuous backward legged locomotion using the rear legs
- Active wheel braking during legged locomotion

The forward and backward walking controllers use reciprocal strut-like motions. While one leg performs a planted propulsion stroke, its counterpart recovers for the next stroke. This behavior was developed experimentally in simulation rather than prescribed from an existing gait.


## Quick Start

### Requirements

This project was developed and tested using:

- Python 3.14.0
- MuJoCo 3.12.0
- NumPy 2.4.4

After cloning or downloading the repository, install the Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains the tested MuJoCo and NumPy versions used during development.

### Getting the Project

The easiest way to reproduce the simulations is to download the entire repository rather than individual files. This preserves the relative paths between the Python controllers, MuJoCo model, and STL assets.

Using Git:

```bash
git clone https://github.com/ReallyBigShoe/General-Robotics-Lab-Getting-started-project-.git
cd General-Robotics-Lab-Getting-started-project-
pip install -r requirements.txt
```

Alternatively, the repository can be downloaded from GitHub using **Code > Download ZIP** and extracted locally.

### Files Required to Run the Simulation

The primary runtime components are:

- `model.xml` - Current MuJoCo physics model.
- `Assets/` - STL meshes exported from the Autodesk Fusion assembly.
- `requirements.txt` - Tested Python dependencies.
- The desired Python controller.

`model.xml` references files inside `Assets/`, and the controllers reference `model.xml`, so the directory structure should be preserved.

### Primary Demonstrations

**`controller_forward_walking_working.py`**  
Continuous forward legged locomotion using the front leg pair.

**`controller_backward_walking_working.py`**  
Continuous backward legged locomotion using the rear leg pair.

**`controller_autonomous_spin.py`**  
Differential wheel control and autonomous rotation.

**`controller_squat_success.py`**  
Coordinated leg articulation and chassis-height control.

**`controller_FR_three_point_success.py`**  
Load transfer, front-right wheel unloading, and three-point support.

Other controllers and XML files in the repository document intermediate experiments and stages in the development of the final simulation.

### Running a Simulation

Run controllers from the repository root so that their relative path to `model.xml` and `Assets/` is preserved.

For the primary forward-walking demonstration:

```bash
python controller_forward_walking_working.py
```

For backward walking:

```bash
python controller_backward_walking_working.py
```

For wheel-based autonomous turning:

```bash
python controller_autonomous_spin.py
```

A MuJoCo viewer should open and execute the selected experiment. Close the viewer to end the simulation.

### Recommended First Run

To reproduce the primary legged-locomotion result, I recommend starting with:

```bash
python controller_forward_walking_working.py
```

The repository was also tested by cloning it into a fresh local directory, installing the listed dependencies, and running the forward-locomotion demonstration from the cloned copy.


## Why Simulate It?

One of the main purposes of this project was to answer questions that were difficult to answer from CAD geometry alone.

MuJoCo made it possible to observe quantities such as individual wheel loads, loss and recovery of ground contact, body attitude, support transitions, displacement, and gait-induced yaw while experimenting with different motions. Several controller ideas that appeared reasonable geometrically were unstable once gravity, contact, friction, and the robot's mass distribution were introduced.

This made simulation part of the design process rather than simply a visualization step. Unsuccessful behaviors could fail virtually, provide measurable information about *why* they failed, and inform the next iteration before anything was committed to hardware.

That became one of my favorite realizations during the project. There are quantities that would be difficult or annoying to measure on an early physical prototype that become immediately observable in simulation. Instead of building a motion, discovering that it fails, and then trying to infer why from the wreckage, I could perform much of that "evolution" virtually first.


## Robot Design

The robot is a four-wheel, four-leg mobile platform that I designed in Autodesk Fusion. Each wheel is mounted at the end of an independently articulated leg, giving the robot eight actuated degrees of freedom across its locomotion system: four wheel rotations and four leg rotations.

The motivation behind the design is to combine two useful modes of mobility rather than asking either one to do everything.

On ordinary, relatively flat terrain, the wheels provide the simplest and most efficient means of locomotion. The legs add another layer of mobility by allowing the robot to change its stance, redistribute its weight, lift individual wheels, and produce locomotion through deliberate interaction with the ground.

This led to a hybrid control philosophy throughout the project. The wheels are used when rolling is the better tool for the job, while the legs can manipulate the geometry and support state of the robot when more deliberate interaction with the environment is useful. For example, turning is currently accomplished through differential wheel motion rather than by developing a dedicated legged turning gait.


### Mechanical Design

The chassis and leg geometry were modeled as an assembly in Fusion. The leg joints were positioned and constrained according to the intended physical mechanism, including mechanical limits on their angular travel.

Rather than treating the CAD model as visual reference alone, I used it as the primary source of geometric and physical information for the simulation.

After assigning material properties to the modeled components, I extracted properties including:

- Component masses
- Centers of mass
- Moments of inertia
- Component dimensions
- Relative component positions
- Joint locations
- Joint axes
- Leg rotation limits
- Wheel and leg geometry

These values were then used to construct the corresponding MuJoCo model.

The CAD design also includes physical interference features that limit excessive leg travel. This became particularly important during simulation development because an incorrect joint range could allow a simulated leg to reach a configuration that the physical design would not permit, while an overly restrictive range could artificially prevent a valid motion. The joint limits in the final model were therefore checked against the Fusion assembly as the locomotion controllers were developed.


### Wheel-Leg Architecture

Each leg rotates about a hip joint attached to the chassis, with an independently driven wheel at its distal end.

This architecture allows the wheel and leg to take on different roles depending on the locomotion mode. On ordinary terrain, the legs can establish a desired stance and body height while the wheels provide efficient continuous locomotion. Because the four legs are independently articulated, the robot can also change or maintain a particular chassis height while moving rather than being restricted to a single fixed ground clearance.

During legged locomotion, the wheels can behave more like controllable contact points or feet. In the walking controllers developed so far, their rotations are actively resisted so that movement of a planted leg can transmit force into the ground rather than simply causing its wheel to roll away.

Early experiments demonstrated why this matters. Moving the legs while allowing the wheels to rotate freely produced relatively little useful chassis translation. Actively holding the wheel angles allowed leg motion to generate substantially greater ground reaction and was one of the key steps toward the eventual strut-like walking behavior.

However, completely locking the wheels is not necessarily the final or most capable use of the architecture. Because each wheel remains actuated during legged motion, its rotation could potentially be coordinated with the motion of its leg.

During a propulsion stroke, for example, controlled wheel torque could supplement the leg-generated ground interaction and make better use of the horizontal traction available at the wheel-ground interface. The amount of useful force would ultimately be limited by the available surface friction and the normal load carried by that wheel.

The opposite idea may be useful during leg recovery. Repositioning a leg can create an unwanted reaction on the chassis if its wheel remains in contact with the ground and scrapes across the surface. Rather than allowing that preparatory motion to partially undo the displacement produced by the previous propulsion stroke, the wheel could be commanded to roll in coordination with the recovering leg. Ideally, the wheel would accommodate the leg's motion relative to the ground, allowing it to glide into its next position with substantially less unwanted chassis translation.

This suggests that wheel rotation can serve more than one purpose:

- **Propulsion:** conventional wheeled driving on suitable terrain.
- **Braking:** resisting wheel rotation so a planted leg can transmit force into the ground.
- **Traction assistance:** applying controlled wheel torque during a leg propulsion stroke when sufficient friction is available.
- **Recovery compensation:** coordinating wheel rotation with leg repositioning to reduce unwanted reaction forces and translational losses.
- **Height-controlled mobility:** using the articulated legs to establish or maintain a desired chassis height while the wheels continue to move the robot.

The current walking controllers primarily demonstrate the braking case. Coordinated wheel-leg propulsion and recovery compensation are natural extensions of the control architecture and remain areas for further development.


## Simulation Model

The simulation is implemented in MuJoCo using an XML model derived from the Fusion assembly.

The goal was not to reproduce every CAD feature as collision geometry. Instead, I wanted a model that retained the important geometry, mass distribution, joint structure, and contact behavior of the design while remaining practical for repeated dynamic simulation.


### Visual and Collision Geometry

The exported STL geometry is used to preserve the appearance and overall geometry of the CAD design in MuJoCo.

Collision geometry is intentionally simpler. Rather than asking MuJoCo to resolve every feature of the detailed CAD meshes during contact, the model uses simplified geometric collision bodies where appropriate, including primitive representations of the chassis, legs, and wheels.

This separation provides detailed visual geometry while keeping contact calculations comparatively manageable and predictable.

The model also excludes collisions between bodies that are mechanically connected and expected to remain in contact as part of the assembly, such as a leg and its corresponding wheel or the chassis and its attached legs.


### Mass and Inertial Properties

Mass properties were derived from the Fusion model after assigning material properties to the CAD components.

The MuJoCo bodies include component masses, centers of mass, and inertia tensors based on these CAD-derived properties rather than relying solely on geometry-generated estimates.

This was particularly important because the project became heavily dependent on weight transfer. The ability to unload a wheel, establish three-point support, lean the chassis, or produce a stable propulsion stroke depends not only on the visible geometry of the robot but also on where its mass actually resides.

Some physical hardware and electronic components were represented through approximations where modeling every component individually would have added considerable complexity for relatively little effect on the behaviors being investigated. These approximations are discussed further in the Limitations section.


### Joints and Actuation

Each of the four legs is represented by a constrained hinge joint, and each wheel has its own rotational joint.

The allowable leg motion is based on the mechanically permitted ranges determined from the Fusion assembly. These limits were especially important during gait development, where several apparent control problems ultimately traced back to incorrectly mapped joint directions or angular limits.

The leg joints are controlled using MuJoCo position actuators. The walking controllers specify desired leg angles and interpolate between them to create the required stance changes, recovery motions, and propulsion strokes.

The wheel joints use motor actuators. This allows the same simulated hardware to support both conventional wheel driving and active resistance to wheel rotation during legged locomotion.


### Wheel Braking During Legged Locomotion

A major control discovery during the project was that the wheels needed to behave differently during walking than during rolling.

If a planted wheel remains free to rotate while its leg sweeps against the ground, a significant portion of the leg motion can simply turn the wheel. The intended propulsion is then lost to rolling.

For the legged locomotion controllers, each wheel's angular position is therefore captured and actively maintained using a simple proportional-derivative control law. The wheel motor applies torque to oppose deviations from that captured angle.

The result is not intended to represent a perfect physical brake. It provides a controllable approximation of a wheel being actively held during the propulsion portion of the gait.

This transformed the wheels from freely rolling elements into useful ground-contact points and was one of the key steps between merely moving the legs and producing meaningful leg-driven chassis displacement.


### Contact and Ground Interaction

MuJoCo handles contact between the wheel collision geometry and a planar ground surface. Friction parameters are defined in the XML model so that wheel-ground interaction can produce both rolling behavior and traction during legged propulsion.

The simulation also exposes the contact forces generated at these interfaces. I used these forces extensively during development to determine how the robot's weight was distributed among its four wheels.

This made it possible to observe transitions such as:

- A wheel becoming progressively unloaded
- Complete loss of wheel-ground contact
- Redistribution of weight onto the remaining support wheels
- Establishment of three-point support
- Touchdown after moving an airborne wheel
- Alternating support states during continuous walking

Those measurements became some of the most useful feedback available during controller development.


### Sensors and State Measurement

The model includes simulated state measurements used by the experimental controllers and diagnostic tools.

These include measurements of:

- Chassis position
- Chassis orientation
- Angular velocity
- Linear acceleration
- Joint state
- Wheel-ground contact force

Together, these measurements allowed controller behavior to be evaluated quantitatively rather than exclusively by watching the MuJoCo viewer.

For example, a gait that appeared to move the robot successfully could still be identified as problematic if it produced excessive lateral displacement or accumulated yaw over repeated cycles. Similarly, a wheel that appeared close to the ground could be identified as genuinely airborne when its measured normal contact force fell to zero.

This instrumentation became increasingly important as the project progressed from basic motion testing into sustained locomotion.


## Control and Locomotion

The control work progressed from simple actuator tests to increasingly coordinated behaviors. I deliberately developed the system incrementally so that each new behavior could build on something already understood.


### Wheeled Locomotion

The wheels provide the robot's simplest locomotion mode.

Initial controllers demonstrated forward and backward translation by commanding all four wheel motors in the appropriate direction. Differential wheel speeds were then used to rotate the chassis, including controlled autonomous turning.

This remains the preferred turning mechanism in the current implementation. Although leg-assisted turning could be explored in the future, the independently driven wheels already provide a direct and effective means of changing heading.


### Body Pose Control

Independent articulation of the four legs allows the chassis pose to be manipulated even when the wheels themselves are not providing locomotion.

Experiments demonstrated:

- Coordinated squatting
- Changes in chassis height
- Controlled leaning
- Load redistribution between wheels
- Individual wheel unloading

These experiments were important precursors to walking because they established that the robot could deliberately manipulate its support state rather than simply moving its legs.


### Three-Point Support and Wheel Manipulation

One particularly useful milestone was deliberately unloading the front-right wheel until its measured normal force reached zero.

Once the wheel was airborne, it could be moved independently and subsequently lowered until ground contact was detected again.

This demonstrated a complete sequence of:

1. Weight transfer
2. Wheel unloading
3. Three-point support
4. Wheel lift
5. Airborne wheel motion
6. Replanting
7. Return to four-wheel support

The experiment also highlighted the value of contact-force sensing. The controller did not have to infer contact solely from geometry. It could directly observe when the wheel ceased carrying load and when contact was re-established.


### Forward Legged Locomotion

Forward walking is produced using the two front legs in a reciprocal strut-like gait.

The gait emerged experimentally rather than from a predetermined trajectory. A successful front-leg propulsion primitive was first identified by manipulating the robot manually and observing how a planted front leg could thrust against the ground.

The continuous gait then developed around a simple principle: while one front leg performs its propulsion stroke, the opposite front leg recovers toward the position required for its next stroke.

The two roles alternate continuously:

```text
Front-right thrust + front-left recovery
Front-left thrust  + front-right recovery
Repeat
```

The wheels are actively held during these strokes so that the planted leg can transmit useful horizontal force through its wheel-ground contact.

The resulting gait produces sustained forward locomotion. It is not yet a fully optimized straight-line gait, and repeated cycles can accumulate lateral motion and yaw, but it demonstrates that the mechanism can produce continuous leg-driven translation under simulated dynamics.


### Backward Legged Locomotion

Backward walking uses the same general locomotion principle with the rear leg pair.

Development of this gait required additional experimentation because simply mirroring the front-leg controller did not initially produce a stable rear gait. Individual rear-leg propulsion strokes were therefore tested independently.

Those experiments showed that both rear legs could produce backward chassis displacement while generating approximately opposing lateral and yaw effects. This provided the basis for the final alternating rear-leg gait:

```text
Back-right thrust + back-left recovery
Back-left thrust  + back-right recovery
Repeat
```

The final controller produces sustained backward locomotion using the rear legs while the wheels remain actively resisted.

Together, the forward and backward controllers demonstrate that the articulated legs are capable of producing useful locomotion in both directions rather than serving only as adjustable suspension or stance mechanisms.


## Experimental Development

The final controllers are the result of a sequence of progressively more complicated experiments.

The rough development path was:

1. Establish basic MuJoCo model geometry and dynamics.
2. Verify independent wheel and leg actuation.
3. Demonstrate wheeled translation.
4. Demonstrate differential turning.
5. Develop squat and lean behaviors.
6. Measure individual wheel-ground forces.
7. Experiment with deliberate load transfer.
8. Unload an individual wheel.
9. Establish stable three-point support.
10. Lift, move, and replant an airborne wheel.
11. Test whether leg motion could create useful chassis propulsion.
12. Discover that freely rolling wheels significantly reduced leg-driven propulsion.
13. Add active wheel-angle holding during legged motion.
14. Develop a manual gait-control and data-logging workflow.
15. Identify a successful front-leg propulsion primitive.
16. Convert that primitive into continuous reciprocal forward walking.
17. Diagnose the asymmetries and joint-range issues exposed by continuous walking.
18. Test rear-leg propulsion independently.
19. Develop continuous reciprocal backward walking.

Not every intermediate controller worked. Some produced almost no translation. Some moved in the wrong direction. Some generated large arcs rather than straight trajectories. A few were impressively good at making the robot fall over.

Those failures were useful.

Because the simulation exposed contact forces, body motion, orientation, and joint state, an unsuccessful experiment could still provide information about what needed to change. This made the controller-development process much closer to experimentation than animation.


### Human-in-the-Loop Gait Discovery

At one stage, attempting to prescribe the gait directly was producing increasingly complicated controllers without a clear understanding of what motion the mechanism actually preferred.

I therefore switched approaches.

A manual controller allowed me to manipulate the legs interactively while the simulation logged robot state to CSV. I could mark a moment in the terminal, execute a motion that seemed promising, and then identify that interval in the recorded data.

This allowed successful motions to be discovered physically first and formalized programmatically afterward.

That workflow was particularly important in identifying the front-leg propulsion motion that eventually became the basis of continuous forward walking.


## Repository Contents

The repository intentionally includes both final demonstrations and selected intermediate experiments. The latter document some of the progression that led to the working locomotion controllers.


### Primary Demonstrations

**`controller_forward_walking_working.py`**  
Final continuous front-leg reciprocal gait for forward locomotion.

**`controller_backward_walking_working.py`**  
Final continuous rear-leg reciprocal gait for backward locomotion.

**`controller_autonomous_spin.py`**  
Differential wheel-based autonomous rotation.

**`controller_squat_success.py`**  
Successful coordinated squat and chassis-height manipulation.

**`controller_FR_three_point_success.py`**  
Successful front-right wheel unloading and three-point-support experiment.


### Development Experiments

**`controller_FR_only_locomotion_working.py`**  
Successful single-front-leg locomotion experiment that preceded the reciprocal forward gait.

**`manual_gait_log.csv`**  
Recorded data from the human-in-the-loop gait-development process.

Additional controller files preserve other useful development milestones and experiments.


### Model Files

**`model.xml`**  
Current MuJoCo model used by the final controllers.

**`Assets/`**  
STL meshes exported from the Autodesk Fusion assembly.

Additional XML files with names such as `*_working.xml` represent earlier milestones in development of the simulation model. They are retained as development history rather than as the recommended model for the final demonstrations.


## Limitations

Although the simulation is derived from the CAD design and incorporates CAD-derived physical properties, it should not yet be interpreted as a fully validated digital twin of a physical robot.

Important limitations include:

- The collision model is intentionally simpler than the detailed CAD geometry.
- Ground and wheel friction are simulation parameters rather than experimentally measured tire-surface properties.
- The wheel braking controller is an approximation of active wheel holding rather than a validated model of a specific physical braking system.
- Real actuator dynamics, backlash, compliance, electrical limitations, control latency, and other hardware effects are not yet fully represented.
- Some hardware and electronics are represented through simplified mass and inertial approximations rather than individually modeled components.
- The current walking gaits are primarily open-loop periodic motions.
- Forward and backward walking can accumulate lateral displacement and heading error over repeated gait cycles.
- The model has not yet been quantitatively validated against motion of a completed physical prototype.

For these reasons, the simulation is best viewed as a physics-based design and control-development environment. Its purpose is to investigate behavior, reject poor ideas early, and develop control strategies that can later be tested and refined on hardware.


## Future Work

There are several natural directions for extending the project.

### Closed-Loop Gait Stabilization

The current walking controllers already measure chassis orientation and displacement. These measurements could be incorporated directly into the gait controller so that leg stroke amplitude, timing, or support configuration is adjusted in response to accumulated yaw or lateral drift.

This would turn the current periodic walking gait into a heading-regulated locomotion controller.


### Coordinated Wheel-Leg Locomotion

The current walking controllers primarily use the wheels as actively held contact points.

A more advanced controller could vary wheel behavior throughout the gait. A wheel might brake during a propulsion stroke, apply additional torque when traction is available, and deliberately roll during leg recovery to reduce unwanted reaction forces.

This could reduce translational losses and make better use of the hybrid morphology.


### Terrain-Adaptive Control

The model could be extended with uneven terrain, obstacles, slopes, or surfaces with varying friction.

The legs could then be used not only for propulsion but also to maintain chassis height, alter ground clearance, redistribute load, or position individual wheels on useful contact surfaces.


### Simulation-to-Hardware Validation

The largest next step is comparison against a physical implementation.

Real measurements of actuator behavior, wheel-ground friction, component mass distribution, structural compliance, and robot motion could be used to refine the MuJoCo model and quantify the simulation-to-reality gap.

The simulation could then serve not only as a design environment but also as a controller-development platform for the physical robot.


### Additional Locomotion Strategies

Wheel-based differential turning already provides a straightforward method of changing heading, so dedicated legged turning was not required for the current demonstration.

However, leg-assisted turning, alternative support sequences, obstacle negotiation, and additional wheel-leg gaits remain interesting possibilities. I really look forward to trying out leaping motion. That would combine all 4 legs, differing from the 2 leg propulsion convention I was following in the current simulations.


## Development Process and AI Assistance

I used ChatGPT (OpenAI) extensively throughout this project as an engineering, simulation, and rapid-prototyping assistant. Since AI played a meaningful role in how I built the simulation and developed its controllers, I want to be transparent about where and how I used it.

I designed and modeled the robot itself in Autodesk Fusion. After assigning specific material properties to the CAD components, I extracted the physical and geometric information needed to build the simulation. This included component masses, centers of mass, dimensions, moments of inertia, positions within the assembly, joint locations and axes, and the mechanically allowable ranges of motion for each leg.

I provided these CAD-derived measurements to ChatGPT, which helped me translate the mechanical design into a MuJoCo XML model. This included developing and iterating on the body and joint definitions, inertial properties, collision geometry, actuators, sensors, and wheel-ground interactions. I continually compared the resulting simulation against my CAD model and the behavior I expected from the actual mechanism, correcting geometry, joint directions, ranges of motion, and other properties as discrepancies appeared.

ChatGPT also became particularly useful for rapid controller prototyping. Rather than jumping straight into trying to make the robot walk, I progressively tested simpler behaviors such as wheel motion, differential turning, squatting, leaning, load transfer, wheel unloading, three-point support, and individual leg movement. We used quantities available within the simulation, including wheel contact forces, body position and orientation, joint states, and displacement, to figure out what the robot was actually doing rather than relying entirely on what looked right on screen.

One of my favorite parts of the development process was the human-in-the-loop workflow we eventually used to discover the walking gait. We created a manual-control and data-logging system that recorded the robot's state to CSV while I manipulated it interactively in MuJoCo. I could mark a moment in the terminal, perform a movement that seemed promising, and then use that marker to locate the corresponding section of recorded data. In effect, I temporarily became the robot's very inefficient human motion planner. Once I found motions that worked, the recorded behavior gave us something concrete to reproduce and refine programmatically.

The final walking controllers came out of a lot of back-and-forth experimentation rather than a magical prompt that made the robot start strolling across the screen. I proposed and tested mechanical motions, watched how the simulated robot responded, and caught problems such as incorrect joint directions, insufficient ranges of motion, poor support configurations, and ineffective weight transfer. ChatGPT helped interpret the quantitative output, suggest experiments and control strategies, and rapidly produce revised versions of the Python controllers. I would run them, report what the robot actually did, correct assumptions when necessary, and we would iterate again. Eventually, that process produced continuous reciprocal-strut gaits for both forward and backward legged locomotion.

That iterative process became one of the most valuable parts of the project. The simulation stopped being just a place to watch my CAD model move and became a tool for testing the design before committing it to hardware. I could inspect individual wheel loads, support transitions, body attitude, loss of ground contact, gait-induced yaw, and other quantities that would be considerably more annoying to measure on an early physical prototype.

I was responsible for the mechanical concept and CAD design, extracting and checking the physical parameters used by the simulation, running the experiments, visually evaluating the robot's behavior, proposing and refining locomotion ideas, and ultimately deciding which behaviors worked. ChatGPT contributed substantially to translating the CAD data into the MuJoCo model, building experimental and data-logging tools, interpreting simulation results, exploring control strategies, and rapidly iterating on the Python and XML implementations.

Although AI assistance was substantial, I worked to understand the tools and implementations being developed throughout the project. I can read and interpret both the Python controllers and MuJoCo XML model, understand the purpose of their major components, and explain the simulation and control workflows used here. I treated AI as a way to accelerate iteration, not as a substitute for understanding the system I was building. When I encountered something I did not understand, I tried to resolve that gap before building further on it.

I should also mention that I used AI to help clean up this README file. It was extremely helpful in making it concise, eliminating redundancies, filling gaps and expanding on explanations.

## Reproducibility

The final repository was tested from a fresh Git clone rather than only from the original development directory.

The repository was cloned into a separate local directory, its dependencies were installed independently, and the forward walking controller was successfully executed using the files contained in the cloned repository.

This test was intended to catch dependencies on untracked local files or development-specific paths before the project was shared.

## Acknowledgments

Thank you to Dr. Boxi Xia for the opportunity to work on this getting-started project with the Duke General Robotics Lab, and for providing the context and direction that made it possible. It was a new and very welcome experience :)

## Author

**David Kiwanga John**  
Duke University  
Mechanical Engineering and Physics
