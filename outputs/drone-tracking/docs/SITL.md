# Optional ArduPilot SITL command replay

This component is a runnable adapter for logged visual-control commands. It has not been exercised against an installed ArduPilot instance in this workspace. It is distinct from the included, verified synthetic camera simulation.

Use Linux/WSL for the simulator and follow ArduPilot's installation instructions:
https://ardupilot.org/dev/docs/SITL-setup-landingpage.html

Start Copter with a MAVProxy output for the local replay script:

```bash
cd ~/ardupilot/ArduCopter
../Tools/autotest/sim_vehicle.py --map --console --out=127.0.0.1:14551
```

In the simulator's MAVProxy console:

```text
mode guided
arm throttle
takeoff 10
```

Wait until the simulated vehicle reaches altitude, then in the project's Python environment:

```bash
python -m pip install pymavlink
python scripts/replay_sitl.py --csv demo/simulation/simulation.csv
```

The replay script binds only to localhost, requires an ArduPilot heartbeat in armed GUIDED mode, and maps positive upward controller velocity to negative NED z velocity. Yaw commands are in radians/second and vertical commands in metres/second for this adapter. Velocity/yaw-rate setpoints use `SET_POSITION_TARGET_LOCAL_NED`. The toy camera simulator uses normalized optical coordinates, so its gains are not calibrated flight gains. Recorded-video commands can also be replayed, but they do not respond to the simulated aircraft's movement.

Observe the Copter's yaw and altitude in the SITL console, then use `mode land` in MAVProxy. Replay cannot change arm/takeoff state and is not a 3D renderer. A complete closed-loop SITL vision test would need a simulator camera feed synchronized with the vehicle pose, calibration of the optical controller, and a live command bridge. These remain integration work for an environment with ArduPilot and a camera renderer installed.

References:
- https://ardupilot.org/dev/docs/copter-sitl-mavproxy-tutorial.html
- https://ardupilot.org/dev/docs/copter-commands-in-guided-mode.html
