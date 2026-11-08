"""Replay logged commands against a locally running ArduPilot SITL instance.

This integration sends velocities/yaw rate, not camera feedback. It cannot itself
demonstrate closed-loop tracking. The autopilot must already be guided and airborne.
"""
import argparse
import csv
import math
import time


def main():
    from pymavlink import mavutil
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--port', type=int, default=14551)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Port must be 1024..65535.')
    with open(args.csv, newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        parser.error('Empty command log.')
    last = -1.
    for row in rows:
        for name in ('time_s','yaw','vertical','forward'):
            row[name] = float(row[name])
            if not math.isfinite(row[name]):
                parser.error('Non-finite command or timestamp.')
        if row['time_s'] < last or abs(row['yaw']) > .6 or abs(row['vertical']) > .4 or row['forward'] != 0:
            parser.error('Invalid timestamp ordering or command limits.')
        last = row['time_s']
    connection = mavutil.mavlink_connection(f'udpin:127.0.0.1:{args.port}')
    heartbeat = connection.wait_heartbeat(timeout=15)
    if heartbeat is None:
        raise RuntimeError('No SITL heartbeat. Start MAVProxy with the matching --out port.')
    if heartbeat.autopilot != mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA:
        raise RuntimeError('Expected ArduPilot SITL.')
    if heartbeat.custom_mode != 4 or not heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
        raise RuntimeError('Set simulated Copter GUIDED, arm, and take off in MAVProxy first.')
    def send(row):
        # Ignore position, acceleration, and yaw angle; use vx/vy/vz and yaw_rate.
        mask = 1|2|4|64|128|256|1024
        connection.mav.set_position_target_local_ned_send(
            0, connection.target_system, connection.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED, mask,
            0,0,0, 0,0,-row['vertical'], 0,0,0, 0,row['yaw'])
    started = time.monotonic()
    offset = rows[0]['time_s']
    try:
        for row in rows:
            delay = started+row['time_s']-offset-time.monotonic()
            if delay > 0:
                time.sleep(delay)
            send(row)
    finally:
        send(dict(vertical=0.,yaw=0.))
        connection.close()
    print('Replay complete. This tested command transport, not camera feedback.')


if __name__ == '__main__':
    main()

