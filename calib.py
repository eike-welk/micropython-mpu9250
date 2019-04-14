# A calibration library for the various sensors of a IMU.

import time
from collections import namedtuple
from enum import Enum
from math import pi
from multiprocessing.dummy import Pool
from multiprocessing import log_to_stderr
import multiprocessing
import logging
from typing import Tuple

import numpy as np
from numpy import abs, all

import ak8963
import mpu6500

MeasData = namedtuple('MeasData', ['time', 'acc', 'rot', 'mag'])
MsgCalib = namedtuple('msgCalib', ['text', 'sound'])
MsgUser = namedtuple('msgUser', ['text', 'command'])

logger = multiprocessing.log_to_stderr()
logger.setLevel(logging.DEBUG)


class Sound(Enum):
    quiet = 0
    continue_sound = 1
    error = 11
    measurement = 12
    question = 13


class UserCommand(Enum):
    exit_program = 11


class CalibratorStill:
    """
    Perform the calibration of IMU sensors.

    This class needs a frontend that communicates with the user.
    """

    def __init__(self):
        # The sensors
        self.accl = mpu6500.MPU6500()
        self.gyro = self.accl
        self.magn = ak8963.AK8963()

        self.read_total_samples = 100
        self.read_n_samples = 20
        self.read_interval = 1.0 / 20        # 1 / 20 Hz

        self.calib_rot_max = pi / 180 * 0.3  # 0.1 deg / s
        self.calib_acc_max = 0.2             # 1 cm / s / s

    def read_sensor_data(self) -> Tuple[np.array, np.array, np.array, np.array]:
        """
        Read sensor data for a short amount of time.
        """
        logger.debug('read_sensor_data: Start')

        n = self.read_n_samples
        tim = np.zeros((n,))
        acc = np.zeros((n, 3))
        rot = np.zeros((n, 3))
        mag = np.zeros((n, 3))
        for i in range(n):
            tim[i] = time.time()
            acc[i, ...] = self.accl.read_acceleration_raw()
            rot[i, ...] = self.gyro.read_gyro_raw()
            mag[i, ...] = self.magn.read_magnetic_raw()
            time.sleep(self.read_interval)
        return tim, acc, rot, mag

    def is_no_motion(self, acc_raw, rot_raw, gyro_is_calibrated=True) -> bool:
        """
        Determine if the device was held still during a measurement.
        Takes raw measurement values.
        """
        # Scale acceleration and rotation for analysis.
        acc = np.zeros_like(acc_raw)
        rot = np.zeros_like(rot_raw)
        for i in range(acc.shape[0]):
            acc[i] = self.accl.compute_acceleration(acc_raw[i])
            rot[i] = self.gyro.compute_gyro(rot_raw[i])

        # Statistical analysis of rotation and acceleration
        #acc_mean = acc.mean(axis=0)
        rot_mean = rot.mean(axis=0)
        acc_std = acc.std(axis=0)
        rot_std = rot.std(axis=0)

        print('acc_std:', acc_std, 'rot_std:', rot_std)

        if gyro_is_calibrated:
            # The average rotation must be zero
            # and rotation and acceleration must not change
            return (    all(abs(rot_mean) < self.calib_rot_max)
                    and all(rot_std * 2 < self.calib_rot_max)
                    and all(acc_std * 2 < self.calib_acc_max))
        else:
            # Rotation and acceleration must not change
            return (    all(rot_std * 2 < self.calib_rot_max)
                    and all(acc_std * 2 < self.calib_acc_max))

    def measure_1(self, gyro_is_calibrated: bool) -> MeasData:
        """
        Take measurement values from one orientation of the device.

        Ensure that the device does not move during the measurement.

        Return the average measurement values.
        """
        # Storage for the accumulated sensor values.
        # Initialized to length 0 along the measurement axis.
        tim_all = np.zeros((0,))
        acc_raw_all = np.zeros((0, 3))
        rot_raw_all = np.zeros((0, 3))
        mag_raw_all = np.zeros((0, 3))

        thread_pool = Pool()

        # Read the first small chunk of data.
        tim, acc_raw, rot_raw, mag_raw = self.read_sensor_data()
        
        for _ in range(100):
            # Read the sensor values in a separate thread.
            res = thread_pool.apply_async(self.read_sensor_data)
            self.raise_user_wants_quit()

            # If the device was held still enough, store the new sensor values.
            if self.is_no_motion(acc_raw, rot_raw, gyro_is_calibrated):
                logger.debug('No motion: storing measured data.')
                self.play_sound(Sound.measurement)
                tim_all = np.concatenate((tim_all, tim), axis=0)
                acc_raw_all = np.concatenate((acc_raw_all, acc_raw), axis=0)
                rot_raw_all = np.concatenate((rot_raw_all, rot_raw), axis=0)
                mag_raw_all = np.concatenate((mag_raw_all, mag_raw), axis=0)
            else:
                logger.debug('Motion: discarding measured data.')

            # Enough data has been collected 
            if len(tim_all) >= self.read_total_samples:
                logger.debug('Enough data has been collected.')
                # Data quality: Was there no motion during the whole measurement?
                if self.is_no_motion(acc_raw_all, rot_raw_all, 
                                     gyro_is_calibrated):
                    logger.debug('Data quality good.')
                    break
                else:
                    # Delete one small chunk of measurement data 
                    # from the front of the arrays.
                    logger.debug('Data quality bad. Delete some.')
                    n_del = len(tim)
                    tim_all = tim_all[-n_del:]
                    acc_raw_all = acc_raw_all[n_del:, :]
                    rot_raw_all = rot_raw_all[n_del:, :]
                    mag_raw_all = mag_raw_all[n_del:, :]

            # Get the new sensor values
            tim, acc_raw, rot_raw, mag_raw = res.get()
        else:
            # Not enough data was collected. The device was moved too many times.
            return None

        # Return only the mean values of all measurements.
        res = MeasData(tim_all[0], 
                       acc_raw_all.mean(axis=0), 
                       rot_raw_all.mean(axis=0), 
                       mag_raw_all.mean(axis=0))
        
        thread_pool.close()
        thread_pool.join()

        return res

    def run(self):
        # Calibrate the gyro
        vals = self.measure_1(gyro_is_calibrated=False)
        print('gyro:', self.gyro.compute_gyro(vals.rot))
        print('accl:', self.accl.compute_acceleration(vals.acc))

        gyro_offset = self.gyro.compute_gyro(vals.rot)
        self.gyro._gyro_calib_offs = -gyro_offset / self.gyro._gyro_unit_fact

        # Test the calibration
        vals = self.measure_1(gyro_is_calibrated=True)
        print('gyro:', self.gyro.compute_gyro(vals.rot))
        print('accl:', self.accl.compute_acceleration(vals.acc))

    # Concepts for data acquisition and evaluation

    # Take some measurements to estimate the noise
    # and to calibrate the gyroscope

    # Estimate if one of the senors is too noisy.
    # Display the noise of each sensor

    # Repeat
    # Take measurements from different orientations of the device
    # Every time the system does not move for 1 second start a measurement.
    # A measurement takes 5 seconds.
    # The device must not move at all, to calibrate the accelerometer.

        # The magnetometer values are recorded too.

        # When there are enough measurements from different directions, the
        # calibration parameters can be computed.

        # multiprocessing.pool.Pool

    def write_text(self, text: str):
        self.communicate(MsgCalib(text, None))

    def play_sound(self, sound: Sound):
        self.communicate(MsgCalib(None, sound))

    def communicate(self, msg: MsgCalib):
        """Send a message to the front end."""
        if msg.text:
            print(msg.text)
        if msg.sound and msg.sound != Sound.quiet:
            print("\a")

    def raise_user_wants_quit(self):
        """
        Raise an exception if the user wants to quit.
        """
        return


class CalibFrontText:
    """
    Communicate with the user during the calibration process of an IMU.

    Uses text that appears in a terminal.
    """
    pass

if __name__ == "__main__":
    cal = CalibratorStill()
    cal.run()
