# Copyright (c) 2018-2019 Mika Tuupola
# Copyright (c) 2019      Eike Welk
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of  this software and associated documentation files (the "Software"), to
# deal in  the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copied of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
Python I2C driver for MPU6500 6-axis motion tracking device
"""

# pylint: disable=import-error
import struct
import time

import numpy as np

import Adafruit_GPIO.I2C

# pylint: enable=import-error

# Internal registers
_GYRO_CONFIG = 0x1b
_ACCEL_CONFIG = 0x1c
_ACCEL_CONFIG2 = 0x1d
_INT_PIN_CFG = 0x37
_ACCEL_XOUT_H = 0x3b
_ACCEL_XOUT_L = 0x3c
_ACCEL_YOUT_H = 0x3d
_ACCEL_YOUT_L = 0x3e
_ACCEL_ZOUT_H = 0x3f
_ACCEL_ZOUT_L= 0x40
_TEMP_OUT_H = 0x41
_TEMP_OUT_L = 0x42
_GYRO_XOUT_H = 0x43
_GYRO_XOUT_L = 0x44
_GYRO_YOUT_H = 0x45
_GYRO_YOUT_L = 0x46
_GYRO_ZOUT_H = 0x47
_GYRO_ZOUT_L = 0x48
_WHO_AM_I = 0x75

# Accelerometer sensitivity
#_ACCEL_FS_MASK = 0b00011000
ACCEL_FS_SEL_2G = 0b00000000
ACCEL_FS_SEL_4G = 0b00001000
ACCEL_FS_SEL_8G = 0b00010000
ACCEL_FS_SEL_16G = 0b00011000

_ACCEL_SO_2G = 16384 # 1 / 16384 ie. 0.061 mg / digit
_ACCEL_SO_4G = 8192 # 1 / 8192 ie. 0.122 mg / digit
_ACCEL_SO_8G = 4096 # 1 / 4096 ie. 0.244 mg / digit
_ACCEL_SO_16G = 2048 # 1 / 2048 ie. 0.488 mg / digit

# Gyroscope sensitivity
#_GYRO_FS_MASK = 0b00011000
GYRO_FS_SEL_250DPS = 0b00000000
GYRO_FS_SEL_500DPS = 0b00001000
GYRO_FS_SEL_1000DPS = 0b00010000
GYRO_FS_SEL_2000DPS = 0b00011000

_GYRO_SO_250DPS = 131
_GYRO_SO_500DPS = 62.5
_GYRO_SO_1000DPS = 32.8
_GYRO_SO_2000DPS = 16.4

_TEMP_SO = 333.87
_TEMP_OFFSET = 21

# Used for enabling and disabling the i2c bypass access
_I2C_BYPASS_MASK = 0b00000010
_I2C_BYPASS_EN = 0b00000010
_I2C_BYPASS_DIS = 0b00000000

#  Units of measurements: conversion factors/selectors
ACCEL_UNIT_G = 1
ACCEL_UNIT_M_S2 = 9.80665 # 1 g = 9.80665 m/s2 ie. standard gravity
GYRO_UNIT_DEG_S = 1
GYRO_UNIT_RAD_S = 0.017453292519943 # 1 deg/s is 0.017453292519943 rad/s

class MPU6500:
    """Class which provides interface to MPU6500 6-axis motion tracking device."""
    def __init__(
        self, i2c_interface=None, busnum=1, address=0x68,
        accel_sensitivity=ACCEL_FS_SEL_2G, gyro_sensitivity=GYRO_FS_SEL_250DPS,
        accel_unit_factor=ACCEL_UNIT_M_S2, gyro_unit_factor=GYRO_UNIT_RAD_S,
    ):
        # Configure I2C connection
        if i2c_interface is None:
            # Use pure python I2C interface if none is specified.
            import Adafruit_PureIO.smbus
            self.i2c = Adafruit_PureIO.smbus.SMBus(busnum)
        else:
            # Otherwise use the provided class to create an smbus interface.
            self.i2c = i2c_interface(busnum)

        self.address = address

        if 0x71 != self.read_whoami():
            raise RuntimeError("MPU6500 not found on I2C bus.")

        self._accel_sens_fact = self._config_accel_sensitivity(accel_sensitivity)
        self._accel_unit_fact = accel_unit_factor
        self._accel_calib_fact = np.ones((3,))
        self._accel_calib_offs = np.zeros((3,))

        self._gyro_sens_fact = self._config_gyro_sensitivity(gyro_sensitivity)
        self._gyro_unit_fact = gyro_unit_factor
        self._gyro_calib_fact = np.ones((3,))
        self._gyro_calib_offs = np.zeros((3,))

        # Enable I2C bypass to access for MPU9250 magnetometer access.
        char = self._read_register_char(_INT_PIN_CFG)
        char &= ~_I2C_BYPASS_MASK # clear I2C bits
        char |= _I2C_BYPASS_EN
        self._write_register_char(_INT_PIN_CFG, char)
    
    def read_acceleration_raw(self):
        """Read the raw acceleration values from the sensor."""
        return self._read_register_three_shorts(_ACCEL_XOUT_H)

    def compute_acceleration(self, xyz_raw):
        """
        Compute calibrated and scaled acceleration values from raw 
        sensor values.

        Uses a linear transformation: 
            `x = f * x_raw + o`

        With:
            f = unit_factor * calibration_factor * sensitivity_factor
            o = unit_factor * calibration_offset

        The unit_factor can be changed at any time, it is independent of 
        the calibration or sensitivity. 
        """
        f = self._accel_unit_fact * self._accel_calib_fact * self._accel_sens_fact
        o = self._accel_unit_fact * self._accel_calib_offs
        return f * xyz_raw + o

    def read_acceleration(self):
        """
        Acceleration measured by the sensor. Will return a 3-tuple of 
        X, Y, Z axis acceleration values as floats. 
        
        By default the units are m/s^2. Will return values in g if 
        constructor is provided with the parameter 
        `accel_unit_factor=ACCEL_UNIT_G`.
        """
        xyz_raw = self.read_acceleration_raw()
        return self.compute_acceleration(xyz_raw)

    def read_gyro_raw(self):
        """Read the raw angular velocity values from the sensor."""
        return np.array(self._read_register_three_shorts(_GYRO_XOUT_H),
                        dtype=np.float64)

    def compute_gyro(self, xyz_raw):
        """
        Compute calibrated and scaled angular velocity values from raw 
        sensor values.

        Uses a linear transformation: 
            `x = f * x_raw + o`

        With:
            f = unit_factor * calibration_factor * sensitivity_factor
            o = unit_factor * calibration_offset

        The unit_factor can be changed at any time, it is independent of 
        the calibration or sensitivity. 
        """
        f = self._gyro_unit_fact * self._gyro_calib_fact * self._gyro_sens_fact
        o = self._gyro_unit_fact * self._gyro_calib_offs
        return f * xyz_raw + o

    def read_gyro(self):
        """
        Angular velocity measured by the sensor. Will return a 3-tuple of 
        angular velocities around the X, Y, Z axes as floats. 
        
        By default the units are rad/s. Will return values in deg/s if 
        constructor is provided with the parameter 
        `gyro_unit_factor=GYRO_UNIT_DEG_S`.
        """
        xyz_raw = self.read_acceleration_raw()
        return self.compute_acceleration(xyz_raw)

    def read_temperature(self):
        """
        Die temperature in celsius as a float.
        """
        temp = self._read_register_short(_TEMP_OUT_H)
        return ((temp - _TEMP_OFFSET) / _TEMP_SO) + _TEMP_OFFSET

    def read_whoami(self):
        """ Value of the whoami register. """
        return self._read_register_char(_WHO_AM_I)

    # def calibrate_gyro(self, count=256, delay=0):
    #     ox, oy, oz = (0.0, 0.0, 0.0)
    #     self._gyro_offset = (0.0, 0.0, 0.0)
    #     n = float(count)

    #     while count:
    #         time.sleep(delay/1000)
    #         gx, gy, gz = self.read_gyro()
    #         ox += gx
    #         oy += gy
    #         oz += gz
    #         count -= 1

    #     self._gyro_offset = (ox / n, oy / n, oz / n)
    #     return self._gyro_offset

    def _read_register_short(self, register):
        buf = self.i2c.read_i2c_block_data(self.address, register, 2)
        return struct.unpack(">h", buf)[0]

    def _write_register_short(self, register, value):
        buf = struct.pack(">h", value)
        self.i2c.write_i2c_block_data(self.address, register, buf)

    def _read_register_three_shorts(self, register):
        buf = self.i2c.read_i2c_block_data(self.address, register, 6)
        return struct.unpack(">hhh", buf)

    def _read_register_char(self, register):
        return self.i2c.read_byte_data(self.address, register)

    def _write_register_char(self, register, value):
        self.i2c.write_byte_data(self.address, register, value)

    def _config_accel_sensitivity(self, value):
        # Set the sensor sensitivity
        self._write_register_char(_ACCEL_CONFIG, value)

        # Return the sensitivity factor.
        if ACCEL_FS_SEL_2G == value:
            return 1.0 / _ACCEL_SO_2G
        elif ACCEL_FS_SEL_4G == value:
            return 1.0 / _ACCEL_SO_4G
        elif ACCEL_FS_SEL_8G == value:
            return 1.0 / _ACCEL_SO_8G
        elif ACCEL_FS_SEL_16G == value:
            return 1.0 / _ACCEL_SO_16G

    def _config_gyro_sensitivity(self, value):
        # Set the sensor sensitivity
        self._write_register_char(_GYRO_CONFIG, value)

        # Return the sensitivity factor.
        if GYRO_FS_SEL_250DPS == value:
            return 1.0 / _GYRO_SO_250DPS
        elif GYRO_FS_SEL_500DPS == value:
            return 1.0 / _GYRO_SO_500DPS
        elif GYRO_FS_SEL_1000DPS == value:
            return 1.0 / _GYRO_SO_1000DPS
        elif GYRO_FS_SEL_2000DPS == value:
            return 1.0 / _GYRO_SO_2000DPS

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass
