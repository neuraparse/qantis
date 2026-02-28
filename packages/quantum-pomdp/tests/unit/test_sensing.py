"""Tests for the sensing module (sensor interface and IMU quantum fusion)."""

import numpy as np
import pytest

from quantum_pomdp.sensing.sensor_interface import SensorInterface, DiscreteSensor
from quantum_pomdp.sensing.imu_fusion import IMUReading, QuantumSensorReading, IMUQuantumFusion


class TestDiscreteSensor:
    def test_creation(self) -> None:
        obs_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
        sensor = DiscreteSensor(obs_matrix)
        assert sensor is not None

    def test_observation_dim(self) -> None:
        obs_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
        sensor = DiscreteSensor(obs_matrix)
        assert sensor.observation_dim == 2

    def test_observation_dim_larger(self) -> None:
        obs_matrix = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        sensor = DiscreteSensor(obs_matrix)
        assert sensor.observation_dim == 3

    def test_observation_probability_correct_state(self) -> None:
        obs_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
        sensor = DiscreteSensor(obs_matrix)
        assert np.isclose(sensor.observation_probability(0, 0), 0.9)
        assert np.isclose(sensor.observation_probability(0, 1), 0.1)
        assert np.isclose(sensor.observation_probability(1, 0), 0.2)
        assert np.isclose(sensor.observation_probability(1, 1), 0.8)

    def test_observation_probability_returns_float(self) -> None:
        obs_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        sensor = DiscreteSensor(obs_matrix)
        prob = sensor.observation_probability(0, 0)
        assert isinstance(prob, float)

    def test_observation_probability_in_valid_range(self) -> None:
        obs_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
        sensor = DiscreteSensor(obs_matrix)
        for state in range(2):
            for obs in range(2):
                prob = sensor.observation_probability(state, obs)
                assert 0.0 <= prob <= 1.0

    def test_get_observation_returns_array(self) -> None:
        obs_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
        sensor = DiscreteSensor(obs_matrix)
        obs = sensor.get_observation()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (2,)

    def test_observation_to_index(self) -> None:
        obs_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
        sensor = DiscreteSensor(obs_matrix)
        obs = np.array([0.1, 0.9])
        idx = sensor.observation_to_index(obs)
        assert idx == 1

    def test_implements_sensor_interface(self) -> None:
        obs_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
        sensor = DiscreteSensor(obs_matrix)
        assert isinstance(sensor, SensorInterface)


class TestIMUReading:
    def test_creation(self) -> None:
        accel = np.array([0.0, 0.0, -9.81])
        gyro = np.array([0.01, -0.02, 0.005])
        reading = IMUReading(acceleration=accel, angular_velocity=gyro)
        assert np.array_equal(reading.acceleration, accel)
        assert np.array_equal(reading.angular_velocity, gyro)
        assert reading.timestamp == 0.0

    def test_creation_with_timestamp(self) -> None:
        reading = IMUReading(
            acceleration=np.array([1.0, 0.0, -9.81]),
            angular_velocity=np.zeros(3),
            timestamp=1.5,
        )
        assert reading.timestamp == 1.5


class TestQuantumSensorReading:
    def test_creation(self) -> None:
        reading = QuantumSensorReading(
            value=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.001, 0.001, 0.001]),
        )
        assert np.array_equal(reading.value, [1.0, 2.0, 3.0])
        assert reading.quantum_enhanced is True
        assert reading.timestamp == 0.0

    def test_creation_classical(self) -> None:
        reading = QuantumSensorReading(
            value=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, 0.1, 0.1]),
            quantum_enhanced=False,
        )
        assert reading.quantum_enhanced is False


class TestIMUQuantumFusion:
    def test_creation_defaults(self) -> None:
        fusion = IMUQuantumFusion()
        assert fusion.process_noise == 0.01
        assert fusion.classical_noise == 0.1
        assert fusion.quantum_noise == 0.002

    def test_initial_position_is_zero(self) -> None:
        fusion = IMUQuantumFusion()
        assert np.allclose(fusion.position, np.zeros(3))

    def test_initial_velocity_is_zero(self) -> None:
        fusion = IMUQuantumFusion()
        assert np.allclose(fusion.velocity, np.zeros(3))

    def test_predict_with_imu_reading(self) -> None:
        fusion = IMUQuantumFusion()
        imu = IMUReading(
            acceleration=np.array([1.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
        )
        dt = 0.1
        state = fusion.predict(imu, dt)
        assert state.shape == (9,)
        # Position should have moved in x direction
        assert fusion.position[0] > 0
        # Velocity should be positive in x direction
        assert fusion.velocity[0] > 0

    def test_predict_updates_velocity(self) -> None:
        fusion = IMUQuantumFusion()
        imu = IMUReading(
            acceleration=np.array([0.0, 2.0, 0.0]),
            angular_velocity=np.zeros(3),
        )
        dt = 0.5
        fusion.predict(imu, dt)
        # v = a * dt = 2.0 * 0.5 = 1.0 in y
        assert np.isclose(fusion.velocity[1], 1.0)

    def test_predict_updates_position(self) -> None:
        fusion = IMUQuantumFusion()
        imu = IMUReading(
            acceleration=np.array([0.0, 0.0, 1.0]),
            angular_velocity=np.zeros(3),
        )
        dt = 1.0
        fusion.predict(imu, dt)
        # pos = 0.5 * a * dt^2 = 0.5 * 1.0 * 1.0 = 0.5 in z
        assert np.isclose(fusion.position[2], 0.5)

    def test_update_classical(self) -> None:
        fusion = IMUQuantumFusion()
        measurement = np.array([1.0, 2.0, 3.0])
        state = fusion.update_classical(measurement)
        assert state.shape == (9,)
        # After Kalman update with measurement, position should move toward measurement
        assert np.linalg.norm(fusion.position - measurement) < np.linalg.norm(
            np.zeros(3) - measurement
        )

    def test_update_classical_reduces_uncertainty(self) -> None:
        fusion = IMUQuantumFusion()
        initial_uncertainty = fusion.position_uncertainty.copy()
        measurement = np.array([1.0, 0.0, 0.0])
        fusion.update_classical(measurement)
        # Kalman update should reduce position uncertainty
        assert np.all(fusion.position_uncertainty <= initial_uncertainty)

    def test_predict_then_update_cycle(self) -> None:
        fusion = IMUQuantumFusion()
        imu = IMUReading(
            acceleration=np.array([1.0, 0.0, -9.81]),
            angular_velocity=np.zeros(3),
        )

        # Predict
        fusion.predict(imu, dt=0.01)
        pos_after_predict = fusion.position.copy()

        # Update with a classical measurement
        measurement = np.array([0.001, 0.0, -0.0005])
        fusion.update_classical(measurement)
        pos_after_update = fusion.position.copy()

        # The positions should differ (update changes state)
        assert not np.allclose(pos_after_predict, pos_after_update)

    def test_position_property(self) -> None:
        fusion = IMUQuantumFusion()
        pos = fusion.position
        assert pos.shape == (3,)

    def test_velocity_property(self) -> None:
        fusion = IMUQuantumFusion()
        vel = fusion.velocity
        assert vel.shape == (3,)

    def test_multiple_predict_steps(self) -> None:
        """Multiple predict steps should accumulate position and velocity."""
        fusion = IMUQuantumFusion()
        imu = IMUReading(
            acceleration=np.array([1.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
        )
        for _ in range(10):
            fusion.predict(imu, dt=0.1)

        # After 10 steps of 0.1s with 1 m/s^2 accel in x:
        # velocity should be approximately 1.0 m/s
        assert fusion.velocity[0] > 0.9
        # Position should be positive
        assert fusion.position[0] > 0
