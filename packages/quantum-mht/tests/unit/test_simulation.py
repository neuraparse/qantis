"""Tests for quantum_mht.simulation module.

Validates the motion models (ConstantVelocity, ConstantTurn, ConstantAcceleration),
Target, Drone, SimulationWorld, DroneSwarm, and scenario generators.
"""

import numpy as np
import pytest

from quantum_mht.simulation.dynamics import (
    ConstantVelocity,
    ConstantTurn,
    ConstantAcceleration,
)
from quantum_mht.simulation.target import Target
from quantum_mht.simulation.drone import Drone
from quantum_mht.simulation.world import SimulationWorld
from quantum_mht.simulation.swarm import DroneSwarm
from quantum_mht.simulation.scenario_generator import (
    crossing_targets,
    dense_clutter,
    swarm_patrol,
)
from quantum_mht.fusion.sensor_model import LinearSensor


class TestConstantVelocity:
    """Test ConstantVelocity motion model."""

    def test_propagate_preserves_state_dimension(self) -> None:
        """Propagated state should have the same dimension as input."""
        cv = ConstantVelocity(process_noise=0.01)
        rng = np.random.default_rng(42)
        state = np.array([0.0, 1.0, 0.0, 2.0])

        new_state = cv.propagate(state, dt=1.0, rng=rng)
        assert new_state.shape == state.shape

    def test_transition_matrix_shape(self) -> None:
        """CV transition matrix should be (4, 4)."""
        cv = ConstantVelocity()
        F = cv.transition_matrix(dt=1.0)
        assert F.shape == (4, 4)

    def test_transition_matrix_structure(self) -> None:
        """CV F matrix should propagate position by velocity * dt."""
        cv = ConstantVelocity()
        dt = 2.0
        F = cv.transition_matrix(dt)

        # F[0,1] = dt (x += vx * dt)
        assert np.isclose(F[0, 1], dt)
        # F[2,3] = dt (y += vy * dt)
        assert np.isclose(F[2, 3], dt)
        # Diagonal is identity
        assert np.isclose(F[0, 0], 1.0)
        assert np.isclose(F[1, 1], 1.0)
        assert np.isclose(F[2, 2], 1.0)
        assert np.isclose(F[3, 3], 1.0)

    def test_propagate_moves_position(self) -> None:
        """With zero noise, position should move by velocity*dt."""
        cv = ConstantVelocity(process_noise=0.0)
        rng = np.random.default_rng(42)
        state = np.array([0.0, 5.0, 0.0, 3.0])

        new_state = cv.propagate(state, dt=1.0, rng=rng)
        # x = 0 + 5*1 = 5, y = 0 + 3*1 = 3 (approximately, noise is 0)
        assert np.isclose(new_state[0], 5.0, atol=0.01)
        assert np.isclose(new_state[2], 3.0, atol=0.01)


class TestConstantTurn:
    """Test ConstantTurn motion model."""

    def test_propagate_with_turn_rate(self) -> None:
        """CT propagation should produce valid state with non-zero turn rate."""
        ct = ConstantTurn(turn_rate=0.2, process_noise=0.01)
        rng = np.random.default_rng(42)
        state = np.array([0.0, 10.0, 0.0, 0.0])

        new_state = ct.propagate(state, dt=1.0, rng=rng)
        assert new_state.shape == (4,)
        # State should have changed
        assert not np.allclose(new_state, state, atol=0.1)

    def test_transition_matrix_shape(self) -> None:
        """CT transition matrix should be (4, 4)."""
        ct = ConstantTurn(turn_rate=0.1)
        F = ct.transition_matrix(dt=1.0)
        assert F.shape == (4, 4)

    def test_zero_turn_rate_equals_cv(self) -> None:
        """CT with turn_rate=0 should reduce to CV."""
        ct = ConstantTurn(turn_rate=0.0)
        cv = ConstantVelocity()

        F_ct = ct.transition_matrix(dt=1.0)
        F_cv = cv.transition_matrix(dt=1.0)

        np.testing.assert_allclose(F_ct, F_cv, atol=1e-10)

    def test_propagate_preserves_dimension(self) -> None:
        ct = ConstantTurn(turn_rate=0.3, process_noise=0.05)
        rng = np.random.default_rng(42)
        state = np.array([10.0, 5.0, 20.0, -3.0])

        new_state = ct.propagate(state, dt=0.5, rng=rng)
        assert new_state.shape == state.shape


class TestConstantAcceleration:
    """Test ConstantAcceleration motion model."""

    def test_propagate_preserves_dimension(self) -> None:
        """CA propagation should preserve the 6D state dimension."""
        ca = ConstantAcceleration(process_noise=0.01)
        rng = np.random.default_rng(42)
        state = np.array([0.0, 1.0, 0.5, 0.0, 2.0, -0.1])

        new_state = ca.propagate(state, dt=1.0, rng=rng)
        assert new_state.shape == (6,)

    def test_transition_matrix_shape(self) -> None:
        """CA transition matrix should be (6, 6)."""
        ca = ConstantAcceleration()
        F = ca.transition_matrix(dt=1.0)
        assert F.shape == (6, 6)

    def test_transition_matrix_acceleration_terms(self) -> None:
        """CA F should include 0.5*dt^2 acceleration terms."""
        ca = ConstantAcceleration()
        dt = 2.0
        F = ca.transition_matrix(dt)

        # Position integrates acceleration: F[0,2] = 0.5*dt^2
        assert np.isclose(F[0, 2], 0.5 * dt**2)
        assert np.isclose(F[3, 5], 0.5 * dt**2)
        # Velocity integrates acceleration: F[1,2] = dt
        assert np.isclose(F[1, 2], dt)
        assert np.isclose(F[4, 5], dt)


class TestTarget:
    """Test Target simulation entity."""

    def test_creation(self) -> None:
        """Target should store initial state and ID."""
        state = np.array([10.0, 1.0, 20.0, 0.5])
        target = Target(target_id=0, initial_state=state)

        assert target.target_id == 0
        np.testing.assert_array_equal(target.state, state)

    def test_step_changes_state(self) -> None:
        """Target.step should advance the state."""
        rng = np.random.default_rng(42)
        state = np.array([0.0, 5.0, 0.0, 3.0])
        target = Target(target_id=0, initial_state=state)

        old_state = target.state.copy()
        target.step(dt=1.0, rng=rng)
        assert not np.array_equal(target.state, old_state)

    def test_position_returns_first_half(self) -> None:
        """Target.position should return the first 2 elements of 4D state."""
        state = np.array([10.0, 1.0, 20.0, 0.5])
        target = Target(target_id=0, initial_state=state)
        np.testing.assert_array_equal(target.position, [10.0, 1.0])

    def test_trajectory_grows_with_steps(self) -> None:
        """Target.trajectory should grow as steps are taken."""
        rng = np.random.default_rng(42)
        state = np.array([0.0, 1.0, 0.0, 1.0])
        target = Target(target_id=0, initial_state=state)

        # Initial trajectory has 1 point (from __post_init__)
        assert target.trajectory.shape[0] == 1

        for _ in range(5):
            target.step(dt=1.0, rng=rng)

        assert target.trajectory.shape[0] == 6  # 1 initial + 5 steps
        assert target.trajectory.shape[1] == 4  # 4D state

    def test_custom_motion_model(self) -> None:
        """Target should use custom motion model if provided."""
        ct = ConstantTurn(turn_rate=0.5, process_noise=0.01)
        state = np.array([0.0, 10.0, 0.0, 0.0])
        target = Target(target_id=0, initial_state=state, motion_model=ct)

        rng = np.random.default_rng(42)
        target.step(dt=1.0, rng=rng)

        # State should have changed (CT produces different trajectory than CV)
        assert not np.allclose(target.state, state, atol=0.1)


class TestDrone:
    """Test Drone simulation entity."""

    def test_creation(self) -> None:
        """Drone should store position, sensor, and FOV."""
        pos = np.array([50.0, 50.0])
        drone = Drone(drone_id=0, position=pos, fov_radius=30.0)

        assert drone.drone_id == 0
        np.testing.assert_array_equal(drone.position, pos)
        assert drone.fov_radius == 30.0

    def test_default_velocity_is_zero(self) -> None:
        """Drone velocity should default to [0, 0]."""
        drone = Drone(drone_id=0, position=np.zeros(2))
        np.testing.assert_array_equal(drone.velocity, [0.0, 0.0])

    def test_can_observe_within_fov(self) -> None:
        """Targets within FOV should be observable."""
        drone = Drone(drone_id=0, position=np.array([0.0, 0.0]), fov_radius=50.0)
        target_pos = np.array([10.0, 10.0])  # dist = ~14.14 < 50
        assert drone.can_observe(target_pos) is True

    def test_cannot_observe_outside_fov(self) -> None:
        """Targets outside FOV should not be observable."""
        drone = Drone(drone_id=0, position=np.array([0.0, 0.0]), fov_radius=10.0)
        target_pos = np.array([100.0, 100.0])  # dist = ~141.4 > 10
        assert drone.can_observe(target_pos) is False

    def test_observe_within_fov_returns_measurement(self) -> None:
        """observe on a target within FOV should return a measurement (with P_D=1)."""
        rng = np.random.default_rng(42)
        sensor = LinearSensor(detection_probability=1.0)
        drone = Drone(drone_id=0, position=np.array([0.0, 0.0]),
                      sensor=sensor, fov_radius=100.0)
        target_state = np.array([10.0, 0.0, 20.0, 0.0])

        result = drone.observe(target_state, rng)
        assert result is not None
        assert result.shape == (2,)

    def test_observe_outside_fov_returns_none(self) -> None:
        """observe on a target outside FOV should return None."""
        rng = np.random.default_rng(42)
        drone = Drone(drone_id=0, position=np.array([0.0, 0.0]), fov_radius=5.0)
        target_state = np.array([100.0, 0.0, 100.0, 0.0])

        result = drone.observe(target_state, rng)
        assert result is None

    def test_step_moves_drone(self) -> None:
        """Drone.step should move position by velocity * dt."""
        drone = Drone(
            drone_id=0,
            position=np.array([0.0, 0.0]),
            velocity=np.array([5.0, 3.0]),
        )
        drone.step(dt=2.0)
        np.testing.assert_allclose(drone.position, [10.0, 6.0])


class TestSimulationWorld:
    """Test SimulationWorld discrete-time simulation."""

    def test_step_returns_measurement_scan(self) -> None:
        """SimulationWorld.step should return a MeasurementScan."""
        rng = np.random.default_rng(42)
        target = Target(target_id=0, initial_state=np.array([50.0, 1.0, 50.0, 0.5]))
        drone = Drone(drone_id=0, position=np.array([50.0, 50.0]), fov_radius=100.0,
                      sensor=LinearSensor(detection_probability=1.0))
        world = SimulationWorld(targets=[target], drones=[drone], clutter_rate=0.0)

        scan = world.step(rng)
        assert hasattr(scan, "measurements")
        assert hasattr(scan, "positions")

    def test_step_advances_time(self) -> None:
        """Each step should advance the world time by dt."""
        rng = np.random.default_rng(42)
        world = SimulationWorld(dt=0.5)
        assert world.time == 0.0

        world.step(rng)
        assert np.isclose(world.time, 0.5)

        world.step(rng)
        assert np.isclose(world.time, 1.0)

    def test_step_produces_target_measurements(self) -> None:
        """With a target in FOV and P_D=1, step should produce measurements."""
        rng = np.random.default_rng(42)
        target = Target(target_id=0, initial_state=np.array([50.0, 0.0, 50.0, 0.0]))
        drone = Drone(drone_id=0, position=np.array([50.0, 50.0]), fov_radius=100.0,
                      sensor=LinearSensor(detection_probability=1.0))
        world = SimulationWorld(targets=[target], drones=[drone], clutter_rate=0.0)

        scan = world.step(rng)
        # Should have at least one measurement from the target
        assert len(scan) >= 1

    def test_clutter_generation(self) -> None:
        """Non-zero clutter_rate should produce false alarm measurements."""
        rng = np.random.default_rng(42)
        world = SimulationWorld(targets=[], drones=[], clutter_rate=10.0)

        scan = world.step(rng)
        # With clutter_rate=10, expect some clutter measurements
        # (Poisson mean=10, very unlikely to be 0)
        assert len(scan) > 0


class TestDroneSwarm:
    """Test DroneSwarm coordinator."""

    def test_add_drone(self) -> None:
        """add_drone should add a drone to the swarm."""
        swarm = DroneSwarm()
        d1 = Drone(drone_id=0, position=np.zeros(2))
        d2 = Drone(drone_id=1, position=np.ones(2))
        swarm.add_drone(d1)
        swarm.add_drone(d2)

        assert len(swarm.drones) == 2

    def test_coverage_area(self) -> None:
        """coverage_area should sum pi*r^2 for each drone."""
        swarm = DroneSwarm()
        d1 = Drone(drone_id=0, position=np.zeros(2), fov_radius=10.0)
        d2 = Drone(drone_id=1, position=np.ones(2), fov_radius=20.0)
        swarm.add_drone(d1)
        swarm.add_drone(d2)

        expected = np.pi * 10.0**2 + np.pi * 20.0**2
        assert np.isclose(swarm.coverage_area, expected)

    def test_empty_swarm_coverage(self) -> None:
        """Empty swarm should have zero coverage area."""
        swarm = DroneSwarm()
        assert swarm.coverage_area == 0.0

    def test_patrol_formation_positions_drones(self) -> None:
        """patrol_formation should arrange drones in a circle."""
        swarm = DroneSwarm()
        for i in range(4):
            swarm.add_drone(Drone(drone_id=i, position=np.zeros(2)))

        swarm.patrol_formation(center=(50.0, 50.0), radius=20.0)

        # All drones should be 20 units from center
        for drone in swarm.drones:
            dist = np.linalg.norm(drone.position - np.array([50.0, 50.0]))
            assert np.isclose(dist, 20.0, atol=1e-10)


class TestScenarioGenerators:
    """Test predefined scenario generator functions."""

    def test_crossing_targets_returns_world(self) -> None:
        """crossing_targets should return a SimulationWorld with targets."""
        world = crossing_targets(n_targets=5)
        assert isinstance(world, SimulationWorld)
        assert len(world.targets) == 5

    def test_crossing_targets_has_drone(self) -> None:
        """crossing_targets should include at least one drone."""
        world = crossing_targets(n_targets=3)
        assert len(world.drones) >= 1

    def test_crossing_targets_different_sizes(self) -> None:
        """crossing_targets should work with different target counts."""
        for n in [2, 5, 10]:
            world = crossing_targets(n_targets=n)
            assert len(world.targets) == n

    def test_dense_clutter_returns_world(self) -> None:
        """dense_clutter should return a SimulationWorld."""
        world = dense_clutter(n_targets=3, clutter_rate=5.0)
        assert isinstance(world, SimulationWorld)
        assert len(world.targets) == 3
        assert world.clutter_rate == 5.0

    def test_dense_clutter_high_clutter_rate(self) -> None:
        """dense_clutter should preserve the configured clutter_rate."""
        world = dense_clutter(n_targets=2, clutter_rate=20.0)
        assert world.clutter_rate == 20.0

    def test_swarm_patrol_returns_world_with_drones(self) -> None:
        """swarm_patrol should return a SimulationWorld with multiple drones."""
        world = swarm_patrol(n_drones=3, n_targets=5)
        assert isinstance(world, SimulationWorld)
        assert len(world.drones) == 3
        assert len(world.targets) == 5

    def test_swarm_patrol_drone_positions(self) -> None:
        """swarm_patrol drones should have different positions."""
        world = swarm_patrol(n_drones=4, n_targets=2)
        positions = [tuple(d.position) for d in world.drones]
        # All positions should be unique
        assert len(set(positions)) == 4

    def test_scenarios_can_run_steps(self) -> None:
        """All scenario worlds should be able to execute simulation steps."""
        rng = np.random.default_rng(42)

        for world_fn in [
            lambda: crossing_targets(n_targets=3),
            lambda: dense_clutter(n_targets=2),
            lambda: swarm_patrol(n_drones=2, n_targets=3),
        ]:
            world = world_fn()
            scan = world.step(rng)
            assert hasattr(scan, "measurements")
