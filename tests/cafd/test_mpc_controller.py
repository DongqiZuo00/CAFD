"""Contract tests for CAFD-MPC specification Appendix A, no torch/GPU."""
import copy
import json

import numpy as np
import pytest

from cafd.mpc_controller import (
    MPCConfig, MPCController, build_features, effective_increment,
    project_state, select_forecast, weighted_ridge,
)


def config(**kwargs):
    values = dict(route_intervals=5, total_rounds=200, family_weights=(0.25, 0.75),
                  exploration_probability=0.0)
    values.update(kwargs)
    return MPCConfig(**values)


def state():
    return np.array([[0.2, 0.1, 0.8], [0.6, 0.3, 0.4]])


def complete(controller, new_state=None):
    choice = controller.choose_action()
    controller.observe(state() if new_state is None else new_state)
    return choice


def test_exact_feature_order_dimensions_and_weighted_global_state():
    cfg = config()
    s = state()
    previous = s - np.array([0.01, 0.02, 0.03])
    x = build_features(s, previous, cfg.family_weights, 1.5, 0.5, 1, 40, cfg)
    assert x.shape == (2, 19)
    expected = np.concatenate(([1, 0], s[0], [0.01, 0.02, 0.03],
                               [0.5, 0.25, 0.5], [0.3, 0.2, 0.5, 1, 0.3], s[0]))
    np.testing.assert_allclose(x[0], expected, atol=1e-15)
    # Shortened final progress must affect every candidate interaction.
    short = build_features(s, previous, cfg.family_weights, 4.75, 1, 1, 190, cfg)
    assert short[0, 13] == 1  # previous b
    assert short[0, 14] == 0.5  # candidate b
    np.testing.assert_allclose(short[0, -3:], 0.5 * s[0])
    assert effective_increment(5, 1, 5, 0.5) == 0


def test_projection_uses_raw_coordinate_mean_before_clipping():
    raw = np.array([[-2., 3., 5.], [2., -1., -2.], [.4, .8, .5], [-4., -2., .2]])
    expected = [[.5, .5, 1.], [1., 0., 0.], [.6, .6, .5], [0., 0., .2]]
    np.testing.assert_allclose(project_state(raw), expected)
    assert raw[0, 0] == -2


def test_weighted_ridge_hand_solution_all_coordinates_penalized():
    # Diagonal Gram: diag(1, 2) plus I; hand W is diag(1, 2).
    x = np.eye(2)
    y = np.array([[2., 0.], [0., 3.]])
    np.testing.assert_allclose(weighted_ridge(x, y, [1., 2.], 1.), np.diag([1., 2.]))


def test_ridge_full_transition_fit_family_weights_and_forgetting():
    c = MPCController(config(forgetting_factor=0.5), state())
    s1 = state() + np.array([0.02, 0.01, -0.03])
    d0 = c.choose_action()
    c.observe(s1)
    s2 = s1 + np.array([0.03, 0.01, -0.04])
    d1 = c.choose_action()
    c.observe(s2)
    features = np.concatenate((d0['features'], d1['features']))
    changes = np.concatenate((s1 - state(), s2 - s1))
    weights = [0.125, 0.375, 0.25, 0.75]
    expected = weighted_ridge(features, changes, weights, 1.0)
    np.testing.assert_allclose(c.W, expected)
    assert c.transitions[0]['valid']  # initial duplicated boundary permits j=0


def test_initialization_is_nonzero_and_seeded_paired_sequence():
    c = MPCController(config(route_intervals=1), state())
    assert c.u == 0.125
    np.testing.assert_array_equal(c.W, np.zeros((3, 19)))
    seq = [complete(c)['action'] for _ in range(6)]
    assert seq[:2] == [0, 1]
    assert sorted(seq[2:4]) == sorted(seq[4:6]) == [0, 1]
    assert c.u == 0.5
    assert c.predictive_decisions == 0
    assert MPCController(config(route_intervals=1), state()).initial_actions == seq


def test_initialization_short_run_uses_prefix_without_fake_predictions():
    c = MPCController(config(total_rounds=30), state())
    seq = [complete(c)['action'] for _ in range(3)]
    assert seq == c.initial_actions[:3]
    assert c.done and c.predictive_decisions == 0


def test_missing_three_boundary_rule_and_actual_block_index():
    c = MPCController(config(), state())
    c.choose_action()
    c.observe([[.2, .1, None], [.6, .3, .4]])
    assert c.state is None and c.transitions[0]['valid'] is False
    c.choose_action()
    assert c.pending['kind'] == 'missing_observation_hold'
    c.observe(state())
    c.choose_action()
    assert c.pending['kind'] == 'missing_observation_hold'
    c.observe(state())
    restored = complete(c)
    assert restored['kind'] == 'initialization'
    assert restored['action'] == c.initial_actions[3]
    assert [tr['valid'] for tr in c.transitions] == [False, False, False, True]


def test_missing_initial_boundary_holds_until_two_complete_boundaries():
    c = MPCController(config(), None)
    assert complete(c)['kind'] == 'missing_observation_hold'
    assert complete(c)['kind'] == 'missing_observation_hold'
    assert complete(c)['kind'] == 'initialization'
    assert [t['valid'] for t in c.transitions] == [False, False, True]


def test_tie_selection_uses_global_reward_then_error_then_lexical_advance():
    rows = [
        dict(actions=[0, 1], reward=0.5, error=0.2),
        dict(actions=[1, 0], reward=0.5 - 0.00009, error=0.2 + 1e-9),
        dict(actions=[1, 1], reward=0.5 - 0.00011, error=0.0),
    ]
    assert select_forecast(rows)['actions'] == [1, 0]
    rows[0]['error'] = 0.1
    assert select_forecast(rows)['actions'] == [0, 1]


def test_zero_predictor_prefers_early_advance_and_counts_actual_mpc():
    c = MPCController(config(), state())
    for _ in range(6):
        complete(c)
    c.W.fill(0)
    d = c.choose_action()
    assert d['kind'] == 'mpc' and d['action'] == 1
    assert len(d['forecasts']) == 4
    assert c.predictive_decisions == c.mpc_multiaction_decisions == 1
    assert c.choose_action() == d  # pending decision is idempotent, no RNG/counter change
    assert c.predictive_decisions == 1


def test_joint_recursive_forecast_updates_lags_time_position_and_actions():
    c = MPCController(config(), state())
    # reward increment = .01*global reward + .1*previous effective action
    # + .01*time + .02*zeta + .01*current local reward delta.
    c.W[0, 8] = .01  # D + 6: weighted reward
    c.W[0, 13] = .1
    c.W[0, 12] = .01
    c.W[0, 11] = .02
    c.W[0, 5] = .01  # D+3: local reward delta
    before = copy.deepcopy(c.state_dict())
    result = next(v for v in c.forecast_candidates() if v['actions'] == [1, 0])
    # Compute second step explicitly via the published features (W stays fixed).
    x1 = c._features(c.state, c.previous_state, c.u, 0, 1, 0)
    s1 = project_state(c.state + x1 @ c.W.T)
    x2 = c._features(s1, c.state, c.u + c.config.delta, 1, 0, 10)
    s2 = project_state(s1 + x2 @ c.W.T)
    np.testing.assert_allclose(result['terminal_state'], s2)
    np.testing.assert_allclose(np.asarray(result['terminal_state'])[:, 0], [.31664, .71664])
    assert c.state_dict() == before


def test_saturation_deduplicates_paths_and_endpoint_only_holds():
    c = MPCController(config(), state())
    c.u = 4.75
    assert sorted(v['actions'] for v in c.forecast_candidates()) == [[0, 0], [0, 1], [1, 0]]
    c.u = 5
    assert c.admissible_actions() == [0]
    assert [v['actions'] for v in c.forecast_candidates()] == [[0, 0]]
    assert complete(c)['kind'] == 'endpoint_hold'
    assert c.u == 5 and c.mpc_multiaction_decisions == 0


def test_horizon_truncation_and_partial_block_hold_no_fit():
    c = MPCController(config(total_rounds=75), state())
    for _ in range(6):
        complete(c)
    choice = complete(c)
    assert all(len(v['actions']) == 1 for v in choice['forecasts'])
    w = c.W.copy()
    u = c.u
    partial = c.choose_action()
    assert partial['kind'] == 'partial_hold' and partial['rounds'] == 5
    assert partial['u'] == u and partial['forecasts'] == []
    c.observe(state() + [0.1, 0.1, 0])
    np.testing.assert_array_equal(c.W, w)
    assert len(c.transitions) == 7 and c.completed_rounds == 75 and c.done
    with pytest.raises(RuntimeError):
        c.choose_action()


def test_external_cap_partial_observation_does_not_fit_or_continue():
    c = MPCController(config(), state())
    c.choose_action()
    c.observe(state(), rounds=3)
    assert c.done and c.stop_reason == 'external_compute_cap'
    assert c.completed_rounds == 3 and not c.transitions


def test_json_resume_pending_block_and_future_rng_decisions_identical():
    c = MPCController(config(total_rounds=200, exploration_probability=0.5), state())
    for _ in range(7):
        complete(c)
    c.choose_action()
    saved = json.loads(json.dumps(c.state_dict(), allow_nan=False))
    restored = MPCController.from_state_dict(saved)
    assert restored.choose_action() == c.choose_action()
    c.observe(state())
    restored.observe(state())
    for _ in range(10):
        assert c.choose_action() == restored.choose_action()
        c.observe(state())
        restored.observe(state())
    assert restored.state_dict() == c.state_dict()


def test_invalid_config_or_complete_state_is_not_silently_changed():
    with pytest.raises(ValueError):
        config(family_weights=(1, 1))
    with pytest.raises(ValueError):
        config(family_weights=(0, 1))
    with pytest.raises(ValueError):
        config(ridge=0)
    with pytest.raises(ValueError):
        MPCController(config(), [[0.2, 0.3, 0.1], [0.4, 0.1, 0.5]])


def test_missing_transition_ages_previous_data_in_real_block_time():
    c = MPCController(config(forgetting_factor=0.5), state())
    first = c.choose_action()
    s1 = state() + [0.02, 0.01, -0.03]
    c.observe(s1)
    c.choose_action()
    c.observe(None)
    expected = weighted_ridge(first['features'], s1 - state(), [.125, .375], 1.0)
    np.testing.assert_allclose(c.W, expected)
    assert c.completed_full_blocks == 2 and len(c.transitions) == 2
    assert c.transitions[1]['valid'] is False


def test_always_random_exploration_is_not_counted_as_predictive_decision():
    c = MPCController(config(exploration_probability=1), state())
    for _ in range(6):
        complete(c)
    for _ in range(3):
        row = complete(c)
        assert row['kind'] == 'random_exploration' and not row['forecasts']
    assert c.randomized_decisions == 3 and c.predictive_decisions == 0
    assert c.mpc_multiaction_decisions == 0


def test_missing_boundary_and_pending_hold_are_strict_json_serializable():
    c = MPCController(config(), [[.2, .1, float('nan')], [.6, .3, .4]])
    decision = c.choose_action()
    encoded = json.loads(json.dumps(c.state_dict(), allow_nan=False))
    assert encoded['state'] is None and encoded['previous_state'] is None
    restored = MPCController.from_state_dict(encoded)
    assert restored.choose_action() == decision
    restored.observe(state())
    assert restored.choose_action()['kind'] == 'missing_observation_hold'
