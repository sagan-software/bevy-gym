#!/usr/bin/env python3
"""Validate the checked-in Gymnasium oracle contracts with the Python stdlib."""

from __future__ import annotations

import json
import unittest
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ORACLE_COMMIT = "7a1191388aa4aa973d3a5e4b039899cd99cc991f"

CATALOG_IDS = {
    "Acrobot-v1",
    "CartPole-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "Pendulum-v1",
    "Blackjack-v1",
    "CliffWalking-v1",
    "FrozenLake-v1",
    "Taxi-v4",
}

REGISTERED_VARIANT_IDS = {"FrozenLake8x8-v1", "CliffWalkingSlippery-v1"}

CONFIGURATION_VARIANT_IDS = {
    "blackjack_legacy",
    "blackjack_natural_bonus",
    "blackjack_sab_natural_ignored",
    "taxi_rainy",
    "taxi_fickle_passenger",
    "taxi_rainy_fickle_passenger",
}


def discrete(size: int) -> dict:
    return {"type": "Discrete", "n": size, "start": 0}


def box(low: list, high: list, shape: list[int]) -> dict:
    return {
        "type": "Box",
        "low": low,
        "high": high,
        "shape": shape,
        "dtype": "float32",
    }


EXPECTED_SPACES = {
    "Acrobot-v1": {
        "observation": box(
            [-1, -1, -1, -1, -12.566371, -28.274334],
            [1, 1, 1, 1, 12.566371, 28.274334],
            [6],
        ),
        "action": discrete(3),
    },
    "CartPole-v1": {
        "observation": box(
            [-4.8, "-Infinity", -0.41887903, "-Infinity"],
            [4.8, "Infinity", 0.41887903, "Infinity"],
            [4],
        ),
        "action": discrete(2),
    },
    "MountainCar-v0": {
        "observation": box([-1.2, -0.07], [0.6, 0.07], [2]),
        "action": discrete(3),
    },
    "MountainCarContinuous-v0": {
        "observation": box([-1.2, -0.07], [0.6, 0.07], [2]),
        "action": box([-1], [1], [1]),
    },
    "Pendulum-v1": {
        "observation": box([-1, -1, -8], [1, 1, 8], [3]),
        "action": box([-2], [2], [1]),
    },
    "Blackjack-v1": {
        "observation": {
            "type": "Tuple",
            "spaces": [discrete(32), discrete(11), discrete(2)],
        },
        "action": discrete(2),
    },
    "CliffWalking-v1": {
        "observation": discrete(48),
        "action": discrete(4),
    },
    "CliffWalkingSlippery-v1": {
        "observation": discrete(48),
        "action": discrete(4),
    },
    "FrozenLake-v1": {
        "observation": discrete(16),
        "action": discrete(4),
    },
    "FrozenLake8x8-v1": {
        "observation": discrete(64),
        "action": discrete(4),
    },
    "Taxi-v4": {
        "observation": discrete(500),
        "action": discrete(6),
    },
}

EXPECTED_CONSTRUCTOR_DEFAULTS = {
    "Acrobot-v1": {},
    "CartPole-v1": {"sutton_barto_reward": False},
    "MountainCar-v0": {"goal_velocity": 0},
    "MountainCarContinuous-v0": {"goal_velocity": 0},
    "Pendulum-v1": {"g": 10.0},
    "Blackjack-v1": {"natural": False, "sab": False},
    "CliffWalking-v1": {"is_slippery": False},
    "CliffWalkingSlippery-v1": {"is_slippery": False},
    "FrozenLake-v1": {
        "desc": None,
        "map_name": "4x4",
        "is_slippery": True,
        "success_rate": 1.0 / 3.0,
        "reward_schedule": [1, 0, 0],
    },
    "FrozenLake8x8-v1": {
        "desc": None,
        "map_name": "4x4",
        "is_slippery": True,
        "success_rate": 1.0 / 3.0,
        "reward_schedule": [1, 0, 0],
    },
    "Taxi-v4": {
        "is_rainy": False,
        "fickle_passenger": False,
        "rainy_probability": 0.8,
        "fickle_probability": 0.3,
    },
}

EXPECTED_REGISTRY = {
    "Acrobot-v1": {
        "entry_point": "gymnasium.envs.classic_control.acrobot:AcrobotEnv",
        "default_kwargs": {},
        "max_episode_steps": 500,
        "reward_threshold": -100.0,
        "viewport": [500, 500],
        "fps": 15,
    },
    "CartPole-v1": {
        "entry_point": "gymnasium.envs.classic_control.cartpole:CartPoleEnv",
        "default_kwargs": {},
        "max_episode_steps": 500,
        "reward_threshold": 475.0,
        "viewport": [600, 400],
        "fps": 50,
    },
    "MountainCar-v0": {
        "entry_point": "gymnasium.envs.classic_control.mountain_car:MountainCarEnv",
        "default_kwargs": {},
        "max_episode_steps": 200,
        "reward_threshold": -110.0,
        "viewport": [600, 400],
        "fps": 30,
    },
    "MountainCarContinuous-v0": {
        "entry_point": "gymnasium.envs.classic_control.continuous_mountain_car:Continuous_MountainCarEnv",
        "default_kwargs": {},
        "max_episode_steps": 999,
        "reward_threshold": 90.0,
        "viewport": [600, 400],
        "fps": 30,
    },
    "Pendulum-v1": {
        "entry_point": "gymnasium.envs.classic_control.pendulum:PendulumEnv",
        "default_kwargs": {},
        "max_episode_steps": 200,
        "reward_threshold": None,
        "viewport": [500, 500],
        "fps": 30,
    },
    "Blackjack-v1": {
        "entry_point": "gymnasium.envs.toy_text.blackjack:BlackjackEnv",
        "default_kwargs": {"natural": False, "sab": True},
        "max_episode_steps": None,
        "reward_threshold": None,
        "viewport": [600, 500],
        "fps": 4,
    },
    "CliffWalking-v1": {
        "entry_point": "gymnasium.envs.toy_text.cliffwalking:CliffWalkingEnv",
        "default_kwargs": {},
        "max_episode_steps": None,
        "reward_threshold": None,
        "viewport": [720, 240],
        "fps": 4,
    },
    "FrozenLake-v1": {
        "entry_point": "gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv",
        "default_kwargs": {"map_name": "4x4"},
        "max_episode_steps": 100,
        "reward_threshold": 0.70,
        "viewport": [256, 256],
        "fps": 4,
    },
    "FrozenLake8x8-v1": {
        "entry_point": "gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv",
        "default_kwargs": {"map_name": "8x8"},
        "max_episode_steps": 200,
        "reward_threshold": 0.85,
        "viewport": [512, 512],
        "fps": 4,
    },
    "CliffWalkingSlippery-v1": {
        "entry_point": "gymnasium.envs.toy_text.cliffwalking:CliffWalkingEnv",
        "default_kwargs": {"is_slippery": True},
        "max_episode_steps": None,
        "reward_threshold": None,
        "viewport": [720, 240],
        "fps": 4,
    },
    "Taxi-v4": {
        "entry_point": "gymnasium.envs.toy_text.taxi:TaxiEnv",
        "default_kwargs": {},
        "max_episode_steps": 200,
        "reward_threshold": 8,
        "viewport": [550, 350],
        "fps": 4,
    },
}

EXPECTED_TABLES = {
    "transitions/frozen_lake_4x4.json": ("FrozenLake-v1", 16, 4),
    "transitions/frozen_lake_8x8.json": ("FrozenLake8x8-v1", 64, 4),
    "transitions/cliff_walking.json": ("CliffWalking-v1", 48, 4),
    "transitions/cliff_walking_slippery.json": (
        "CliffWalkingSlippery-v1",
        48,
        4,
    ),
    "transitions/taxi_dry.json": ("Taxi-v4", 500, 6),
    "transitions/taxi_rainy.json": ("taxi_rainy", 500, 6),
}


def read_json(relative_path: str) -> dict:
    with (ROOT / relative_path).open(encoding="utf-8") as fixture:
        return json.load(fixture)


def transition(table: dict, state: int, action: int) -> list[dict]:
    return next(
        entry["outcomes"]
        for entry in table["entries"]
        if entry["state"] == state and entry["action"] == action
    )


class GymnasiumFixtureContractTests(unittest.TestCase):
    def test_manifest_pins_oracle_and_rejects_rng_identity_claim(self) -> None:
        manifest = read_json("manifest.json")
        self.assertEqual(manifest["oracle"]["commit"], ORACLE_COMMIT)
        self.assertEqual(manifest["oracle"]["version"], "1.3.0")
        self.assertFalse(manifest["rng_policy"]["numpy_sequence_identity"])
        self.assertEqual(
            manifest["global_release_gate"],
            {
                "independent_training_seeds": 5,
                "minimum_passing_seeds": 4,
                "median_final_result_must_pass_environment_gate": True,
            },
        )
        self.assertEqual(set(manifest["transition_tables"]), set(EXPECTED_TABLES))
        self.assertEqual(
            manifest["auxiliary_tables"], ["transitions/taxi_action_masks.json"]
        )

    def test_catalog_records_all_registry_and_project_contracts(self) -> None:
        fixture = read_json("catalog.json")
        self.assertEqual(fixture["oracle_commit"], ORACLE_COMMIT)

        catalog = {item["registry_id"]: item for item in fixture["catalog"]}
        variants = {
            item["registry_id"]: item for item in fixture["registered_variants"]
        }
        self.assertEqual(set(catalog), CATALOG_IDS)
        self.assertEqual(set(variants), REGISTERED_VARIANT_IDS)

        for registry_id, expected in EXPECTED_REGISTRY.items():
            item = catalog.get(registry_id, variants.get(registry_id))
            self.assertIsNotNone(item, registry_id)
            assert item is not None
            registry = item["registry"]
            render = item["render"]
            for field in (
                "entry_point",
                "default_kwargs",
                "max_episode_steps",
                "reward_threshold",
            ):
                self.assertEqual(registry[field], expected[field], registry_id)
            self.assertEqual(render["viewport"], expected["viewport"], registry_id)
            self.assertEqual(render["fps"], expected["fps"], registry_id)
            self.assertEqual(item["spaces"], EXPECTED_SPACES[registry_id], registry_id)
            self.assertEqual(
                item["constructor_default_kwargs"],
                EXPECTED_CONSTRUCTOR_DEFAULTS[registry_id],
                registry_id,
            )

        config_ids = {item["contract_id"] for item in fixture["configuration_variants"]}
        self.assertEqual(config_ids, CONFIGURATION_VARIANT_IDS)
        self.assertEqual(
            variants["FrozenLake8x8-v1"]["effective_kwargs"],
            {
                "map_name": "8x8",
                "is_slippery": True,
                "success_rate": 1.0 / 3.0,
                "reward_schedule": [1, 0, 0],
            },
        )
        self.assertEqual(
            variants["CliffWalkingSlippery-v1"]["effective_kwargs"],
            {"is_slippery": True},
        )
        expected_config_kwargs = {
            "blackjack_legacy": {"sab": False, "natural": False},
            "blackjack_natural_bonus": {"sab": False, "natural": True},
            "blackjack_sab_natural_ignored": {"sab": True, "natural": True},
            "taxi_rainy": {
                "is_rainy": True,
                "fickle_passenger": False,
                "rainy_probability": 0.8,
                "fickle_probability": 0.3,
            },
            "taxi_fickle_passenger": {
                "is_rainy": False,
                "fickle_passenger": True,
                "rainy_probability": 0.8,
                "fickle_probability": 0.3,
            },
            "taxi_rainy_fickle_passenger": {
                "is_rainy": True,
                "fickle_passenger": True,
                "rainy_probability": 0.8,
                "fickle_probability": 0.3,
            },
        }
        for item in fixture["configuration_variants"]:
            self.assertIn("entry_point", item)
            self.assertIn("registry_default_kwargs", item)
            self.assertIn("constructor_default_kwargs", item)
            self.assertIn("effective_kwargs", item)
            self.assertIn("max_episode_steps", item)
            self.assertIn("reward_threshold", item)
            self.assertIn("spaces", item)
            self.assertIn("render", item)
            self.assertEqual(
                item["effective_kwargs"], expected_config_kwargs[item["contract_id"]]
            )
            if item["registry_id"] == "Blackjack-v1":
                self.assertIsNone(item["max_episode_steps"])
                self.assertIsNone(item["reward_threshold"])
                self.assertEqual(item["spaces"], EXPECTED_SPACES["Blackjack-v1"])
            else:
                self.assertEqual(item["max_episode_steps"], 200)
                self.assertEqual(item["reward_threshold"], 8)
                self.assertEqual(item["spaces"], EXPECTED_SPACES["Taxi-v4"])

    def test_project_gate_literals_match_plan(self) -> None:
        fixture = read_json("catalog.json")
        items = {
            item["registry_id"]: item
            for item in fixture["catalog"] + fixture["registered_variants"]
        }
        expected_conditions = {
            "Acrobot-v1": {"mean_return": -100, "target_reach_rate": 0.95},
            "CartPole-v1": {"mean_return": 475, "ceiling_hit_rate": 0.90},
            "MountainCar-v0": {"mean_return": -110, "position_0_5_reach_rate": 0.95},
            "MountainCarContinuous-v0": {
                "mean_return": 90,
                "position_0_45_reach_rate": 0.95,
            },
            "Pendulum-v1": {"mean_return": -200, "upright_dwell_rate": 0.70},
            "Blackjack-v1": {
                "mean_return_lower_95_confidence_bound": -0.08,
                "improvement_over_random": 0.25,
            },
            "CliffWalking-v1": {
                "return": -13,
                "goal_completion_rate": 1.0,
                "cliff_entries": 0,
            },
            "CliffWalkingSlippery-v1": {"success_lower_wilson_95_bound": 0.90},
            "FrozenLake-v1": {"success_lower_wilson_95_bound": 0.70},
            "FrozenLake8x8-v1": {"success_lower_wilson_95_bound": 0.85},
            "Taxi-v4": {
                "mean_return": 8,
                "delivery_rate": 1.0,
                "illegal_pickup_dropoff_actions": 0,
            },
        }
        for registry_id, conditions in expected_conditions.items():
            self.assertEqual(
                items[registry_id]["project_gate"]["conditions"],
                conditions,
                registry_id,
            )

        exact_condition_ids = {
            "CliffWalking-v1": {"return", "goal_completion_rate", "cliff_entries"},
            "Taxi-v4": {"delivery_rate", "illegal_pickup_dropoff_actions"},
        }
        for registry_id, item in items.items():
            operators = item["project_gate"]["operators"]
            for condition in item["project_gate"]["conditions"]:
                expected_operator = (
                    "=="
                    if condition in exact_condition_ids.get(registry_id, set())
                    else ">="
                )
                self.assertEqual(operators[condition], expected_operator, registry_id)

        self.assertEqual(items["Acrobot-v1"]["project_gate"]["episodes"], 100)
        for registry_id in (
            "Acrobot-v1",
            "CartPole-v1",
            "MountainCar-v0",
            "MountainCarContinuous-v0",
            "Pendulum-v1",
        ):
            self.assertEqual(items[registry_id]["project_gate"]["episodes"], 100)
        self.assertEqual(items["Blackjack-v1"]["project_gate"]["games"], 100_000)
        self.assertEqual(
            items["CliffWalkingSlippery-v1"]["project_gate"]["episodes"], 10_000
        )
        self.assertEqual(items["FrozenLake-v1"]["project_gate"]["episodes"], 10_000)
        self.assertEqual(items["FrozenLake8x8-v1"]["project_gate"]["episodes"], 10_000)
        self.assertEqual(items["Taxi-v4"]["project_gate"]["initial_states"], 300)
        self.assertEqual(items["Taxi-v4"]["project_gate"]["max_steps"], 200)
        self.assertEqual(
            items["CliffWalking-v1"]["project_gate"]["project_safety_cap_steps"],
            200,
        )
        self.assertEqual(
            items["Pendulum-v1"]["project_gate"]["upright_dwell_window"],
            {
                "start_step": 50,
                "end_step_inclusive": 199,
                "absolute_angle_degrees_lte": 15,
                "absolute_angular_velocity_rad_s_lte": 1,
            },
        )

    def test_transition_tables_are_total_and_probabilities_sum_to_one(self) -> None:
        for relative_path, (contract_id, states, actions) in EXPECTED_TABLES.items():
            table = read_json(relative_path)
            self.assertEqual(table["oracle_commit"], ORACLE_COMMIT, relative_path)
            self.assertEqual(table["contract_id"], contract_id, relative_path)
            self.assertEqual(table["state_count"], states, relative_path)
            self.assertEqual(table["action_count"], actions, relative_path)
            self.assertEqual(len(table["entries"]), states * actions, relative_path)
            initial_total = sum(
                Fraction(
                    item["probability"]["numerator"],
                    item["probability"]["denominator"],
                )
                for item in table["initial_states"]
            )
            self.assertEqual(initial_total, 1, relative_path)
            self.assertEqual(
                len({item["state"] for item in table["initial_states"]}),
                len(table["initial_states"]),
                relative_path,
            )
            keys = {(entry["state"], entry["action"]) for entry in table["entries"]}
            self.assertEqual(
                keys,
                {
                    (state, action)
                    for state in range(states)
                    for action in range(actions)
                },
                relative_path,
            )
            for entry in table["entries"]:
                total = sum(
                    Fraction(
                        outcome["probability"]["numerator"],
                        outcome["probability"]["denominator"],
                    )
                    for outcome in entry["outcomes"]
                )
                self.assertEqual(total, 1, (relative_path, entry))
                for outcome in entry["outcomes"]:
                    self.assertGreaterEqual(outcome["next_state"], 0)
                    self.assertLess(outcome["next_state"], states)
                    self.assertIsInstance(outcome["terminated"], bool)

    def test_transition_table_sentinels_match_pinned_source_rules(self) -> None:
        one = {"numerator": 1, "denominator": 1}
        one_third = {"numerator": 1, "denominator": 3}

        frozen = read_json("transitions/frozen_lake_4x4.json")
        self.assertEqual(
            transition(frozen, 14, 2),
            [
                {
                    "probability": one_third,
                    "next_state": 14,
                    "reward": 0,
                    "terminated": False,
                },
                {
                    "probability": one_third,
                    "next_state": 15,
                    "reward": 1,
                    "terminated": True,
                },
                {
                    "probability": one_third,
                    "next_state": 10,
                    "reward": 0,
                    "terminated": False,
                },
            ],
        )

        frozen_8x8 = read_json("transitions/frozen_lake_8x8.json")
        self.assertEqual(
            transition(frozen_8x8, 54, 0),
            [{"probability": one, "next_state": 54, "reward": 0, "terminated": True}],
        )

        cliff = read_json("transitions/cliff_walking.json")
        self.assertEqual(
            transition(cliff, 36, 1),
            [
                {
                    "probability": one,
                    "next_state": 36,
                    "reward": -100,
                    "terminated": False,
                }
            ],
        )

        cliff_slippery = read_json("transitions/cliff_walking_slippery.json")
        self.assertEqual(
            transition(cliff_slippery, 36, 0),
            [
                {
                    "probability": one_third,
                    "next_state": 36,
                    "reward": -1,
                    "terminated": False,
                },
                {
                    "probability": one_third,
                    "next_state": 24,
                    "reward": -1,
                    "terminated": False,
                },
                {
                    "probability": one_third,
                    "next_state": 36,
                    "reward": -100,
                    "terminated": False,
                },
            ],
        )

        taxi = read_json("transitions/taxi_dry.json")
        self.assertEqual(len(taxi["initial_states"]), 300)
        self.assertEqual(
            transition(taxi, 1, 4),
            [{"probability": one, "next_state": 17, "reward": -1, "terminated": False}],
        )
        self.assertEqual(
            transition(taxi, 97, 5),
            [{"probability": one, "next_state": 85, "reward": 20, "terminated": True}],
        )

        taxi_rainy = read_json("transitions/taxi_rainy.json")
        self.assertEqual(
            transition(taxi_rainy, 1, 0),
            [
                {
                    "probability": {"numerator": 4, "denominator": 5},
                    "next_state": 101,
                    "reward": -1,
                    "terminated": False,
                },
                {
                    "probability": {"numerator": 1, "denominator": 10},
                    "next_state": 21,
                    "reward": -1,
                    "terminated": False,
                },
                {
                    "probability": {"numerator": 1, "denominator": 10},
                    "next_state": 1,
                    "reward": -1,
                    "terminated": False,
                },
            ],
        )
        self.assertEqual(
            [outcome["next_state"] for outcome in transition(taxi_rainy, 21, 2)],
            [21, 21, 21],
        )

    def test_taxi_action_masks_cover_all_encoded_states(self) -> None:
        fixture = read_json("transitions/taxi_action_masks.json")
        self.assertEqual(fixture["oracle_commit"], ORACLE_COMMIT)
        self.assertEqual(fixture["contract_id"], "Taxi-v4")
        self.assertEqual(fixture["state_count"], 500)
        self.assertEqual(len(fixture["entries"]), 500)
        self.assertEqual(
            [entry["state"] for entry in fixture["entries"]], list(range(500))
        )
        masks = {entry["state"]: entry["mask"] for entry in fixture["entries"]}
        self.assertEqual(masks[1], [1, 0, 1, 0, 1, 0])
        self.assertEqual(masks[17], [1, 0, 1, 0, 0, 1])
        self.assertEqual(masks[21], [1, 0, 0, 1, 0, 0])

    def test_explicit_transition_cases_cover_all_nine_catalog_entries(self) -> None:
        fixture = read_json("transition_cases.json")
        self.assertEqual(fixture["oracle_commit"], ORACLE_COMMIT)
        covered = {case["registry_id"] for case in fixture["cases"]}
        self.assertTrue(CATALOG_IDS <= covered)
        self.assertIn("CliffWalkingSlippery-v1", covered)
        self.assertIn("FrozenLake8x8-v1", covered)
        self.assertTrue(
            all(
                "seed" not in case and "rng_sequence" not in case
                for case in fixture["cases"]
            )
        )
        for case in fixture["cases"]:
            outcomes = case["expected"].get("outcomes")
            if outcomes is not None:
                self.assertEqual(
                    sum(
                        Fraction(
                            outcome["probability"]["numerator"],
                            outcome["probability"]["denominator"],
                        )
                        for outcome in outcomes
                    ),
                    1,
                    case["case_id"],
                )
        cases = {case["case_id"]: case for case in fixture["cases"]}
        covered_contracts = {case["contract_id"] for case in fixture["cases"]}
        self.assertTrue(CONFIGURATION_VARIANT_IDS <= covered_contracts)
        self.assertEqual(
            cases["acrobot_zero_state_negative_torque"]["injection"]["internal_state"],
            [0, 0, 0, 0],
        )
        self.assertEqual(
            cases["taxi_legal_pickup_at_red"]["expected"]["action_mask_after"],
            [1, 0, 1, 0, 0, 1],
        )

    def test_generator_support_manifest_is_explicit_about_remaining_work(self) -> None:
        fixture = read_json("generator_support.json")
        self.assertEqual(fixture["oracle_commit"], ORACLE_COMMIT)
        expected = {
            "gymnasium_oracle_regeneration",
            "blackjack_hidden_hand_transition_enumeration",
            "taxi_fickle_augmented_state_enumeration",
            "statistical_reset_and_stochastic_transition_checks",
        }
        self.assertEqual({item["id"] for item in fixture["requirements"]}, expected)
        for item in fixture["requirements"]:
            self.assertEqual(item["status"], "requires_generator_support")
            self.assertTrue(item["source_paths"])
            self.assertTrue(item["required_output"])
            self.assertTrue(item["reason"])


if __name__ == "__main__":
    unittest.main()
