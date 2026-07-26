#!/usr/bin/env python3
"""Validate the versioned training-artifact JSON Schemas and test fixtures.

This intentionally uses only the Python standard library.  It checks the
Draft 2020-12 vocabulary subset used by this directory, resolves every local
``$ref``, and exercises each public schema with valid and invalid fixtures.
It is not a replacement for validation against the official Draft 2020-12
meta-schema; that remains a separate tooling decision.
"""

from __future__ import annotations

import copy
import json
import math
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
TESTS = ROOT / "tests"
DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
PUBLIC_SCHEMAS = {
    "config.schema.json",
    "seeds.schema.json",
    "metrics-row.schema.json",
    "eval-row.schema.json",
    "provenance.schema.json",
    "summary.schema.json",
    "proof.schema.json",
    "video-manifest.schema.json",
}


class SchemaCheckError(Exception):
    """A schema document is malformed or inconsistent."""


class InstanceValidationError(Exception):
    """An instance does not satisfy its selected schema."""


def load_json(path: Path) -> Any:
    """Load one JSON file while rejecting duplicate object keys."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SchemaCheckError(f"{path}: duplicate object key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> Any:
        raise SchemaCheckError(f"{path}: non-standard JSON number {value!r}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=reject_duplicates,
                parse_constant=reject_nonfinite,
            )
    except json.JSONDecodeError as error:
        raise SchemaCheckError(f"{path}: invalid JSON: {error}") from error


def json_pointer(document: Any, fragment: str, source: str) -> Any:
    """Resolve an RFC 6901 JSON pointer fragment."""

    if not fragment:
        return document
    if not fragment.startswith("/"):
        raise SchemaCheckError(f"{source}: unsupported non-pointer fragment #{fragment}")
    current = document
    for encoded_token in fragment[1:].split("/"):
        token = encoded_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
        else:
            raise SchemaCheckError(f"{source}: unresolved JSON pointer token {token!r}")
    return current


class SchemaRegistry:
    """Loaded schema documents indexed by path and canonical ``$id``."""

    def __init__(self, schema_paths: list[Path]) -> None:
        self.documents = {path.resolve(): load_json(path) for path in schema_paths}
        self.ids: dict[str, Path] = {}
        for path, document in self.documents.items():
            identifier = document.get("$id")
            if not isinstance(identifier, str):
                raise SchemaCheckError(f"{path}: root $id must be a string")
            if identifier in self.ids:
                raise SchemaCheckError(
                    f"{path}: duplicate $id {identifier!r} also used by {self.ids[identifier]}"
                )
            self.ids[identifier] = path

    def resolve(self, reference: str, base_path: Path) -> tuple[Any, Path]:
        """Resolve a local path or known canonical-id reference."""

        target_name, separator, fragment = reference.partition("#")
        if target_name in self.ids:
            target_path = self.ids[target_name]
        elif not target_name:
            target_path = base_path.resolve()
        elif "://" in target_name:
            raise SchemaCheckError(
                f"{base_path}: remote $ref {reference!r} is not locally resolvable"
            )
        else:
            target_path = (base_path.parent / target_name).resolve()
        if target_path not in self.documents:
            raise SchemaCheckError(f"{base_path}: missing $ref target {target_path}")
        document = self.documents[target_path]
        target = json_pointer(document, fragment if separator else "", reference)
        return target, target_path


SCHEMA_MAP_KEYS = {
    "$defs",
    "dependentSchemas",
    "patternProperties",
    "properties",
}
SCHEMA_SINGLE_KEYS = {
    "additionalProperties",
    "contains",
    "else",
    "if",
    "items",
    "not",
    "propertyNames",
    "then",
    "unevaluatedProperties",
}
SCHEMA_ARRAY_KEYS = {"allOf", "anyOf", "oneOf", "prefixItems"}
ANNOTATION_KEYS = {
    "$comment",
    "$id",
    "$schema",
    "$ref",
    "default",
    "deprecated",
    "description",
    "examples",
    "readOnly",
    "title",
    "writeOnly",
    "x-bevy-gym-invariants",
}
VALUE_KEYS = {
    "const",
    "dependentRequired",
    "enum",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "format",
    "maxContains",
    "maximum",
    "maxItems",
    "maxLength",
    "maxProperties",
    "minContains",
    "minimum",
    "minItems",
    "minLength",
    "minProperties",
    "multipleOf",
    "pattern",
    "required",
    "type",
    "uniqueItems",
}
KNOWN_KEYS = SCHEMA_MAP_KEYS | SCHEMA_SINGLE_KEYS | SCHEMA_ARRAY_KEYS | ANNOTATION_KEYS | VALUE_KEYS
VALID_TYPES = {"array", "boolean", "integer", "null", "number", "object", "string"}


def check_schema_node(node: Any, path: str, registry: SchemaRegistry, base_path: Path) -> None:
    """Check keyword shapes for the supported Draft 2020-12 subset."""

    if isinstance(node, bool):
        return
    if not isinstance(node, dict):
        raise SchemaCheckError(f"{path}: schema must be an object or boolean")
    unknown = set(node) - KNOWN_KEYS
    if unknown:
        raise SchemaCheckError(f"{path}: unsupported or misspelled keywords: {sorted(unknown)}")

    if "$ref" in node:
        if not isinstance(node["$ref"], str):
            raise SchemaCheckError(f"{path}/$ref: expected string")
        registry.resolve(node["$ref"], base_path)

    type_value = node.get("type")
    if type_value is not None:
        types = [type_value] if isinstance(type_value, str) else type_value
        if (
            not isinstance(types, list)
            or not types
            or not all(isinstance(item, str) and item in VALID_TYPES for item in types)
            or len(types) != len(set(types))
        ):
            raise SchemaCheckError(f"{path}/type: invalid or duplicate JSON type")

    required = node.get("required")
    if required is not None:
        if (
            not isinstance(required, list)
            or not all(isinstance(item, str) for item in required)
            or len(required) != len(set(required))
        ):
            raise SchemaCheckError(f"{path}/required: expected unique string array")
        properties = node.get("properties")
        if isinstance(properties, dict) and not set(required).issubset(properties):
            missing = sorted(set(required) - set(properties))
            raise SchemaCheckError(f"{path}/required: names absent from properties: {missing}")

    if node.get("type") == "object" and "properties" in node:
        if node.get("additionalProperties") is not False:
            raise SchemaCheckError(
                f"{path}: object with named properties must set additionalProperties=false"
            )

    if "enum" in node:
        values = node["enum"]
        if not isinstance(values, list) or not values:
            raise SchemaCheckError(f"{path}/enum: expected non-empty array")
        encoded = [json.dumps(value, sort_keys=True) for value in values]
        if len(encoded) != len(set(encoded)):
            raise SchemaCheckError(f"{path}/enum: duplicate values")

    if "pattern" in node:
        if not isinstance(node["pattern"], str):
            raise SchemaCheckError(f"{path}/pattern: expected string")
        try:
            re.compile(node["pattern"])
        except re.error as error:
            raise SchemaCheckError(
                f"{path}/pattern: invalid regular expression: {error}"
            ) from error

    for key in (
        "maxContains",
        "maxItems",
        "maxLength",
        "maxProperties",
        "minContains",
        "minItems",
        "minLength",
        "minProperties",
    ):
        if key in node and (
            not isinstance(node[key], int) or isinstance(node[key], bool) or node[key] < 0
        ):
            raise SchemaCheckError(f"{path}/{key}: expected non-negative integer")
    if "multipleOf" in node and (
        not isinstance(node["multipleOf"], (int, float))
        or isinstance(node["multipleOf"], bool)
        or node["multipleOf"] <= 0
    ):
        raise SchemaCheckError(f"{path}/multipleOf: expected positive number")

    for key in SCHEMA_MAP_KEYS:
        if key not in node:
            continue
        value = node[key]
        if not isinstance(value, dict):
            raise SchemaCheckError(f"{path}/{key}: expected object")
        for name, child in value.items():
            if key == "patternProperties":
                try:
                    re.compile(name)
                except re.error as error:
                    raise SchemaCheckError(
                        f"{path}/{key}/{name}: invalid regular expression: {error}"
                    ) from error
            check_schema_node(child, f"{path}/{key}/{name}", registry, base_path)

    for key in SCHEMA_SINGLE_KEYS:
        if key in node:
            check_schema_node(node[key], f"{path}/{key}", registry, base_path)

    for key in SCHEMA_ARRAY_KEYS:
        if key not in node:
            continue
        value = node[key]
        if not isinstance(value, list) or not value:
            raise SchemaCheckError(f"{path}/{key}: expected non-empty schema array")
        for index, child in enumerate(value):
            check_schema_node(child, f"{path}/{key}/{index}", registry, base_path)


def type_matches(instance: Any, expected: str) -> bool:
    """Return whether a Python JSON value has the requested JSON type."""

    return {
        "null": instance is None,
        "boolean": isinstance(instance, bool),
        "integer": isinstance(instance, int) and not isinstance(instance, bool),
        "number": isinstance(instance, (int, float)) and not isinstance(instance, bool),
        "string": isinstance(instance, str),
        "array": isinstance(instance, list),
        "object": isinstance(instance, dict),
    }[expected]


def instance_path(parts: tuple[str, ...]) -> str:
    return "$" + "".join(f"[{part}]" if part.isdigit() else f".{part}" for part in parts)


def validate_format(value: str, format_name: str, where: str) -> None:
    """Validate the formats used by these schema documents."""

    if format_name == "date-time":
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise InstanceValidationError(f"{where}: invalid date-time") from error
        if parsed.tzinfo is None:
            raise InstanceValidationError(f"{where}: date-time must include an offset")
    elif format_name == "uuid":
        try:
            parsed_uuid = uuid.UUID(value)
        except ValueError as error:
            raise InstanceValidationError(f"{where}: invalid UUID") from error
        if str(parsed_uuid) != value.lower():
            raise InstanceValidationError(f"{where}: UUID must use canonical hyphenated form")
    elif format_name == "uint64-decimal":
        if re.fullmatch(r"0|[1-9][0-9]{0,19}", value) is None:
            raise InstanceValidationError(f"{where}: invalid uint64 decimal text")
        if int(value) > (1 << 64) - 1:
            raise InstanceValidationError(f"{where}: value exceeds uint64 maximum")
    else:
        raise SchemaCheckError(f"unsupported format used by schema: {format_name}")


def is_valid(instance: Any, schema: Any, registry: SchemaRegistry, base_path: Path) -> bool:
    try:
        validate_instance(instance, schema, registry, base_path, ())
    except InstanceValidationError:
        return False
    return True


def validate_instance(
    instance: Any,
    schema: Any,
    registry: SchemaRegistry,
    base_path: Path,
    path: tuple[str, ...],
) -> None:
    """Validate an instance against the vocabulary subset used in this directory."""

    where = instance_path(path)
    if schema is True:
        return
    if schema is False:
        raise InstanceValidationError(f"{where}: rejected by false schema")
    if "$ref" in schema:
        target, target_path = registry.resolve(schema["$ref"], base_path)
        validate_instance(instance, target, registry, target_path, path)

    if "type" in schema:
        expected = schema["type"]
        expected_types = [expected] if isinstance(expected, str) else expected
        if not any(type_matches(instance, item) for item in expected_types):
            raise InstanceValidationError(
                f"{where}: expected type {' or '.join(expected_types)}, got {type(instance).__name__}"
            )
    if "const" in schema and instance != schema["const"]:
        raise InstanceValidationError(f"{where}: expected constant {schema['const']!r}")
    if "enum" in schema and instance not in schema["enum"]:
        raise InstanceValidationError(f"{where}: value is not in enum")

    for child in schema.get("allOf", []):
        validate_instance(instance, child, registry, base_path, path)
    if "anyOf" in schema and not any(
        is_valid(instance, child, registry, base_path) for child in schema["anyOf"]
    ):
        raise InstanceValidationError(f"{where}: no anyOf branch matched")
    if "oneOf" in schema:
        matches = sum(is_valid(instance, child, registry, base_path) for child in schema["oneOf"])
        if matches != 1:
            raise InstanceValidationError(
                f"{where}: expected exactly one oneOf match, got {matches}"
            )
    if "not" in schema and is_valid(instance, schema["not"], registry, base_path):
        raise InstanceValidationError(f"{where}: matched forbidden schema")
    if "if" in schema:
        branch = "then" if is_valid(instance, schema["if"], registry, base_path) else "else"
        if branch in schema:
            validate_instance(instance, schema[branch], registry, base_path, path)

    if isinstance(instance, str):
        if len(instance) < schema.get("minLength", 0):
            raise InstanceValidationError(f"{where}: string shorter than minLength")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            raise InstanceValidationError(f"{where}: string longer than maxLength")
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            raise InstanceValidationError(f"{where}: string does not match pattern")
        if "format" in schema:
            validate_format(instance, schema["format"], where)

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if not math.isfinite(instance):
            raise InstanceValidationError(f"{where}: number must be finite")
        if "minimum" in schema and instance < schema["minimum"]:
            raise InstanceValidationError(f"{where}: number is below minimum")
        if "maximum" in schema and instance > schema["maximum"]:
            raise InstanceValidationError(f"{where}: number is above maximum")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            raise InstanceValidationError(f"{where}: number is not above exclusiveMinimum")
        if "exclusiveMaximum" in schema and instance >= schema["exclusiveMaximum"]:
            raise InstanceValidationError(f"{where}: number is not below exclusiveMaximum")
        if "multipleOf" in schema:
            quotient = instance / schema["multipleOf"]
            if not math.isclose(quotient, round(quotient), rel_tol=1e-9, abs_tol=1e-9):
                raise InstanceValidationError(f"{where}: number is not a multipleOf")

    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            raise InstanceValidationError(f"{where}: array shorter than minItems")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            raise InstanceValidationError(f"{where}: array longer than maxItems")
        if schema.get("uniqueItems"):
            encoded = [json.dumps(value, sort_keys=True) for value in instance]
            if len(encoded) != len(set(encoded)):
                raise InstanceValidationError(f"{where}: array items are not unique")
        prefix_items = schema.get("prefixItems", [])
        for index, child in enumerate(prefix_items):
            if index < len(instance):
                validate_instance(instance[index], child, registry, base_path, path + (str(index),))
        if "items" in schema:
            start = len(prefix_items)
            for index, value in enumerate(instance[start:], start=start):
                validate_instance(value, schema["items"], registry, base_path, path + (str(index),))
        if "contains" in schema:
            matches = sum(
                is_valid(value, schema["contains"], registry, base_path) for value in instance
            )
            if matches < schema.get("minContains", 1):
                raise InstanceValidationError(f"{where}: too few contains matches")
            if "maxContains" in schema and matches > schema["maxContains"]:
                raise InstanceValidationError(f"{where}: too many contains matches")

    if isinstance(instance, dict):
        if len(instance) < schema.get("minProperties", 0):
            raise InstanceValidationError(f"{where}: object has too few properties")
        if "maxProperties" in schema and len(instance) > schema["maxProperties"]:
            raise InstanceValidationError(f"{where}: object has too many properties")
        for required_name in schema.get("required", []):
            if required_name not in instance:
                raise InstanceValidationError(
                    f"{where}: missing required property {required_name!r}"
                )
        properties = schema.get("properties", {})
        pattern_properties = schema.get("patternProperties", {})
        evaluated: set[str] = set()
        for name, child in properties.items():
            if name in instance:
                evaluated.add(name)
                validate_instance(instance[name], child, registry, base_path, path + (name,))
        for pattern, child in pattern_properties.items():
            for name, value in instance.items():
                if re.search(pattern, name):
                    evaluated.add(name)
                    validate_instance(value, child, registry, base_path, path + (name,))
        additional = schema.get("additionalProperties", True)
        for name, value in instance.items():
            if name in evaluated:
                continue
            if additional is False:
                raise InstanceValidationError(f"{where}: unexpected property {name!r}")
            if isinstance(additional, (dict, bool)):
                validate_instance(value, additional, registry, base_path, path + (name,))
        for trigger, dependencies in schema.get("dependentRequired", {}).items():
            if trigger in instance:
                for dependency in dependencies:
                    if dependency not in instance:
                        raise InstanceValidationError(
                            f"{where}: {trigger!r} requires property {dependency!r}"
                        )


def fixture_schema_name(path: Path) -> str:
    """Map ``config.valid.json`` to ``config.schema.json``."""

    return f"{path.name.split('.', 1)[0]}.schema.json"


def mutated_invalid_instance(path: Path) -> Any:
    """Build an invalid instance from a valid fixture and one explicit mutation."""

    specification = load_json(path)
    if not isinstance(specification, dict) or set(specification) != {"fixture", "mutation"}:
        raise SchemaCheckError(f"{path}: invalid case must contain exactly fixture and mutation")
    fixture_name = specification["fixture"]
    mutation = specification["mutation"]
    if not isinstance(fixture_name, str) or Path(fixture_name).name != fixture_name:
        raise SchemaCheckError(f"{path}: fixture must be a simple file name")
    if not isinstance(mutation, dict) or set(mutation) - {"op", "path", "value"}:
        raise SchemaCheckError(f"{path}: malformed mutation")
    operation = mutation.get("op")
    pointer = mutation.get("path")
    if operation not in {"add", "remove", "replace"} or not isinstance(pointer, str):
        raise SchemaCheckError(f"{path}: unsupported mutation")
    if operation in {"add", "replace"} and "value" not in mutation:
        raise SchemaCheckError(f"{path}: {operation} mutation requires value")
    if not pointer.startswith("/"):
        raise SchemaCheckError(f"{path}: mutation path must be an RFC 6901 pointer")

    source_path = TESTS / "valid" / fixture_name
    instance = copy.deepcopy(load_json(source_path))
    tokens = [token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/")]
    parent = instance
    for token in tokens[:-1]:
        if isinstance(parent, list) and token.isdigit():
            parent = parent[int(token)]
        elif isinstance(parent, dict) and token in parent:
            parent = parent[token]
        else:
            raise SchemaCheckError(f"{path}: mutation path does not exist")
    final = tokens[-1]
    if isinstance(parent, list) and final.isdigit():
        index = int(final)
        if operation == "add":
            parent.insert(index, mutation["value"])
        elif operation == "replace":
            parent[index] = mutation["value"]
        else:
            del parent[index]
    elif isinstance(parent, dict):
        if operation in {"remove", "replace"} and final not in parent:
            raise SchemaCheckError(f"{path}: mutation target does not exist")
        if operation == "remove":
            del parent[final]
        else:
            parent[final] = mutation["value"]
    else:
        raise SchemaCheckError(f"{path}: mutation parent is not a container")
    return instance


def require_invariant(condition: bool, where: str, message: str) -> None:
    """Raise an instance error when a documented cross-field invariant fails."""

    if not condition:
        raise InstanceValidationError(f"{where}: {message}")


def check_run_identity(run: dict[str, Any], where: str) -> None:
    prior = run["prior_run_uuid"]
    require_invariant(
        Path(run["directory"]).name == run["id"],
        where,
        "run directory basename must equal the human-readable run ID",
    )
    if run["collision_policy"] == "reject-existing":
        require_invariant(prior is None, where, "reject-existing requires no prior run UUID")
    else:
        require_invariant(prior is not None, where, "resume/overwrite requires a prior run UUID")
        require_invariant(prior != run["uuid"], where, "prior and current run UUIDs must differ")


def check_gate(gate: dict[str, Any], where: str) -> None:
    raw = gate["raw_counts"]
    require_invariant(
        raw["success_count"] + raw["failure_count"] == raw["sample_count"],
        where,
        "success and failure counts must sum to sample count",
    )
    require_invariant(
        gate["sample_count"] == raw["sample_count"],
        where,
        "gate and raw sample counts differ",
    )
    confidence = gate["confidence"]
    if confidence is not None:
        require_invariant(
            confidence["lower"] <= confidence["upper"],
            where,
            "confidence lower bound exceeds upper bound",
        )
    comparisons = {
        "gte": gate["observed"] >= gate["threshold"],
        "lte": gate["observed"] <= gate["threshold"],
        "gt": gate["observed"] > gate["threshold"],
        "lt": gate["observed"] < gate["threshold"],
        "eq": gate["observed"] == gate["threshold"],
    }
    require_invariant(
        gate["passed"] == comparisons[gate["comparator"]],
        where,
        "passed flag disagrees with comparator, observed value, and threshold",
    )


def check_checkpoint_counts(checkpoint: dict[str, Any], counts: dict[str, Any], where: str) -> None:
    require_invariant(
        checkpoint["global_step"] == counts["environment_steps"],
        where,
        "checkpoint step differs from environment-step count",
    )
    require_invariant(
        checkpoint["episode_count"] == counts["episodes"],
        where,
        "checkpoint episode count differs from run count",
    )
    require_invariant(
        checkpoint["update_count"] == counts["optimizer_updates"] + counts["q_table_updates"],
        where,
        "checkpoint update count differs from optimizer plus Q-table updates",
    )


def check_checkpoint_within_run(
    checkpoint: dict[str, Any], counts: dict[str, Any], where: str
) -> None:
    """Check that a selected checkpoint was created no later than run completion."""

    require_invariant(
        checkpoint["global_step"] <= counts["environment_steps"],
        where,
        "checkpoint step exceeds the completed run step count",
    )
    require_invariant(
        checkpoint["episode_count"] <= counts["episodes"],
        where,
        "checkpoint episode count exceeds the completed run count",
    )
    require_invariant(
        checkpoint["update_count"] <= counts["optimizer_updates"] + counts["q_table_updates"],
        where,
        "checkpoint update count exceeds the completed run count",
    )


def check_evaluation_snapshot(snapshot: dict[str, Any], split: str, where: str) -> None:
    require_invariant(snapshot["record"]["suite"] == split, where, f"expected {split} split")
    check_gate(snapshot["gate"], f"{where}.gate")


def check_semantics(instance: dict[str, Any], schema_name: str) -> None:
    """Enforce documented relationships that plain JSON Schema cannot compare."""

    if "run" in instance:
        check_run_identity(instance["run"], "$.run")
    if "environment" in instance:
        registry_version = instance["environment"]["registry_id"].rsplit("-", 1)[-1]
        require_invariant(
            registry_version == instance["environment"]["version"],
            "$.environment",
            "registry ID and explicit environment version disagree",
        )

    if schema_name == "config.schema.json":
        require_invariant(
            instance["created_at"] == instance["run"]["created_at"],
            "$",
            "config and run creation timestamps disagree",
        )
        if instance["qualification_mode"] == "qualifying":
            constraints = instance["training_constraints"]
            require_invariant(
                constraints["official_semantics"],
                "$.training_constraints",
                "official semantics required",
            )
            for field in ("reward_shaping", "heuristic_fallback", "behavior_cloning"):
                require_invariant(
                    not constraints[field], "$.training_constraints", f"{field} must be disabled"
                )
            require_invariant(
                constraints["warmup_imitation_updates"] == 0,
                "$.training_constraints",
                "warmup imitation must be disabled",
            )

    elif schema_name == "seeds.schema.json":
        partitions = instance["partitions"]
        validation = partitions["validation"]
        test = partitions["test"]
        separation = instance["validation_test_separation"]
        require_invariant(validation["id"] != test["id"], "$.partitions", "suite IDs must differ")
        require_invariant(
            validation["sha256"] != test["sha256"],
            "$.partitions",
            "validation and test hashes must differ",
        )
        require_invariant(
            set(validation["seeds"]).isdisjoint(test["seeds"]),
            "$.partitions",
            "validation and test seed values overlap",
        )
        require_invariant(
            separation["validation_sha256"] == validation["sha256"]
            and separation["test_sha256"] == test["sha256"],
            "$.validation_test_separation",
            "separation hashes do not match seed suites",
        )
        environment_ids = [
            item["environment_id"] for item in partitions["environment_resets"]["per_environment"]
        ]
        require_invariant(
            len(environment_ids) == len(set(environment_ids)),
            "$.partitions.environment_resets",
            "environment IDs must be unique",
        )

    elif schema_name == "metrics-row.schema.json":
        checkpoint = instance["checkpoint"]
        if checkpoint is not None:
            check_checkpoint_counts(checkpoint, instance["counts"], "$.checkpoint")
            if instance["policy_sha256"] is not None:
                require_invariant(
                    checkpoint["sha256"] == instance["policy_sha256"],
                    "$.checkpoint",
                    "checkpoint and active policy hashes differ",
                )

    elif schema_name == "eval-row.schema.json":
        check_checkpoint_counts(instance["checkpoint"], instance["counts"], "$.checkpoint")
        statistics = instance["statistics"]
        gate = instance["gate"]
        require_invariant(
            statistics["sample_count"] == gate["sample_count"],
            "$.statistics",
            "statistics and gate sample counts differ",
        )
        require_invariant(
            statistics["minimum_return"]
            <= statistics["mean_return"]
            <= statistics["maximum_return"],
            "$.statistics",
            "mean return is outside minimum/maximum range",
        )
        check_gate(gate, "$.gate")
        if instance["suite"]["split"] == "test":
            process = instance["process"]
            require_invariant(
                process["kind"] == "fresh-process"
                and process["exit_code"] == 0
                and process["checkpoint_loaded_from_disk"],
                "$.process",
                "test evaluation must reload successfully in a fresh process",
            )

    elif schema_name == "provenance.schema.json":
        source = instance["source"]
        require_invariant(
            (source["dirty"] and source["dirty_diff_sha256"] is not None)
            or (not source["dirty"] and source["dirty_diff_sha256"] is None),
            "$.source",
            "dirty flag and diff hash disagree",
        )
        started = datetime.fromisoformat(instance["timing"]["started_at"].replace("Z", "+00:00"))
        finished = datetime.fromisoformat(instance["timing"]["finished_at"].replace("Z", "+00:00"))
        require_invariant(finished >= started, "$.timing", "finish precedes start")
        check_checkpoint_within_run(
            instance["best_checkpoint"], instance["counts"], "$.best_checkpoint"
        )
        replay = instance["reproducibility"]
        require_invariant(
            replay["same_seed_replay_checked"]
            == (replay["same_seed_checkpoint_sha256"] is not None),
            "$.reproducibility",
            "same-seed replay flag and checkpoint hash disagree",
        )

    elif schema_name == "summary.schema.json":
        policy = instance["policy_change"]
        if policy["changed"]:
            require_invariant(
                policy["step_zero_sha256"] != policy["best_sha256"],
                "$.policy_change",
                "changed policy must have different hashes",
            )
        require_invariant(
            policy["optimizer_updates"] + policy["q_table_updates"]
            == instance["best_checkpoint"]["update_count"],
            "$.policy_change",
            "policy and checkpoint update counts differ",
        )
        check_evaluation_snapshot(
            instance["initial_validation"], "validation", "$.initial_validation"
        )
        check_evaluation_snapshot(instance["best_validation"], "validation", "$.best_validation")
        check_evaluation_snapshot(instance["final_test"], "test", "$.final_test")
        best_hash = instance["best_checkpoint"]["sha256"]
        require_invariant(
            instance["best_validation"]["record"]["checkpoint_sha256"] == best_hash
            and instance["final_test"]["record"]["checkpoint_sha256"] == best_hash,
            "$",
            "best validation/test records do not reference the best checkpoint",
        )
        check_checkpoint_within_run(
            instance["best_checkpoint"], instance["counts"], "$.best_checkpoint"
        )

    elif schema_name == "proof.schema.json":
        qualification = instance["qualification"]
        best_hash = instance["best_checkpoint"]["sha256"]
        require_invariant(
            qualification["step_zero_policy_sha256"] != qualification["best_policy_sha256"],
            "$.qualification",
            "step-zero and best policy hashes must differ",
        )
        require_invariant(
            qualification["best_policy_sha256"] == best_hash,
            "$.qualification",
            "qualification best hash differs from checkpoint hash",
        )
        require_invariant(
            qualification["combined_update_count"]
            == instance["counts"]["optimizer_updates"] + instance["counts"]["q_table_updates"],
            "$.qualification",
            "qualification update count differs from completed run counts",
        )
        check_gate(instance["validation_selection"]["gate"], "$.validation_selection.gate")
        check_gate(instance["held_out_test"]["gate"], "$.held_out_test.gate")
        check_gate(instance["result"], "$.result")
        validation = instance["validation_selection"]
        held_out = instance["held_out_test"]
        require_invariant(
            validation["record"]["suite"] == "validation" and held_out["record"]["suite"] == "test",
            "$",
            "evaluation records use the wrong split",
        )
        separation = instance["validation_test_separation"]
        require_invariant(
            validation["suite"]["sha256"] == separation["validation_suite_sha256"]
            and held_out["suite"]["sha256"] == separation["test_suite_sha256"],
            "$.validation_test_separation",
            "suite hashes do not match separation evidence",
        )
        require_invariant(
            separation["validation_suite_sha256"] != separation["test_suite_sha256"],
            "$.validation_test_separation",
            "validation and test suites must differ",
        )
        linked_hashes = {
            validation["record"]["checkpoint_sha256"],
            held_out["record"]["checkpoint_sha256"],
            held_out["fresh_process"]["checkpoint_sha256"],
            best_hash,
        }
        require_invariant(len(linked_hashes) == 1, "$", "proof checkpoint hashes disagree")
        require_invariant(
            instance["status"] == ("passed" if instance["result"]["passed"] else "failed"),
            "$",
            "proof status disagrees with result",
        )
        require_invariant(
            instance["result"] == held_out["gate"],
            "$.result",
            "proof result must exactly equal the held-out test gate",
        )
        check_checkpoint_within_run(
            instance["best_checkpoint"], instance["counts"], "$.best_checkpoint"
        )
        cohort = instance["release_cohort"]
        if cohort is not None:
            passes = sum(member["passed"] for member in cohort["members"])
            require_invariant(
                passes == cohort["actual_pass_count"],
                "$.release_cohort",
                "actual pass count differs from member results",
            )
            check_gate(cohort["median_gate"], "$.release_cohort.median_gate")

    elif schema_name == "video-manifest.schema.json":
        require_invariant(
            instance["canvas"] == {"width": 1280, "height": 720},
            "$.canvas",
            "canvas must be 1280x720",
        )
        output = instance["output"]
        layout = instance["layout"]
        require_invariant(
            layout["timelapse_frames"] + layout["final_best_frames"] == output["frame_count"],
            "$.layout",
            "layout frames do not sum to output frame count",
        )
        rate = output["frame_rate"]["numerator"] / output["frame_rate"]["denominator"]
        require_invariant(
            math.isclose(output["frame_count"] / rate, output["duration_seconds"]),
            "$.output",
            "frame count, frame rate, and duration disagree",
        )
        segments = instance["segments"]
        expected_start = 0
        demo_hash = instance["demo_suite"]["sha256"]
        demo_cases = instance["demo_suite"]["case_ids"]
        demo_seeds = segments[0]["demo_seeds"]
        require_invariant(
            instance["demo_suite"]["sample_count"] == len(demo_cases) == len(demo_seeds),
            "$.demo_suite",
            "demo sample count, case IDs, and seed counts disagree",
        )
        chronological_steps: list[int] = []
        chronological_frames = 0
        final_best_frames = 0
        final_best_started = False
        final_best_hashes: set[str] = set()
        for index, segment in enumerate(segments):
            where = f"$.segments.{index}"
            require_invariant(segment["index"] == index, where, "segment index is not canonical")
            require_invariant(
                segment["frame_start"] == expected_start, where, "segments are not contiguous"
            )
            require_invariant(
                segment["frame_end_exclusive"] > segment["frame_start"], where, "empty segment"
            )
            expected_start = segment["frame_end_exclusive"]
            checkpoint = segment["checkpoint"]
            metrics_record = segment["metrics_record"]
            record = segment["evaluation_record"]
            label = segment["label"]
            require_invariant(
                checkpoint["global_step"]
                == metrics_record["global_step"]
                == record["global_step"]
                == label["global_step"],
                where,
                "checkpoint, metrics, evaluation, and label steps differ",
            )
            require_invariant(
                checkpoint["sha256"]
                == metrics_record["policy_sha256"]
                == record["checkpoint_sha256"]
                and checkpoint["sha256"].startswith(label["checkpoint_short_sha256"]),
                where,
                "checkpoint, metrics, evaluation, and label hashes differ",
            )
            require_invariant(
                segment["demo_suite_sha256"] == demo_hash
                and segment["demo_seeds"] == demo_seeds
                and segment["demo_case_ids"] == demo_cases,
                where,
                "segments do not share the fixed demo suite, seeds, and case IDs",
            )
            require_invariant(
                label["environment_registry_id"] == instance["environment"]["registry_id"]
                and label["algorithm_id"] == instance["algorithm"]["id"]
                and label["root_seed"] == instance["root_seed"]
                and label["episode_count"] == checkpoint["episode_count"]
                and label["update_count"] == checkpoint["update_count"],
                where,
                "segment label disagrees with manifest or checkpoint metadata",
            )
            require_invariant(
                segment["reloaded_best"] == (segment["phase"] == "final-best"),
                where,
                "reloaded-best flag disagrees with segment phase",
            )
            segment_frames = segment["frame_end_exclusive"] - segment["frame_start"]
            if segment["phase"] == "chronological-training":
                require_invariant(
                    not final_best_started,
                    where,
                    "chronological segment appears after final-best playback began",
                )
                chronological_steps.append(checkpoint["global_step"])
                chronological_frames += segment_frames
            else:
                final_best_started = True
                final_best_frames += segment_frames
                final_best_hashes.add(checkpoint["sha256"])
                require_invariant(
                    checkpoint["kind"] == "best" and segment["playback_speed"] == 1,
                    where,
                    "final-best segment must use a best checkpoint at real-time speed",
                )
        require_invariant(
            expected_start == output["frame_count"], "$.segments", "segments do not cover output"
        )
        require_invariant(
            chronological_steps
            and chronological_steps[0] == 0
            and chronological_steps == sorted(chronological_steps),
            "$.segments",
            "chronological checkpoint steps must begin at zero and be nondecreasing",
        )
        require_invariant(
            final_best_started and len(final_best_hashes) == 1,
            "$.segments",
            "manifest must contain one consistent final-best checkpoint",
        )
        require_invariant(
            chronological_frames == layout["timelapse_frames"]
            and final_best_frames == layout["final_best_frames"],
            "$.segments",
            "segment phases do not exactly fill the declared 20/10 layout",
        )
        require_invariant(
            instance["poster"]["source_frame"] < output["frame_count"],
            "$.poster",
            "poster frame is outside video",
        )
        required_inspection_frames = {0, output["frame_count"] // 2, output["frame_count"] - 1}
        require_invariant(
            required_inspection_frames.issubset(
                set(instance["verification"]["inspected_frame_indices"])
            ),
            "$.verification.inspected_frame_indices",
            "manual inspection must include first, middle, and final frames",
        )


def main() -> int:
    schema_paths = sorted(ROOT.glob("*.schema.json"))
    found_public = {path.name for path in schema_paths} - {"common.schema.json"}
    if found_public != PUBLIC_SCHEMAS:
        missing = sorted(PUBLIC_SCHEMAS - found_public)
        extra = sorted(found_public - PUBLIC_SCHEMAS)
        raise SchemaCheckError(f"public schema set mismatch; missing={missing}, extra={extra}")

    registry = SchemaRegistry(schema_paths)
    for schema_path in schema_paths:
        document = registry.documents[schema_path.resolve()]
        if document.get("$schema") != DRAFT_2020_12:
            raise SchemaCheckError(f"{schema_path}: expected Draft 2020-12 $schema")
        check_schema_node(document, schema_path.name, registry, schema_path.resolve())

    valid_paths = sorted((TESTS / "valid").glob("*.json"))
    invalid_paths = sorted((TESTS / "invalid").glob("*.json"))
    valid_coverage = {fixture_schema_name(path) for path in valid_paths}
    invalid_coverage = {fixture_schema_name(path) for path in invalid_paths}
    if valid_coverage != PUBLIC_SCHEMAS:
        raise SchemaCheckError(
            f"valid fixture coverage mismatch: {sorted(PUBLIC_SCHEMAS - valid_coverage)}"
        )
    if invalid_coverage != PUBLIC_SCHEMAS:
        raise SchemaCheckError(
            f"invalid fixture coverage mismatch: {sorted(PUBLIC_SCHEMAS - invalid_coverage)}"
        )

    for fixture_path in valid_paths:
        schema_path = (ROOT / fixture_schema_name(fixture_path)).resolve()
        instance = load_json(fixture_path)
        validate_instance(
            instance,
            registry.documents[schema_path],
            registry,
            schema_path,
            (),
        )
        check_semantics(instance, schema_path.name)

    for fixture_path in invalid_paths:
        schema_path = (ROOT / fixture_schema_name(fixture_path)).resolve()
        try:
            instance = mutated_invalid_instance(fixture_path)
            validate_instance(
                instance,
                registry.documents[schema_path],
                registry,
                schema_path,
                (),
            )
            check_semantics(instance, schema_path.name)
        except InstanceValidationError:
            continue
        raise SchemaCheckError(f"{fixture_path}: invalid fixture unexpectedly passed")

    print(
        f"validated {len(schema_paths)} schema documents, "
        f"{len(valid_paths)} valid fixtures, and {len(invalid_paths)} invalid fixtures"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SchemaCheckError, InstanceValidationError) as error:
        print(f"schema validation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
