"""Keep the Azure ML environments in step with pyproject.toml.

The training image installs from environments/train/requirements.txt and the
endpoints from environments/endpoint/conda.yml. Neither can read
pyproject.toml, so these tests check that shared packages use the same
version as the one the tests run against.
"""

import re
import tomllib
from pathlib import Path

import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
TRAIN_REQUIREMENTS = PROJECT_ROOT / "environments" / "train" / "requirements.txt"
ENDPOINT_CONDA = PROJECT_ROOT / "environments" / "endpoint" / "conda.yml"
TRAIN_DOCKERFILE = PROJECT_ROOT / "environments" / "train" / "Dockerfile"
ENDPOINT_ENVIRONMENT = PROJECT_ROOT / "azureml" / "conda.environment.template.yaml"

# Packages the endpoint image may pin differently from pyproject.toml, with
# the reason. Anything else shared between the two must match exactly.
ENDPOINT_ALLOWED_DIFFERENCES: dict[str, str] = {}


def _project_requirements() -> dict[str, str]:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    lines = list(project["dependencies"])
    for extra in project["optional-dependencies"].values():
        lines.extend(extra)

    pins: dict[str, str] = {}
    for line in lines:
        requirement = Requirement(line)
        name = canonicalize_name(requirement.name)
        if name == "walmart-ml":  # self-referencing extras such as walmart-ml[all]
            continue
        pins[name] = str(requirement.specifier)
    return pins


def _train_requirements() -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw_line in TRAIN_REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        requirement = Requirement(line)
        pins[canonicalize_name(requirement.name)] = str(requirement.specifier)
    return pins


def test_pyproject_lists_each_package_once_with_one_version():
    # Purpose: a package repeated across extras (e.g. xgboost in train and
    # scoring) must use the same version everywhere.
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    lines = list(project["dependencies"])
    for extra in project["optional-dependencies"].values():
        lines.extend(extra)

    seen: dict[str, str] = {}
    for line in lines:
        requirement = Requirement(line)
        name = canonicalize_name(requirement.name)
        if name == "walmart-ml":
            continue
        specifier = str(requirement.specifier)
        assert seen.setdefault(name, specifier) == specifier, (
            f"{name} has different versions in pyproject.toml: "
            f"{seen[name]!r} and {specifier!r}"
        )


def test_training_image_matches_pyproject_pins():
    # Purpose: packages shared by the training image and pyproject.toml
    # must be pinned identically.
    project = _project_requirements()
    train = _train_requirements()

    shared = sorted(set(project) & set(train))
    assert shared, "Expected shared packages between the two files."

    mismatches = {
        name: {"pyproject.toml": project[name], "train/requirements.txt": train[name]}
        for name in shared
        if project[name] != train[name]
    }
    assert not mismatches, f"Version drift: {mismatches}"


def _endpoint_requirements() -> dict[str, str]:
    conda = yaml.safe_load(ENDPOINT_CONDA.read_text(encoding="utf-8"))
    pip_section = next(
        item["pip"] for item in conda["dependencies"] if isinstance(item, dict) and "pip" in item
    )
    pins: dict[str, str] = {}
    for line in pip_section:
        requirement = Requirement(line)
        pins[canonicalize_name(requirement.name)] = str(requirement.specifier)
    return pins


def test_endpoint_environment_pins_every_pip_package():
    # Purpose: an unpinned package lets a rebuild of the endpoint image pick
    # up a different version from the one that was tested.
    unpinned = sorted(
        name for name, specifier in _endpoint_requirements().items()
        if not specifier.startswith("==")
    )
    assert not unpinned, f"Pin these in environments/endpoint/conda.yml: {unpinned}"


def test_endpoint_environment_matches_pyproject_pins():
    # Purpose: the endpoints must run the versions the scoring tests ran
    # against. Any deliberate difference must be listed, with a reason, in
    # ENDPOINT_ALLOWED_DIFFERENCES.
    project = _project_requirements()
    endpoint = _endpoint_requirements()

    shared = sorted(set(project) & set(endpoint))
    assert {"xgboost", "scikit-learn", "pandas"} <= set(shared)

    mismatches = {
        name: {"pyproject.toml": project[name], "endpoint/conda.yml": endpoint[name]}
        for name in shared
        if project[name] != endpoint[name] and name not in ENDPOINT_ALLOWED_DIFFERENCES
    }
    assert not mismatches, f"Version drift: {mismatches}"


def test_endpoint_environment_uses_ci_python_version():
    # Purpose: CI tests and type-checks on the Python version set in
    # [tool.mypy]; the endpoints must run the same minor version.
    python_version = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["mypy"]["python_version"]
    conda = yaml.safe_load(ENDPOINT_CONDA.read_text(encoding="utf-8"))
    conda_python = next(
        item.split("=", 1)[1]
        for item in conda["dependencies"]
        if isinstance(item, str) and item.startswith("python=")
    )
    assert conda_python == python_version or conda_python.startswith(f"{python_version}."), (
        f"environments/endpoint/conda.yml uses Python {conda_python}, CI uses {python_version}"
    )


def _ci_python_version() -> str:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["mypy"]["python_version"]


def test_training_image_tag_is_pinned():
    # Purpose: ":latest" (or no tag) lets a rebuild pull a different base image.
    dockerfile = TRAIN_DOCKERFILE.read_text(encoding="utf-8")
    image = re.search(r"^FROM\s+(\S+)", dockerfile, re.M).group(1)
    assert _is_pinned_image(image), f"Pin the training base image to a dated tag: {image}"


def test_training_image_uses_ci_python_version():
    # Purpose: the Dockerfile records the base image's Python version (checked
    # with docker run ... python --version); it must match CI's minor version.
    dockerfile = TRAIN_DOCKERFILE.read_text(encoding="utf-8")
    match = re.search(r"Base image Python: (\d+\.\d+)", dockerfile)
    assert match, "Record the base image's Python version in environments/train/Dockerfile"
    assert match.group(1) == _ci_python_version(), (
        f"Training image uses Python {match.group(1)}, CI uses {_ci_python_version()}"
    )


def _is_pinned_image(image: str) -> bool:
    """Return True if an image reference has a digest or a tag other than latest."""
    if "@sha256:" in image:
        return True
    name = image.rsplit("/", 1)[-1]
    return ":" in name and name.rsplit(":", 1)[1] != "latest"


def test_endpoint_base_image_is_pinned():
    # Purpose: an untagged or ":latest" base image lets a rebuild of the
    # endpoint environment pull a different image.
    image = yaml.safe_load(ENDPOINT_ENVIRONMENT.read_text(encoding="utf-8"))["image"]
    assert _is_pinned_image(image), f"Pin the endpoint base image to a tag or digest: {image}"
