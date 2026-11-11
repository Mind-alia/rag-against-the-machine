"""Tests for the macOS Docker wrapper around the Linux moulinette."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_wrapper_runs_moulinette_in_linux_amd64_container(
    tmp_path: Path,
) -> None:
    """The wrapper mounts the project and forwards evaluator arguments."""
    project_root = Path(__file__).resolve().parents[1]
    wrapper = project_root / "moulinette-docker.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = info ]; then exit 0; fi\n"
        "printf '%s\\n' \"$@\" > \"$DOCKER_ARGS_FILE\"\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    moulinette = tmp_path / "moulinette-ubuntu"
    moulinette.write_text("binary", encoding="utf-8")
    arguments_file = tmp_path / "docker-arguments.txt"
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "MOULINETTE_BINARY": str(moulinette),
            "DOCKER_ARGS_FILE": str(arguments_file),
        }
    )

    result = subprocess.run(
        [
            str(wrapper),
            "evaluate_student_search_results",
            "results.json",
            "dataset.json",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0
    arguments = arguments_file.read_text(encoding="utf-8").splitlines()
    assert arguments[0] == "run"
    assert "linux/amd64" in arguments
    assert "ubuntu:22.04" in arguments
    assert "/moulinette" in arguments
    assert arguments[-3:] == [
        "evaluate_student_search_results",
        "results.json",
        "dataset.json",
    ]
