from pathlib import Path
from subprocess import Popen, PIPE
import json
import yaml
import sys


def load_yaml(path: Path) -> dict:
    try:
        with open(path, "r") as file:
            return yaml.safe_load(file)
    except (OSError, IOError) as file_error:
        sys.exit(f"YAML file '{path}' can not be opened: {file_error}")
    except yaml.YAMLError as yaml_error:
        sys.exit(f"YAML file '{path}' can not be parsed: {yaml_error}")


def formatAndWriteFile(path: Path, content: str) -> None:
    formatConfig = json.dumps(
        load_yaml(Path(__file__).parent.resolve() / "config/.clang-format")
    )
    with open(path, "w") as f:
        formatterProcess = Popen(
            ["clang-format", "--assume-filename=file.cpp", f"--style={formatConfig}"],
            stdin=PIPE,
            stdout=f,
            encoding="UTF-8",
        )
        formatterProcess.stdin.write(content)
        formatterProcess.stdin.close()
        formatterProcess.wait()
