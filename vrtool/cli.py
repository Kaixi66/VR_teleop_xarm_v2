from __future__ import annotations

import argparse
import signal
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_CONFIG_PATH, ConfigError, load_config
from .doctor import run_doctor
from .manager import (
    ManagerError,
    prepare_quest,
    run_collector,
    stack_status,
    start_stack,
    stop_stack,
)


class _TerminationRequested(BaseException):
    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vrctl",
        description="Safe UF850 VR teleoperation and legacy-compatible data collection",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="TOML configuration path")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("doctor", help="check dependencies, devices, network, ports and disk")

    prepare = commands.add_parser(
        "prepare",
        help="prepare Quest networking without starting or moving the robot",
    )
    prepare.add_argument("--hotspot", action="store_true", help="enable the configured Wi-Fi hotspot first")

    start = commands.add_parser("start", help="start description, UF850 driver and VR bridge")
    start.add_argument("--hotspot", action="store_true", help="explicitly enable the configured Wi-Fi hotspot")
    start.add_argument("--wait-quest", action="store_true", help="wait for one valid Quest packet before returning")

    collect = commands.add_parser("collect", help="record one episode; Ctrl+C finalizes it")
    collect.add_argument("--dataset", required=True, help="dataset directory name below paths.data_root")
    collect.add_argument("--task", default="", help="optional task text stored in episode metadata")
    collect.add_argument("--mock", action="store_true", help="use synthetic robot and camera data")

    run = commands.add_parser("run", help="doctor, start, collect one episode, then stop")
    run.add_argument("--dataset", required=True)
    run.add_argument("--task", default="")
    run.add_argument("--hotspot", action="store_true")
    run.add_argument("--wait-quest", action="store_true", help="verify fresh Quest packets before recording")

    commands.add_parser("status", help="show managed process status")
    commands.add_parser("stop", help="safely stop bridge and ROS processes")

    validate = commands.add_parser("validate", help="read-only validation of a legacy or v2 dataset")
    validate.add_argument("target", help="dataset path, or a dataset name below paths.data_root")
    validate.add_argument("--strict-v2", action="store_true", help="require v2 timestamps and episode metadata")
    validate.add_argument("--fast", action="store_true", help="check JPEG structure without full decoding")
    return parser


def _dataset_path(config: dict, target: str) -> Path:
    candidate = Path(target).expanduser()
    if candidate.exists() or candidate.is_absolute() or "/" in target:
        return candidate.resolve()
    return (Path(config["paths"]["data_root"]) / target).resolve()


def _run_validator(config: dict, target: str, strict_v2: bool, fast: bool = False) -> int:
    recording = config["recording"]
    command = [
        str(config["paths"]["collector_python"]),
        "-B",
        "-m",
        "vrtool.validator",
        str(_dataset_path(config, target)),
        "--width",
        str(recording["width"]),
        "--height",
        str(recording["height"]),
        "--max-camera-skew-ms",
        str(recording["max_camera_skew_ms"]),
        "--max-frame-age-ms",
        str(recording["max_frame_age_ms"]),
    ]
    if strict_v2:
        command.append("--strict-v2")
    if fast:
        command.append("--fast")
    return subprocess.run(command, cwd=config["_meta"]["project_root"], check=False).returncode


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config: dict | None = None
    previous_handlers: dict[int, signal.Handlers] = {}

    def request_termination(signum: int, _frame: object) -> None:
        raise _TerminationRequested(signum)

    for signum in (signal.SIGTERM, signal.SIGHUP):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_termination)
    try:
        config = load_config(args.config)

        if args.command == "doctor":
            report = run_doctor(config)
            print(report.render())
            return 0 if report.ok else 1

        if args.command == "prepare":
            prepare_quest(config, hotspot=args.hotspot)
            return 0

        if args.command == "start":
            state = start_stack(
                config,
                hotspot=args.hotspot,
                wait_quest=args.wait_quest,
            )
            print(f"Stack started. Logs: {state['log_dir']}")
            return 0

        if args.command == "collect":
            code = run_collector(
                config,
                dataset=args.dataset,
                task=args.task,
                mock=args.mock,
            )
            if code != 0 and not args.mock:
                print(f"Collector failed with code {code}; stopping the control stack for safety.", file=sys.stderr)
                stop_stack(config)
            elif not args.mock:
                print(
                    "Episode saved. VR control is PAUSED; the hotspot and bridge remain running. "
                    "Run easy_collect again for the next episode, or './vrctl stop' when finished."
                )
            return code

        if args.command == "run":
            report = run_doctor(config)
            print(report.render())
            if not report.ok:
                print("Preflight failed; no process was started.", file=sys.stderr)
                return 1
            start_stack(
                config,
                hotspot=args.hotspot,
                wait_quest=args.wait_quest,
            )
            try:
                return run_collector(
                    config,
                    dataset=args.dataset,
                    task=args.task,
                )
            finally:
                stop_stack(config)

        if args.command == "status":
            statuses = stack_status(config)
            if not statuses:
                print("Stack is not running.")
                return 1
            failed = False
            for label, alive, pid, log_path in statuses:
                print(f"{label:12} {'RUNNING' if alive else 'STOPPED':8} pid={pid} log={log_path}")
                failed = failed or not alive
            return 1 if failed else 0

        if args.command == "stop":
            stop_stack(config)
            print("Stack stopped safely.")
            return 0

        if args.command == "validate":
            return _run_validator(config, args.target, args.strict_v2, args.fast)
    except _TerminationRequested as exc:
        if config is not None:
            try:
                stop_stack(config)
            except Exception as stop_exc:
                print(f"ERROR during managed cleanup: {stop_exc}", file=sys.stderr)
        print(f"Received signal {exc.signum}; managed stack stopped.", file=sys.stderr)
        return 128 + exc.signum
    except KeyboardInterrupt:
        if config is not None:
            try:
                stop_stack(config)
            except Exception:
                pass
        print("Interrupted; managed stack stopped.", file=sys.stderr)
        return 130
    except (ConfigError, ManagerError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        for signum, old_handler in previous_handlers.items():
            signal.signal(signum, old_handler)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
