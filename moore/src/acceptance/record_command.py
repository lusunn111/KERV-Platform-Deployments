"""Record an exact ancillary command, timestamps, output and exit status."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from acceptance.common import dump


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--id", required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        p.error("A command is required after --")
    directory = args.campaign / "operations" / args.id
    directory.mkdir(parents=True, exist_ok=False)
    record = {"argv":command,"cwd":str(Path.cwd()),"start_timestamp":datetime.now(timezone.utc).isoformat()}
    dump(directory / "command.json", record)
    with (directory / "stdout_stderr.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    record.update({"exit_code":result.returncode,"end_timestamp":datetime.now(timezone.utc).isoformat()})
    dump(directory / "command.json", record)
    print(directory, "exit=", result.returncode, flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
