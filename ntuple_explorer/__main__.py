"""
Launch the app:

    python -m ntuple_explorer --data DIR [--presets DIR ...] [--port 8501]

--data     folder whose .root files are listed in the app; relative paths in presets are resolved against it
           (default: $NTUPLE_EXPLORER_DATA, else the current directory)
--presets  folder(s) with preset files (*.py defining PRESET = dict(...)); may be given several times
           (default: $NTUPLE_EXPLORER_PRESETS, os.pathsep-separated)
--cache    where the column cache is written (default: <data>/.ntuple_cache)

The server listens on localhost only: the app compiles C++ typed into its text boxes (ROOT expressions), so it must
not be reachable by other people. Use --address only on a machine and network you trust.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ntuple_explorer", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("NTUPLE_EXPLORER_DATA", "."),
                    help="folder with the ROOT files (default: $NTUPLE_EXPLORER_DATA, else the current directory)")
    ap.add_argument("--presets", action="append", default=[], help="folder with preset files (repeatable; default: $NTUPLE_EXPLORER_PRESETS)")
    ap.add_argument("--cache", help="cache folder (default: <data>/.ntuple_cache)")
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--address", default="localhost", help="address to listen on (default: localhost only)")
    a = ap.parse_args(argv)

    env = os.environ.copy()
    env["NTUPLE_EXPLORER_DATA"] = str(Path(a.data).expanduser().resolve())
    if a.presets:
        env["NTUPLE_EXPLORER_PRESETS"] = os.pathsep.join(str(Path(p).expanduser().resolve()) for p in a.presets)
    if a.cache:
        env["NTUPLE_EXPLORER_CACHE"] = str(Path(a.cache).expanduser().resolve())
    app = Path(__file__).resolve().with_name("app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app),
           "--server.address", a.address, "--server.port", str(a.port), "--server.headless", "true",
           "--server.maxUploadSize", "5", "--browser.gatherUsageStats", "false", "--theme.base", "light"]
    print(f"Ntuple Explorer on http://{a.address}:{a.port}  (data: {env['NTUPLE_EXPLORER_DATA']})")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
