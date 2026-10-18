"""Set up the Styled Clone voice converters (Seed-VC and OpenVoice v2).

Their PyPI packages pin old numpy/protobuf versions that conflict with qwen-tts, so instead of
installing them this script:

1. installs the dependencies their model code actually needs (``uv sync --extra qwen --extra vc``),
2. checks out each repository at the tested commit into its vendor directory
   (``<data dir>/vendor/seed-vc`` and ``<data dir>/vendor/OpenVoice``, or ``POLY_TTS_SEED_VC_DIR`` /
   ``POLY_TTS_OPENVOICE_DIR``), from where poly-tts imports it,
3. checks that poly-tts can use them (``--load`` also downloads the weights and loads the models).

Usage (from the repository root):

    uv run python scripts/setup_vc.py                  # both converters
    uv run python scripts/setup_vc.py --only seed-vc   # just one
    uv run python scripts/setup_vc.py --load           # also download weights and load the models
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from poly_tts.config import Settings  # noqa: E402  (stdlib-only module, importable before the sync)

CONVERTERS = ("seed-vc", "openvoice")


def run(*cmd: str, cwd: Path | None = None, check: bool = True, quiet: bool = False) -> str:
    if not quiet:
        print("  $", Path(cmd[0]).stem, *cmd[1:])
    done = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if check and done.returncode != 0:
        sys.exit(f"error: `{' '.join(cmd)}` failed:\n{done.stderr.strip() or done.stdout.strip()}")
    return done.stdout.strip()


def sync() -> None:
    uv = shutil.which("uv")
    if not uv:
        sys.exit("error: uv not found on PATH (https://docs.astral.sh/uv/). Or rerun with --no-sync after "
                 "installing the 'qwen' and 'vc' extras yourself.")
    print("Installing the qwen and vc extras")
    run(uv, "sync", "--extra", "qwen", "--extra", "vc", cwd=ROOT)


def checkout(name: str, url: str, commit: str, dest: Path, marker: str) -> None:
    """Shallow-fetch exactly ``commit`` into ``dest`` (a new or existing checkout)."""
    print(f"{name}: {dest}")
    git = shutil.which("git")
    if not git:
        sys.exit("error: git not found on PATH.")
    if dest.exists() and not (dest / ".git").exists():
        if any(dest.iterdir()):
            sys.exit(f"error: {dest} exists but is not a git checkout; move it away or set its POLY_TTS_*_DIR.")
    if (dest / ".git").exists():
        if run(git, "rev-parse", "-q", "--verify", "HEAD", cwd=dest, check=False, quiet=True) == commit:
            print("  already at the tested commit")
            return
        if run(git, "status", "--porcelain", "--untracked-files=no", cwd=dest, quiet=True):
            sys.exit(f"error: {dest} has local changes; commit or discard them first.")
    else:
        dest.mkdir(parents=True, exist_ok=True)
        run(git, "init", "-q", cwd=dest)
        run(git, "remote", "add", "origin", url, cwd=dest)
    # Fetch only if the commit isn't already present; a full clone stays a full clone.
    if run(git, "cat-file", "-t", commit, cwd=dest, check=False, quiet=True) != "commit":
        shallow = run(git, "rev-parse", "--is-shallow-repository", cwd=dest, quiet=True) == "true"
        has_history = bool(run(git, "rev-parse", "-q", "--verify", "HEAD", cwd=dest, check=False, quiet=True))
        depth = ["--depth", "1"] if shallow or not has_history else []
        run(git, "fetch", "-q", *depth, "origin", commit, cwd=dest)
    run(git, "-c", "advice.detachedHead=false", "checkout", "-q", commit, cwd=dest)
    if not (dest / marker).is_file():
        sys.exit(f"error: {dest} does not look like a {name} checkout (missing {marker}).")
    print(f"  checked out {commit[:7]}")


def verify(names: list[str], load: bool) -> None:
    """Check availability (and optionally load) in a fresh interpreter, so the just-synced packages are seen."""
    code = (
        "import sys\n"
        "from poly_tts.config import Settings\n"
        "from poly_tts.vc import create_converters\n"
        "s = Settings(); s.backend = 'qwen'\n"
        "convs = create_converters(s); failed = False\n"
        f"for name in {names!r}:\n"
        "    ok, why = convs[name].available()\n"
        f"    if ok and {load!r}:\n"
        "        try:\n"
        "            convs[name].load()\n"
        "        except Exception as e:\n"
        "            ok, why = False, str(e)\n"
        "    print(f'  {name}: ' + ('ready' if ok else 'NOT ready: ' + why))\n"
        "    failed |= not ok\n"
        "sys.exit(1 if failed else 0)\n"
    )
    print("Checking" + (" (downloading weights and loading models)" if load else ""))
    if subprocess.run([sys.executable, "-c", code], cwd=ROOT).returncode != 0:
        sys.exit("error: setup finished but a converter is not usable (see above).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=CONVERTERS, help="Set up just this converter.")
    parser.add_argument("--no-sync", action="store_true", help="Don't run `uv sync --extra qwen --extra vc`.")
    parser.add_argument("--load", action="store_true", help="Also download the weights and load each model once.")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # keep our lines in order with the verify subprocess's output

    os.chdir(ROOT)  # the default data dir (./data) is relative to the repository root, as for `poly-tts serve`
    names = [args.only] if args.only else list(CONVERTERS)
    if not args.no_sync:
        sync()
    # Imported after the sync: poly_tts.vc needs numpy, which the sync guarantees.
    from poly_tts.vc import OpenVoiceConverter, SeedVCConverter

    settings = Settings()
    for cls, dest in ((SeedVCConverter, settings.seed_vc_dir), (OpenVoiceConverter, settings.openvoice_dir)):
        if cls.name in names:
            checkout(cls.name, cls.repo_url, cls.commit, dest.resolve(), cls.marker)
    verify(names, args.load)
    print("Done. Styled Clone is ready (web UI mode, `poly-tts styled-clone`, POST /api/tts/styled-clone).")


if __name__ == "__main__":
    main()
