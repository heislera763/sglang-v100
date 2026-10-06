"""Reject unrecorded upstream-core changes on the maintained main branch."""

import argparse
import json
import subprocess


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="main")
    args = parser.parse_args()
    provenance = json.loads(git("show", args.ref + ":v100_lite/provenance.json"))
    manifest = json.loads(git("show", args.ref + ":v100_lite/core-patches.json"))
    base = provenance["current_upstream"]
    subprocess.run(["git", "merge-base", "--is-ancestor", base, args.ref], check=True)
    changed = set(
        git("diff", "--name-only", base, args.ref, "--", "python/sglang").splitlines()
    )
    unexpected = changed - set(manifest["upstream_paths"])
    if unexpected:
        raise SystemExit("Unrecorded core patches:\n" + "\n".join(sorted(unexpected)))
    print(f"{args.ref}: {len(changed)} recorded core patches against {base[:12]}")
    for path in sorted(changed):
        print(path + ": " + manifest["upstream_paths"][path])


if __name__ == "__main__":
    main()
