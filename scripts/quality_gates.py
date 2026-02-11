import argparse
import os

def _apply_site_root():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--site-root", default=os.environ.get("SITE_ROOT", ""))
    args, _ = parser.parse_known_args()
    if args.site_root:
        os.chdir(args.site_root)

_apply_site_root()

# (Existing quality gate logic continues below)
