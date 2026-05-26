import sys

import launcher


if __name__ == "__main__":
    raise SystemExit(launcher.main(["--docker", *sys.argv[1:]]))
