### src/project/cli.py
import argparse
import logging
import sys

from .main import test_sim



# Configure logging
logging.basicConfig(
    filename="simulation.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="jqm",
        description="Run quantum simulations using JAX",
    )
    # parser.add_argument(
    #     "-p", "--pdb", required=True, help="place holder"
    # )
    parser.add_argument(
        "-d",
        "--device",
        choices=["cpu", "gpu", "rocm"],
        default="cpu",
        help="Choose JAX execution device (cpu or gpu). Default: cpu",
    )

    return parser.parse_args()


def main() -> None:

    # grab cli information
    args = parse_args()

    test_sim()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Failed to run simulation:")
        sys.exit(1)
