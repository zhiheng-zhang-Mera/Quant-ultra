"""One-command Quant-Ultra launcher."""
from Main.main import parse_args, run_pipeline


if __name__ == "__main__":
    run_pipeline(parse_args())
