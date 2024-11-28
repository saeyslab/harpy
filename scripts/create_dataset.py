"""Used to create a large dataset config folder for the harpy mutlirun analysis, see docs/hpc.md."""

import argparse
import logging
from pathlib import Path

parser = argparse.ArgumentParser(description="Create dataset configs.")
parser.add_argument(
    "--dataset_folder",
    "-f",
    metavar="D",
    type=Path,
    help="Location of the dataset folder.",
)
parser.add_argument(
    "--default_dataset",
    "-d",
    metavar="D",
    type=str,
    help="Name of the default dataset config. e.g. resolve_melanoma",
)
parser.add_argument(
    "--config_folder",
    "-c",
    metavar="C",
    type=Path,
    default="configs/dataset",
    help="Location of the config folder.",
)
parser.add_argument("--suffixes", "-s", metavar="N", type=str, nargs="+", help="List of all suffixes")
args = parser.parse_args()

if not args.default_dataset:
    args.default_dataset = args.dataset_folder.stem

assert args.dataset_folder.exists()

args.config_folder.mkdir(exist_ok=True, parents=True)
logging.debug(f"Config folder at {args.config_folder}")

TEMPLATE = r"""# @package dataset

defaults:
- {default_dataset}

image: ${{dataset.data_dir}}/{tiff}
coords: ${{dataset.data_dir}}/{txt}
"""

# for every tiff in the dataset folder
for tiff in Path(args.dataset_folder).glob("*_DAPI.tiff"):
    tiff = Path(tiff)
    # assume the omics data with the same name as a text file
    txt = Path(tiff.stem.replace("_DAPI", "_results") + ".txt")
    # output to a config file with the same name
    file = args.config_folder / f"{tiff.stem}.yaml"
    logging.debug(tiff.name)
    logging.debug(txt.name)
    logging.debug(file)
    with file.open("w") as f:
        print(
            TEMPLATE.format(default_dataset=args.default_dataset, tiff=tiff.name, txt=txt.name),
            file=f,
        )
