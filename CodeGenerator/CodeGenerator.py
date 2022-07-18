from jinja2 import Environment, PackageLoader, select_autoescape
from Formatter import formatAndWriteFile
import argparse
from pathlib import Path


def isValid(value):
    iValue = int(value)
    if not value.isdigit() or iValue < 1:
        raise argparse.ArgumentTypeError(
            "%s is an invalid positive int value that is larger than 1." % value
        )
    return iValue


parser = argparse.ArgumentParser(description="Generate code from a template")
parser.add_argument(
    "stage",
    type=isValid,
    help="The number of stages to generate",
)
args = parser.parse_args()

env = Environment(
    loader=PackageLoader("CodeGenerator", "templates"),
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=select_autoescape(),
)

template = env.get_template("SparseLDL.cpp.jinja")
data = {"stage": args.stage}
rendered = template.render(data)

formatAndWriteFile(
    Path(__file__).parent.parent.resolve()
    / "include/SparseLDL/CodeGen/SparseLDLGenerated.h",
    rendered,
)
