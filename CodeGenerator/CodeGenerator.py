from jinja2 import Environment, PackageLoader, select_autoescape
from Formatter import formatAndWriteFile


env = Environment(
    loader=PackageLoader("CodeGenerator", "templates"),
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=select_autoescape(),
)

template = env.get_template("SparseLDL.cpp.jinja")
data = {"stage": 50}
rendered = template.render(data)

formatAndWriteFile("include/SparseLDL/CodeGen/SparseLDLGenerated.h", rendered)
