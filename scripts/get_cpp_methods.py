# Requires `python3 -m pip install cxxheaderparser pcpp`
from enum import Enum
from pathlib import Path
from typing import Callable

import cxxheaderparser.parser
import cxxheaderparser.preprocessor
import cxxheaderparser.visitor

scripts_folder = Path(__file__).parent


class FunctionParam:
    def __init__(self, name: str, proto: str) -> None:
        self.proto = proto
        self.name = name


class ReturnType(Enum):
    VOID = 0
    OTHER = 1


class ConnectionMethod:
    def __init__(self, name: str, params: list[FunctionParam], return_type: ReturnType) -> None:
        self.name = name
        self.params = params
        self.return_type = return_type


class Visitor:
    def __init__(self, class_name: str) -> None:
        self.methods_dict = {}
        self.class_name = class_name

    def __getattr__(self, name) -> Callable[[...], bool]:
        return lambda *state: True

    def on_class_start(self, state):
        name = state.class_decl.typename.segments[0].format()
        return name == self.class_name

    def on_class_method(self, state, node):
        name = node.name.format()
        return_type = ReturnType.VOID
        if node.return_type and node.return_type.format() == "void":
            return_type = ReturnType.OTHER
        params = [
            FunctionParam(
                x.name,
                x.type.format() + " " + x.name + (" = " + x.default.format() if x.default else ""),
            )
            for x in node.parameters
        ]

        self.methods_dict[name] = ConnectionMethod(name, params, return_type)


def get_methods(class_name: str) -> dict[str, ConnectionMethod]:
    CLASSES = {
        "DuckDBPyConnection": Path(scripts_folder)
        / ".."
        / "src"
        / "duckdb_py"
        / "include"
        / "duckdb_python"
        / "pyconnection"
        / "pyconnection.hpp",
        "DuckDBPyRelation": Path(scripts_folder)
        / ".."
        / "src"
        / "duckdb_py"
        / "include"
        / "duckdb_python"
        / "pyrelation.hpp",
    }

    path = CLASSES[class_name]

    visitor = Visitor(class_name)
    preprocessor = cxxheaderparser.preprocessor.make_pcpp_preprocessor(retain_all_content=True)
    tu = cxxheaderparser.parser.CxxParser(
        path,
        None,
        visitor,
        options=cxxheaderparser.parser.ParserOptions(
            preprocessor=preprocessor,
        ),
    )
    tu.parse()

    return visitor.methods_dict


if __name__ == "__main__":
    print("This module should not called directly, please use `make generate-files` instead")
    exit(1)
