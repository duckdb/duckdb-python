import json
import os
from pathlib import Path

os.chdir(Path(__file__).parent)

JSON_PATH = "connection_methods.json"
WRAPPER_JSON_PATH = "connection_wrapper_methods.json"
DUCKDB_STUBS_FILE = Path("..") / "duckdb" / "__init__.pyi"

START_MARKER = "# START OF CONNECTION WRAPPER"
END_MARKER = "# END OF CONNECTION WRAPPER"


def generate():
    # Read the DUCKDB_STUBS_FILE file
    source_code = Path(DUCKDB_STUBS_FILE).read_text().splitlines()

    start_index = -1
    end_index = -1
    for i, line in enumerate(source_code):
        if line.startswith(START_MARKER):
            if start_index != -1:
                msg = "Encountered the START_MARKER a second time, quitting!"
                raise ValueError(msg)
            start_index = i
        elif line.startswith(END_MARKER):
            if end_index != -1:
                msg = "Encountered the END_MARKER a second time, quitting!"
                raise ValueError(msg)
            end_index = i

    if start_index == -1 or end_index == -1:
        msg = "Couldn't find start or end marker in source file"
        raise ValueError(msg)

    start_section = source_code[: start_index + 1]
    end_section = source_code[end_index:]
    # ---- Generate the definition code from the json ----

    methods = []

    # Read the JSON files
    connection_methods = json.loads(Path(JSON_PATH).read_text())
    wrapper_methods = json.loads(Path(WRAPPER_JSON_PATH).read_text())

    methods.extend(connection_methods)
    methods.extend(wrapper_methods)

    # On DuckDBPyConnection these are read_only_properties, they're basically functions without requiring () to invoke
    # that's not possible on 'duckdb' so it becomes a function call with no arguments (i.e duckdb.description())
    READONLY_PROPERTY_NAMES = ["description", "rowcount"]

    # These methods are not directly DuckDBPyConnection methods,
    # they first call 'from_df' and then call a method on the created DuckDBPyRelation
    SPECIAL_METHOD_NAMES = [x["name"] for x in wrapper_methods if x["name"] not in READONLY_PROPERTY_NAMES]

    def create_arguments(arguments) -> list:
        result = []
        for arg in arguments:
            argument = f"{arg['name']}: {arg['type']}"
            # Add the default argument if present
            if "default" in arg:
                default = arg["default"]
                argument += f" = {default}"
            result.append(argument)
        return result

    def create_definition(name, method) -> str:
        definition = f"def {name}("
        arguments = []
        if name in SPECIAL_METHOD_NAMES:
            arguments.append("df: pandas.DataFrame")
        if "args" in method:
            arguments.extend(create_arguments(method["args"]))
        if "kwargs" in method:
            if not any(x.startswith("*") for x in arguments):
                arguments.append("*")
            arguments.extend(create_arguments(method["kwargs"]))
        definition += ", ".join(arguments)
        definition += ")"
        definition += f" -> {method['return']}: ..."
        return definition

    def create_overloaded_definition(name, method) -> str:
        return f"@overload\n{create_definition(name, method)}"

    # We have "duplicate" methods, which are overloaded.
    # We keep note of them to add the @overload decorator.
    overloaded_methods: set[str] = {m for m in connection_methods if isinstance(m["name"], list)}

    body = []
    for method in methods:
        names = method["name"] if isinstance(method["name"], list) else [method["name"]]

        # Artificially add 'connection' keyword argument
        if "kwargs" not in method:
            method["kwargs"] = []
        method["kwargs"].append({"name": "connection", "type": "DuckDBPyConnection", "default": "..."})

        for name in names:
            if name in overloaded_methods:
                body.append(create_overloaded_definition(name, method))
            else:
                body.append(create_definition(name, method))

    # ---- End of generation code ----

    with_newlines = [x + "\n" for x in body]
    # Recreate the file content by concatenating all the pieces together

    new_content = start_section + with_newlines + end_section

    # Write out the modified DUCKDB_STUBS_FILE file
    Path(DUCKDB_STUBS_FILE).write_text("".join(new_content))


if __name__ == "__main__":
    msg = "Please use 'generate_connection_code.py' instead of running the individual script(s)"
    raise ValueError(msg)
    # generate()
