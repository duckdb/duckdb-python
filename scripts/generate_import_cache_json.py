import contextlib
import json
from pathlib import Path
from typing import Union

script_dir = Path(__file__).parent
lines = [line for line in (script_dir / "imports.py").read_text().splitlines() if line]


class ImportCacheAttribute:
    def __init__(self, full_path: str) -> None:
        parts = full_path.split(".")
        self.type = "attribute"
        self.name = parts[-1]
        self.full_path = full_path
        self.children: dict[str, ImportCacheAttribute] = {}

    def has_item(self, item_name: str) -> bool:
        return item_name in self.children

    def get_item(self, item_name: str) -> "ImportCacheAttribute":
        assert item_name in self.children
        return self.children[item_name]

    def add_item(self, item: "ImportCacheAttribute"):
        assert not self.has_item(item.full_path)
        self.children[item.full_path] = item

    def __repr__(self) -> str:
        return str(self.children)

    def populate_json(self, json_data: dict):
        json_data[self.full_path] = {
            "type": self.type,
            "full_path": self.full_path,
            "name": self.name,
            "children": list(self.children.keys()),
        }
        for item in self.children:
            self.children[item].populate_json(json_data)


class ImportCacheModule:
    def __init__(self, full_path) -> None:
        parts = full_path.split(".")
        self.type = "module"
        self.name = parts[-1]
        self.full_path = full_path
        self.items: dict[str, Union[ImportCacheAttribute, ImportCacheModule]] = {}

    def add_item(self, item: Union[ImportCacheAttribute, "ImportCacheModule"]):
        assert self.full_path != item.full_path
        assert not self.has_item(item.full_path)
        self.items[item.full_path] = item

    def get_item(self, item_name: str) -> Union[ImportCacheAttribute, "ImportCacheModule"]:
        assert self.has_item(item_name)
        return self.items[item_name]

    def populate_json(self, json_data: dict):
        json_data[self.full_path] = {
            "type": self.type,
            "full_path": self.full_path,
            "name": self.name,
            "children": list(self.items.keys()),
        }
        for key_name in self.items:
            self.items[key_name].populate_json(json_data)

    def has_item(self, item_name: str) -> bool:
        return item_name in self.items

    def __repr__(self) -> str:
        return str(self.items)

    def root_module(self) -> bool:
        return self.name == self.full_path


class ImportCacheGenerator:
    def __init__(self) -> None:
        self.modules: dict[str, ImportCacheModule] = {}

    def add_module(self, path: str):
        assert path.startswith("import")
        path = path[7:]
        module = ImportCacheModule(path)
        self.modules[module.full_path] = module

        # Add it to the parent module if present
        parts = path.split(".")
        if len(parts) == 1:
            return

        # This works back from the furthest child module to the top level module
        child_module = module
        for i in range(1, len(parts)):
            parent_path = ".".join(parts[: len(parts) - i])
            parent_module = self.add_or_get_module(parent_path)
            parent_module.add_item(child_module)
            child_module = parent_module

    def add_or_get_module(self, module_name: str) -> ImportCacheModule:
        if module_name not in self.modules:
            self.add_module(f"import {module_name}")
        return self.get_module(module_name)

    def get_module(self, module_name: str) -> ImportCacheModule:
        if module_name not in self.modules:
            msg = "Import the module before registering its attributes!"
            raise ValueError(msg)
        return self.modules[module_name]

    def get_item(self, item_name: str) -> Union[ImportCacheModule, ImportCacheAttribute]:
        parts = item_name.split(".")
        if len(parts) == 1:
            return self.get_module(item_name)

        parent = self.get_module(parts[0])
        for i in range(1, len(parts)):
            child_path = ".".join(parts[: i + 1])
            if parent.has_item(child_path):
                parent = parent.get_item(child_path)
            else:
                attribute = ImportCacheAttribute(child_path)
                parent.add_item(attribute)
                parent = attribute
        return parent

    def add_attribute(self, path: str):
        assert not path.startswith("import")
        parts = path.split(".")
        assert len(parts) >= 2
        self.get_item(path)

    def populate_json(self, json_data: dict):
        for module_name in self.modules:
            self.modules[module_name].populate_json(json_data)

    def to_json(self):
        json_data = {}
        self.populate_json(json_data)
        return json_data


generator = ImportCacheGenerator()

for line in lines:
    if line.startswith("#"):
        continue
    if line.startswith("import"):
        generator.add_module(line)
    else:
        generator.add_attribute(line)

# Load existing JSON data from a file if it exists
existing_json_data = {}
json_cache_path = Path(script_dir) / "cache_data.json"
with contextlib.suppress(FileNotFoundError):
    existing_json_data = json.loads(json_cache_path.read_text())


def update_json(existing: dict, new: dict) -> dict:
    # Iterate over keys in the new dictionary.
    for key in new:
        new_value = new[key]
        old_value = existing.get(key)

        # If both values are dictionaries, update recursively.
        if isinstance(new_value, dict) and isinstance(old_value, dict):
            updated = update_json(old_value, new_value)
            existing[key] = updated
        else:
            # Otherwise, overwrite the existing value.
            existing[key] = new_value
    return existing


# Merge the existing JSON data with the new data
json_data = generator.to_json()
json_data = update_json(existing_json_data, json_data)

# Save the merged JSON data back to the file
json_cache_path.write_text(json.dumps(json_data, indent=4))
