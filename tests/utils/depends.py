import re

import colorama
import networkx
import pytest

MARKER_NAME = "depends_on"
MARKER_KWARG_ID = "name"
MARKER_KWARG_DEPENDENCIES = "on"

REGEX_PARAMETERS = re.compile(r"\[.+\]$")


def clean_nodeid(nodeid):
    return nodeid.replace("::()::", "::").split("@dependency_group_")[0]


def get_names(item):
    names = set()

    # Node id
    nodeid = clean_nodeid(item.nodeid)
    names.add(nodeid)

    # Node id without parameter
    nodeid = REGEX_PARAMETERS.sub("", nodeid)
    names.add(nodeid)

    # Node id scopes
    while "::" in nodeid:
        nodeid = nodeid.rsplit("::", 1)[0]
        names.add(nodeid)

    # Custom name
    for marker in item.iter_markers():
        if marker.name == MARKER_NAME and MARKER_KWARG_ID in marker.kwargs:
            for name in as_list(marker.kwargs[MARKER_KWARG_ID]):
                names.add(name)

    return names


def as_list(lst):
    return [lst] if isinstance(lst, str) else lst


STEPS = ["setup", "call", "teardown"]
GOOD_OUTCOME = "passed"


class DependencyManager:
    """
    A simplified and improved version of pytest-depends. Main differences are the following:
    * Add compatibility with pytest-xdist: group connected components of the dependency graph together,
        and rename them with the `@dependency_group_{i}` suffix so they are run in the same worker, assuming
        group scheduling is used.
    * Improved parameterized dependencies so tests can depend on other tests with matching parametrization.
        Ex. a test `test_model` with parameter `model` can depend on `test_other[{model}]`,
            then `test_model[llama]` will depend on `test_other[llama]`, and so on.
    * Improved description of missing/failed dependencies.
    * Some option hard-coded for Fast-LLM.
    """

    def __init__(self, items: list[pytest.Function]):
        self._items = items
        self._name_to_nodeids: dict[str, list[str]] = {}
        self._nodeid_to_item: dict[str, pytest.Function] = {}
        self._results: dict[str, dict[str, str]] = {}
        self._dependencies: dict[str, set[str]] = {}
        self._unresolved: dict[str, set[str]] = {}

        for item in self._items:
            nodeid = clean_nodeid(item.nodeid)
            # Add the mapping from nodeid to the test item
            self._nodeid_to_item[nodeid] = item
            # Add the mappings from all names to the node id
            for name in get_names(item):
                if name not in self._name_to_nodeids:
                    self._name_to_nodeids[name] = []
                self._name_to_nodeids[name].append(nodeid)
            # Create the object that will contain the results of this test
            self._results[nodeid] = {}

        for item in self._items:
            # Process the dependencies of this test
            # This uses the mappings created in the previous loop, and can thus not be merged into that loop
            nodeid = clean_nodeid(item.nodeid)
            self._dependencies[nodeid], self._unresolved[nodeid] = self._resolve_dependencies(item)

        self._items = self._sort_dependencies()

    @property
    def items(self) -> list[pytest.Function]:
        return self._items

    def register_result(self, item: pytest.Function, result: pytest.TestReport):
        self._results[clean_nodeid(item.nodeid)][result.when] = result.outcome

    def handle_missing(self, item: pytest.Function):
        nodeid = clean_nodeid(item.nodeid)
        if missing := self._unresolved[nodeid]:
            pytest.fail(f'{item.nodeid} depends on {", ".join(missing)}, which was not found', False)

        if failed := [
            f"{dependency} ({", ".join(f"{key}: {value}" for key, value in self._results[dependency].items()) if self._results[dependency] else "missing"})"
            for dependency in self._dependencies[nodeid]
            if not all(self._results[dependency].get(step, None) == "passed" for step in ("setup", "call", "teardown"))
        ]:
            pytest.skip(f'{item.nodeid} depends on {", ".join(failed)}')

    def _resolve_dependencies(self, item: pytest.Function):
        dependencies = set()
        unresolved = set()

        if "skip" in item.keywords:
            return dependencies, unresolved

        nodeid = clean_nodeid(item.nodeid)

        for marker in item.iter_markers():
            if marker.name == MARKER_NAME:
                for dependency in as_list(marker.kwargs.get(MARKER_KWARG_DEPENDENCIES, [])):
                    if hasattr(item, "callspec"):
                        dependency = dependency.format(**item.callspec.params)

                    # If the name is not known, try to make it absolute (ie file::[class::]method)
                    if dependency not in self._name_to_nodeids:
                        absolute_dependency = self._get_absolute_nodeid(dependency, nodeid)
                        if absolute_dependency in self._name_to_nodeids:
                            dependency = absolute_dependency

                    # Add all items matching the name
                    if dependency in self._name_to_nodeids:
                        for nodeid in self._name_to_nodeids[dependency]:
                            dependencies.add(nodeid)
                    else:
                        unresolved.add(dependency)

        return dependencies, unresolved

    def _sort_dependencies(self):
        # Build a directed graph for sorting
        dag = networkx.DiGraph()

        for item in self.items:
            nodeid = clean_nodeid(item.nodeid)
            dag.add_node(nodeid)
            for dependency in self._dependencies[nodeid]:
                dag.add_edge(dependency, nodeid)

        for i, nodeids in enumerate(sorted(networkx.weakly_connected_components(dag), key=len, reverse=True)):
            if len(nodeids) > 1:
                for nodeid in nodeids:
                    self._nodeid_to_item[nodeid]._nodeid = (
                        f"{self._nodeid_to_item[nodeid]._nodeid}@dependency_group_{i}"
                    )

        return [self._nodeid_to_item[nodeid] for nodeid in networkx.topological_sort(dag)]

    @staticmethod
    def _get_absolute_nodeid(nodeid: str, scope: str):
        parts = nodeid.split("::")
        # Completely relative (test_name), so add the full current scope (either file::class or file)
        if len(parts) == 1:
            base_nodeid = scope.rsplit("::", 1)[0]
            nodeid = f"{base_nodeid}::{nodeid}"
        # Contains some scope already (Class::test_name), so only add the current file scope
        elif "." not in parts[0]:
            base_nodeid = scope.split("::", 1)[0]
            nodeid = f"{base_nodeid}::{nodeid}"
        return clean_nodeid(nodeid)

    def print_name_map(self, verbose: bool = False):
        """Print a human-readable version of the name -> test mapping."""
        print("Available dependency names:")
        for name, nodeids in sorted(self._name_to_nodeids.items(), key=lambda x: x[0]):
            if len(nodeids) == 1:
                if name == nodeids[0]:
                    # This is just the base name, only print this when verbose
                    if verbose:
                        print(f"  {name}")
                else:
                    # Name refers to a single node id, so use the short format
                    print(f"  {name} -> {nodeids[0]}")
            else:
                # Name refers to multiple node ids, so use the long format
                print(f"  {name} ->")
                for nodeid in sorted(nodeids):
                    print(f"    {nodeid}")

    def print_processed_dependencies(self, colors: bool = False):
        """Print a human-readable list of the processed dependencies."""
        missing = "MISSING"
        if colors:
            missing = f"{colorama.Fore.RED}{missing}{colorama.Fore.RESET}"
            colorama.init()
        try:
            print("Dependencies:")

            for nodeid in sorted(self._dependencies):
                descriptions = []
                for dependency in self._dependencies[nodeid]:
                    descriptions.append(dependency)
                for dependency in self._unresolved[nodeid]:
                    descriptions.append(f"{dependency} ({missing})")
                if descriptions:
                    print(f"  {nodeid} depends on")
                    for description in sorted(descriptions):
                        print(f"    {description}")
        finally:
            if colors:
                colorama.deinit()
