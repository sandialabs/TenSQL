from __future__ import annotations

import abc
import pathlib
import json
from typing import Optional, Generator, Tuple, TYPE_CHECKING, Dict, Any, List

if TYPE_CHECKING:
    pass


class Node(abc.ABC):
    def __init__(self, name: Optional[str]) -> None:
        if name is not None:
            self._name = name
        else:
            self._name = self.description

    @property
    def name(self) -> str:
        return self._name

    @property
    @abc.abstractmethod
    def dsl(self) -> str:
        pass

    @property
    def description(self) -> str:
        return type(self).__name__

    @abc.abstractmethod
    def __iter__(self) -> Generator[Tuple[str, Node], None, None]:
        pass

    @abc.abstractmethod
    def get_attributes(self) -> Dict[str, Any]:
        pass

    def describe_as_edges(self) -> Generator[Tuple[Node, Node], None, None]:
        # print(type(self))
        for relationship, parent in self:
            # print(type(parent), type(self))
            yield (parent, self, relationship)
            yield from parent.describe_as_edges()

    @staticmethod
    def save_graph(path: str, *ir_nodes: Node) -> None:
        path = pathlib.Path(path)
        with path.open("w") as fout:
            all_edges: List[Tuple[Node, Node]] = []
            for node in ir_nodes:
                all_edges.extend(node.describe_as_edges())

            print("digraph G {", file=fout)
            nodes = set()
            for head, tail, edge_name in all_edges:
                nodes.add((head.description, id(head)))
                nodes.add((tail.description, id(tail)))
                print(
                    f"    n{id(head)} -> n{id(tail)} [label={json.dumps(edge_name)}]",
                    file=fout,
                )

            print("", file=fout)
            for node_type, node_id in nodes:
                print(f"    n{node_id} [label={json.dumps(node_type)}]", file=fout)

            print("}", file=fout)
