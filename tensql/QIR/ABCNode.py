from __future__ import annotations

import abc
import pathlib
import json

from typing import Generator, TYPE_CHECKING, Tuple, List, Optional

if TYPE_CHECKING:
    from .ABCExpression import ABCExpression


class ABCNode(abc.ABC):
    def __init__(self, name: Optional[str]):
        if name is not None:
            self._name = name
        else:
            self._name = self.description
        self._alias = self._name

    @property
    def name(self) -> str:
        return self._name

    @property
    def alias(self) -> str:
        return self._alias

    def aliased(self, name: str) -> ABCNode:
        self._alias = name
        return self
      
    @abc.abstractmethod
    def __iter__(self) -> Generator[Tuple[str, ABCNode], None, None]:
        pass

    @property
    def description(self) -> str:
        return f"{type(self).__name__}"

    def describe_as_edges(self) -> Generator[Tuple[ABCNode, ABCNode, str], None, None]:
        # print(type(self))
        for edge_name, child in self:
            # print(type(parent), type(self))
            yield (self, child, edge_name)
            yield from child.describe_as_edges()

    @staticmethod
    def save_graph(path: str, *ir_nodes: ABCNode) -> None:
        path = pathlib.Path(path)
        with path.open("w") as fout:
            all_edges: List[Tuple[ABCNode, ABCNode]] = []
            for node in ir_nodes:
                all_edges.extend(node.describe_as_edges())

            print("digraph G {", file=fout)
            nodes = set()
            for tail, head, edge_name in all_edges:
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
