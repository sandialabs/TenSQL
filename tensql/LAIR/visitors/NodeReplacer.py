from ... import LAIR
from ..Visitor import Visitor


class NodeReplacer(Visitor):
    def __init__(self, old_node: LAIR.Node, new_node: LAIR.Node):
        self.old_node = old_node
        self.new_node = new_node

    def visit(self, node: LAIR.Node) -> LAIR.Node:
        old_node_attributes = node.get_attributes()
        new_node_attributes = {}
        for name, attr in old_node_attributes.items():
            if attr is self.old_node:
                new_node_attributes[name] = self.new_node
            elif isinstance(attr, LAIR.Node):
                new_node_attributes[name] = self.visit(attr)
            else:
                new_node_attributes[name] = attr

        return type(node)(**new_node_attributes)
