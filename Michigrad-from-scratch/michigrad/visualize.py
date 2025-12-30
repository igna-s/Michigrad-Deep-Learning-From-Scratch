import networkx as nx
import pyvis
from pyvis.network import Network
from graphviz import Digraph

from .engine import Value

def show_graph_interactive(self, filename="graph.html"):
    """
    Generates an interactive visualization of the computational graph using PyVis.
    
    Constructs a directed graph where nodes represent Value objects and operations.
    The result is saved as an HTML file.
    """
    nodes = {} # Dictionary to track visited nodes by their unique ID to avoid duplicates
    edges = []

    def build(v):
        if id(v) not in nodes: # Use object memory address (id) as unique key
            nodes[id(v)] = {"label": f"{v.name} | data={v.data:.2f} | grad={v.grad:.2f}", "shape": "box"}
            if v._op: # If the node is the result of an operation, create a separate operation node
                op_node_id = f"op_{id(v)}" # Unique ID for the operation node
                nodes[op_node_id] = {"label": v._op, "shape": "circle", "color": "lightblue", "size": 20}
                for child in v._prev:
                    edges.append((id(child), op_node_id)) # Edge from input (child) to operation
                edges.append((op_node_id, id(v))) # Edge from operation to output (result)
                for child in v._prev:
                    build(child) # Recursively build the graph for children

    build(self)

    graph = nx.DiGraph()
    for node_id, node_data in nodes.items():
        graph.add_node(node_id, **node_data) #Agrega los nodos al grafo con sus atributos
    for edge in edges:
        graph.add_edge(edge[0], edge[1], arrows={'to': True, 'from': False})

    net = Network(notebook=True, cdn_resources='in_line', directed=True)

    net.from_nx(graph)
    net.prep_notebook()
    net.show(filename)

def trace(root):
    """
    Traverses the computational graph starting from the root node.
    
    Returns:
        nodes: A set of all Value nodes in the graph.
        edges: A set of all edges (parent -> child) in the graph.
    """
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def show_graph(root, format='svg', rankdir='LR'):
    """
    Generates a static visualization of the computational graph using Graphviz.
    
    Args:
        root: The root Value node (usually the loss).
        format: Output format (png, svg, etc.).
        rankdir: Layout direction ('LR' for left-to-right, 'TB' for top-to-bottom).
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes:
        dot.node(name=str(id(n)), label = "{%s | data %.4f | grad %.4f }" % (n.name, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot

