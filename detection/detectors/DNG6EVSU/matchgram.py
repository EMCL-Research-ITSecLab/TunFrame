class Edge:

    def __init__(self, end: 'Node', weight: str):
        self.end = end
        self.weights = [weight]



class Node:

    def __init__(self):
        self.edges = []

    def is_leaf(self) -> bool:
        return len(self.edges) == 0

    def get_main_edge(self) -> 'Edge':
        for edge in self.edges:
            if not edge.end.is_leaf():
                return edge


class MatchGram:

    def __init__(self, n: int):
        self.root = Node()
        self.nodes = [self.root]
        self.n = n

    def add_domain(self, domain: str):
        current_node = self.root
        for i in range(0, len(domain) - self.n + 1):
            #look for non-leaf edge
            main_edge = current_node.get_main_edge()
            if i + self.n > len(domain) - 1:
                for edge in current_node.edges:
                    if edge.end.is_leaf() and domain[i : i + self.n] in edge.weights:
                        return
                    else:
                        new_node = Node()
                        new_edge = Edge(new_node, domain[i : i+self.n])
                        current_node.edges.append(new_edge)
                        self.nodes.append(new_node)
                        return
            # create new node
            if not main_edge:
                new_node = Node()
                new_edge = Edge(new_node, domain[i : i+self.n])
                current_node.edges.append(new_edge)
                self.nodes.append(new_node)
                current_node = new_node
                continue
            # search through graph
            found = False
            for weight in main_edge.weights:
                if weight == domain[i : i+self.n]:
                    found = True
                    break

            if not found:
                # Character Not Found
                main_edge.weights.append(domain[i : i+self.n])

            current_node = main_edge.end


    def check_domain(self, domain: str) -> bool:
        current_node = self.root
        for i in range(0, len(domain) - self.n + 1):
            if i + self.n > len(domain) - 1:
                for edge in current_node.edges:
                    if edge.end.is_leaf() and domain[i : i + self.n] in edge.weights:
                        return True
            main_edge = current_node.get_main_edge()
            if not main_edge:
                return False
            found = False
            for weight in main_edge.weights:
                if weight == domain[i : i+self.n]:
                   found = True
                   break
            if not found:
                return False
            current_node = main_edge.end
        return False


    def print_tree(self) -> str:
        """Print the MatchGram in a format resembling the paper's diagrams."""
        # Assign each node a unique ID based on creation order
        node_to_id = {node: idx for idx, node in enumerate(self.nodes)}

        # Collect all edges with their metadata
        edges = []
        for node in self.nodes:
            node_id = node_to_id[node]
            for edge in node.edges:
                end_id = node_to_id[edge.end]
                # Format weights as a sorted, comma-separated set
                weights_sorted = sorted(edge.weights)
                weights_str = "{" + ", ".join(f"'{w}'" for w in weights_sorted) + "}"
                edges.append((node_id, end_id, weights_str))

        # Sort edges by start node, then end node for readability
        edges.sort(key=lambda x: (x[0], x[1]))

        # Build the output string
        lines = ["MatchGram Tree:"]
        for start, end, weights in edges:
            lines.append(f"  Edge {start}→{end}: weights={weights}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.print_tree()

'''
test = MatchGram(2)
test.add_domain("yahoo.com")
print(test)
test.add_domain("cnn.com")
print(test)
test.add_domain("bcc.com")
print(test)
test.add_domain("mk.in")
print(test)
test.add_domain("m.tv")
print(test)

print(test.check_domain("yahoo.com"))
print(test.check_domain("yahic.com"))
'''