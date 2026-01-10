import networkx as nx
import matplotlib.pyplot as plt

def view_graph(self):
    
    def build_graph(G,self):
        G.add_node(f"{self.data.shape} {self._op} {self.fl_back}")
        for p in self._parents:
            G.add_node(f"{p.data.shape} {p._op} {p.fl_back}")
            G.add_edge(f"{p.data.shape} {p._op} {p.fl_back}",f"{self.data.shape} {self._op} {self.fl_back}")
            G = build_graph(G,p)
        return G
    
    G = nx.DiGraph()
    G = build_graph(G,self)
    nx.draw(G, pos = nx.spring_layout(G),
        with_labels=True,
        arrows=True,
        arrowsize=20,
        arrowstyle='->')
    plt.show()