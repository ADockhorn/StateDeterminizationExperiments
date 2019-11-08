__author__ = 'Alex'

""" This file implements connected component subgraph extraction based on the pre-condition of
self-edge < max_core_distance
"""
def connected_components_core(G, max_core_distance):
    """Generate connected components.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph

    Returns
    -------
    comp : generator of lists
       A list of nodes for each component of G.

    See Also
    --------
    strongly_connected_components

    Notes
    -----
    For undirected graphs only.
    """
    seen={}
    for v in G:
        if v not in seen:
            c = single_source_shortest_path_length_core(G, v, max_core_distance)
            yield list(c)
            seen.update(c)


def connected_component_subgraphs_core(G, max_core_distance, copy=True):
    """Generate connected components as subgraphs.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    copy: bool (default=True)
      If True make a copy of the graph attributes

    Returns
    -------
    comp : generator
      A generator of graphs, one for each connected component of G.


    See Also
    --------
    connected_components

    Notes
    -----
    For undirected graphs only.
    Graph, node, and edge attributes are copied to the subgraphs by default.
    """
    for c in connected_components_core(G, max_core_distance):
        if copy:
            yield G.subgraph(c).copy()
        else:
            yield G.subgraph(c)


def single_source_shortest_path_length_core(G,source, max_core_distance, cutoff=None):
    """Compute the shortest path lengths from source to all reachable nodes.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    cutoff : integer, optional
        Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    lengths : dictionary
        Dictionary of shortest path lengths keyed by target.


    See Also
    --------
    shortest_path_length
    """
    seen={}                  # level (number of hops) when seen in BFS
    level=0                  # the current level
    nextlevel={source:1}  # dict of nodes to check at next level
    while nextlevel:
        thislevel=nextlevel  # advance to next level
        nextlevel={}         # and start a new list (fringe)
        for v in thislevel:
            if v not in seen:
                seen[v]=level # set the level of vertex v
                if v in G[v] and G[v][v]["weight"] <= max_core_distance:

                    nextlevel.update(G[v]) # add neighbors of v
        if (cutoff is not None and cutoff <= level):  break
        level=level+1
    return seen  # return all path lengths as dictionary