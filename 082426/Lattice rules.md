Lattice rules:

Rule 1: Every node is connected to 8 neighbors along the 111 direction at unit length, always.

Rule 2: Every node is 'virtually connected' to each of the 6 neighboring 100 directions, with valid distances between 1 and sqrt(2), inclusive. This enables activated shortcuts (distance of 1) and octahedrons (sqrt(2)). 2/sqrt(3) is the rest lattice config for these virtual connections.

Rule 3: No other connections are possible.

That's it. Nodes at 200 are of no concern; if you're interested in a node that is 200 with respect to a given node, move your focus node closer so it becomes a 100 node and use local rules.