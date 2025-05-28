# Convert this into executbale python code with pylog by defining all facts and rules using assertz

# ```prolog
# % dfs_better(Scenario, Why, Example).
# dfs_better(deep_solution,
#            'DFS can reach deep solutions faster, as it explores one branch fully before backtracking.',
#            'Finding a deep configuration in a puzzle game').

# dfs_better(memory_constraints,
#            'DFS uses less memory on wide graphs, since it only tracks the current path.',
#            'Very wide trees or mazes').

# dfs_better(any_solution,
#            'DFS may quickly find some solution, without guaranteeing the shortest path.',
#            'Maze generation; any valid maze is acceptable').

# dfs_better(topological_sorting,
#            'DFS naturally enables post-order traversal required for topological sorting.',
#            'Build dependency resolution in compilers').

# dfs_better(cycle_detection,
#            'DFS can efficiently check for cycles by tracking the current path.',
#            'Deadlock detection in operating systems').

# dfs_better(strongly_connected_components,
#            'DFS is the basis of Kosaraju\'s and Tarjan\'s algorithms for SCCs.',
#            'Analyzing SCCs in social networks').


# % bfs_better(Scenario, Why, Example).
# bfs_better(shortest_path,
#            'BFS finds the shortest path in unweighted graphs.',
#            'Finding the shortest route in a maze').

# bfs_better(level_order_traversal,
#            'BFS explores layer by layer, so all nodes at one depth are explored before going deeper.',
#            'Level-order traversal of trees').

# bfs_better(shallow_solutions,
#            'BFS is more efficient when all solutions are close to the root.',
#            'Locating nearest exit in an emergency route plan').


# % summary_table(Scenario, DFSbetter, BFSbetter)
# summary_table(memory_usage_wide_graphs,    yes, no).
# summary_table(deep_solution_search,        yes, no).
# summary_table(shortest_path_unweighted,    no, yes).
# summary_table(topological_sorting,         yes, no).
# summary_table(any_solution,                yes, no).
# summary_table(cycle_scc_detection,         yes, no).


# % Helper rule: when is DFS preferred?
# dfs_preferred(Scenario, Why, Example) :-
#     dfs_better(Scenario, Why, Example).

# % Helper rule: when is BFS preferred?
# bfs_preferred(Scenario, Why, Example) :-
#     bfs_better(Scenario, Why, Example).

# % Query example:
# % ?- dfs_preferred(Scenario, Why, Example).
# % ?- bfs_preferred(Scenario, Why, Example).
# % ?- summary_table(Scenario, DFS, BFS).
# ```
# ---

# **How to use these facts and rules:**

# - To find all scenarios where DFS is better, run:
#   ```prolog
#   ?- dfs_preferred(Scenario, Why, Example).
#   ```
# - To see when BFS is preferred:
#   ```prolog
#   ?- bfs_preferred(Scenario, Why, Example).
#   ```
# - To see a summary table:
#   ```prolog
#   ?- summary_table(Scenario, DFSbetter, BFSbetter).
#   ```

from pyswip import Prolog

prolog = Prolog()

# Assert facts
prolog.assertz(
    "dfs_better(deep_solution, 'DFS can reach deep solutions faster', 'Puzzle game')"
)
prolog.assertz(
    "dfs_better(memory_constraints, 'DFS uses less memory on wide graphs, since it only tracks the current path.','Very wide trees or mazes')"
)
prolog.assertz(
    "dfs_better(any_solution, 'DFS may quickly find some solution, without guaranteeing the shortest path.', 'Maze generation; any valid maze is acceptable')"
)

prolog.assertz(
    "dfs_better(topological_sorting, 'DFS naturally enables post-order traversal required for topological sorting, Build dependency resolution in compilers')",
)


prolog.assertz(
    "dfs_better(cycle_detection, 'DFS can efficiently check for cycles by tracking the current path.', 'Deadlock detection in operating systems')"
)


prolog.assertz(
    "dfs_better(strongly_connected_components, 'DFS is the basis of Kosarajus and Tarjans algorithms for SCCs.',  'Analyzing SCCs in social networks')"
)

prolog.assertz("bfs_better(shortest_path, 'BFS finds shortest path', 'Maze solving')")


prolog.assertz(
    "bfs_better(level_order_traversal, 'BFS explores layer by layer, so all nodes at one depth are explored before going deeper.',            'Level-order traversal of trees')"
)

prolog.assertz(
    "bfs_better(shallow_solutions, 'BFS is more efficient when all solutions are close to the root.',            'Locating nearest exit in an emergency route plan')"
)

prolog.assertz("summary_table(memory_usage_wide_graphs,    yes, no)")
prolog.assertz("summary_table(deep_solution_search,        yes, no)")
prolog.assertz("summary_table(shortest_path_unweighted,    no, yes)")
prolog.assertz("summary_table(topological_sorting,         yes, no)")
prolog.assertz("summary_table(any_solution,                yes, no)")
prolog.assertz("summary_table(cycle_scc_detection,         yes, no)")


prolog.assertz("dfs_preferred(Scenario, Why, Example) :- dfs_better(Scenario, Why, Example)")
prolog.assertz("bfs_preferred(Scenario, Why, Example) :- bfs_better(Scenario, Why, Example)")



# Query
for result in prolog.query("dfs_better(Scenario, Why, Example)"):
    print(result)


for result in prolog.query("bfs_better(Scenario, Why, Example)"):
    print(result)
