
# Chess Analysis Using AI Methodology
## SMAI Project – CS7.403b | Team 32


# Chess Feature Set Description

This project generates two feature sets for chess position analysis from PGN files, aimed at machine learning tasks to evaluate position strength. The feature sets are extracted using the `python-chess` library, NetworkX for graph-based features, and a chess engine (e.g., Stockfish) for position evaluation. The first feature set is solely graph-based, capturing network properties of chess piece interactions. The second is a mixed feature set, combining traditional chess evaluation metrics with graph-based network features. Both are saved as CSV files with features and a strength label (`strong`, `weak`, `neutral`) based on engine evaluation.

## Dataset Overview

### 1. Graph-Based Feature Set
This dataset contains 56 features derived from a position network combining support and mobility networks for a given player's color (White or Black). The network represents pieces as nodes, with edges indicating defensive relationships (support) or legal moves (mobility). Features are computed using the `compute_network_features` function, focusing on topological properties of the subgraph for the current player's color.

#### Features (56)
1. **min_degree**: Minimum degree (>0) in the undirected subgraph.
2. **max_degree**: Maximum degree in the undirected subgraph.
3. **avg_degree**: Average degree across nodes.
4. **deg_dist_1 to deg_dist_5**: Percentage of nodes in 5 degree bins.
5. **density**: Ratio of actual to possible edges.
6. **node_count**: Number of nodes in the subgraph.
7. **edge_count**: Number of edges in the undirected subgraph.
8. **num_wcc_large**: Number of weakly connected components (WCCs) with >1 node.
9. **num_scc_large**: Number of strongly connected components (SCCs) with >1 node.
10. **num_trees**: Number of WCCs that are trees with >1 node.
11. **pct_nodes_wcc_large**: Percentage of nodes in WCCs with >1 node.
12. **pct_nodes_scc_large**: Percentage of nodes in SCCs with >1 node.
13. **pct_nodes_trees**: Percentage of nodes in tree-like WCCs with >1 node.
14. **pct_nodes_wcc_size1**: Percentage of nodes in isolated WCCs (size 1).
15. **pct_smallest_wcc**: Percentage of nodes in the smallest WCC with >1 node.
16. **pct_largest_wcc**: Percentage of nodes in the largest WCC.
17. **avg_pct_wcc_large**: Average percentage of nodes in WCCs with >1 node.
18. **wcc_size_dist_1 to wcc_size_dist_3**: Top 3 WCC sizes (% of nodes), sorted descending.
19. **min_wcc_diameter**: Diameter of the smallest WCC with >1 node.
20. **max_wcc_diameter**: Diameter of the largest WCC with >1 node.
21. **avg_wcc_diameter**: Average diameter of WCCs with >1 node.
22. **wcc_diam_dist_1 to wcc_diam_dist_3**: Top 3 WCC diameters, sorted descending.
23. **pct_smallest_scc**: Percentage of nodes in the smallest SCC with >1 node.
24. **pct_largest_scc**: Percentage of nodes in the largest SCC.
25. **avg_pct_scc_large**: Average percentage of nodes in SCCs with >1 node.
26. **scc_size_dist_1 to scc_size_dist_3**: Top 3 SCC sizes (% of nodes), sorted descending.
27. **min_scc_diameter**: Diameter of the smallest SCC with >1 node.
28. **max_scc_diameter**: Diameter of the largest SCC with >1 node.
29. **avg_scc_diameter**: Average diameter of SCCs with >1 node.
30. **scc_diam_dist_1 to scc_diam_dist_3**: Top 3 SCC diameters, sorted descending.
31. **pct_trees_size1**: Percentage of nodes in trees of size 1 (isolated nodes).
32. **pct_smallest_tree**: Percentage of nodes in the smallest tree with >1 node.
33. **pct_largest_tree**: Percentage of nodes in the largest tree with >1 node.
34. **avg_pct_trees_large**: Average percentage of nodes in trees with >1 node.
35. **tree_size_dist_1 to tree_size_dist_3**: Top 3 tree sizes (% of nodes), sorted descending.
36. **min_tree_depth**: Minimum diameter (approximated as depth) of trees with >1 node.
37. **max_tree_depth**: Maximum diameter (approximated as depth) of trees with >1 node.
38. **avg_tree_depth**: Average diameter (approximated as depth) of trees with >1 node.
39. **tree_diam_dist_1 to tree_diam_dist_3**: Top 3 tree diameters, sorted descending.
40. **clustering_coeff**: Average clustering coefficient.
41. **largest_clique**: Size of the largest clique.
42. **min_edge_dominating_set**: Size of an approximated minimal edge dominating set.
43. **max_independent_set**: Size of an approximated maximum independent set.
44. **smallest_maximal_matching**: Size of a maximal matching.
45. **degree_assortativity**: Degree assortativity coefficient.
46. **transitivity**: Ratio of triangles to possible triangles.
47. **num_attracting_components**: Number of attracting components in the directed subgraph.
48. **nodes_3_core**: Number of nodes in the 3-core.
49. **nodes_2_core**: Number of nodes in the 2-core.
50. **rich_club_coeff**: Rich club coefficient for the highest degree.
51. **ratio_friendly_to_opposing**: Ratio of edges to opposing color vs. within color.
52. **ratio_opposing_to_friendly**: Ratio of edges from opposing to friendly vs. friendly to opposing.
53. **avg_friendly_to_opposing**: Average edges per node to opposing color.
54. **avg_opposing_to_friendly**: Average edges per node from opposing to friendly.
55. **pct_friendly_to_opposing**: Percentage of nodes with edges to opposing color.
56. **pct_opposing_in_degree_gt0**: Percentage of nodes with incoming edges from opposing color.

#### Label
- **y_label**: Position strength (`strong`, `weak`, `neutral`) based on engine evaluation (pawn units: >1.5 for strong, ≤0.26 for neutral, else weak).

### 2. Mixed Feature Set
This dataset contains 56 features combining traditional chess metrics (material, piece-square tables, mobility, king safety, pawn structure) with 10 graph-based features from support and mobility networks. It provides a balanced representation of chess-specific knowledge and network topology.

#### Features (56)
1. **white_pawns to white_kings (6)**: Count of each piece type for White.
2. **black_pawns to black_kings (6)**: Count of each piece type for Black.
3. **white_pawn_pst to white_king_pst (6)**: Sum of piece-square table (PST) values for each piece type for White.
4. **black_pawn_pst to black_king_pst (6)**: Sum of PST values for each piece type for Black (mirrored for board perspective).
5. **white_knight_mobility to white_king_mobility (5)**: Number of legal moves for non-pawn pieces for White.
6. **black_knight_mobility to black_king_mobility (5)**: Number of legal moves for non-pawn pieces for Black.
7. **white_king_center_dist**: King's distance from board center (sum of file and rank distances).
8. **white_king_pawn_shield**: Number of friendly pawns in front of White's king (within 1 file).
9. **black_king_center_dist**: King's distance from board center for Black.
10. **black_king_pawn_shield**: Number of friendly pawns in front of Black's king.
11. **white_doubled_pawns**: Number of doubled pawns for White (stacked on same file).
12. **white_isolated_pawns**: Number of isolated pawns for White (no pawns on adjacent files).
13. **white_passed_pawns**: Number of passed pawns for White (no enemy pawns blocking path to promotion).
14. **white_pawn_chains**: Number of pawn chains for White (pawns diagonally supporting each other).
15. **black_doubled_pawns to black_pawn_chains (4)**: Same pawn structure metrics for Black.
16. **support_avg_degree**: Average degree in the undirected support network.
17. **support_clustering**: Average clustering coefficient in the support network.
18. **support_nodes**: Number of nodes in the support network.
19. **support_edges**: Number of edges in the support network.
20. **support_centrality**: Average degree centrality in the support network.
21. **mobility_avg_degree**: Average degree in the undirected mobility network.
22. **mobility_clustering**: Average clustering coefficient in the mobility network.
23. **mobility_nodes**: Number of nodes in the mobility network.
24. **mobility_edges**: Number of edges in the mobility network.
25. **mobility_centrality**: Average degree centrality in the mobility network.

#### Label
- **y_label**: Position strength (`strong`, `weak`, `neutral`) based on engine evaluation (pawn units: >1.5 for strong, ≤0.26 for neutral, else weak).

## Implementation Details
- **Input**: PGN file (e.g., `filtered_25k_moves.pgn`) containing chess games.
- **Output**: CSV file (e.g., `chess_features.csv`) with features and labels for each move.
- **Networks**:
  - **Support Network**: Directed graph where nodes are pieces, edges represent friendly pieces defending each other (same color).
  - **Mobility Network**: Directed graph where nodes are squares, edges represent legal moves by pieces.
  - **Position Network** (for graph-based set): Combines support and mobility networks, with features computed for the current player's color.
- **Evaluation**: Uses a chess engine (e.g., Stockfish) to evaluate positions, converting scores to pawn units and assigning labels based on thresholds.
- **Libraries**: `python-chess` for board manipulation, `networkx` for graph analysis, `numpy` for numerical computations.
- **Error Handling**: Skips moves with errors, logs issues, and continues processing.

## Notes
- The graph-based feature set emphasizes structural relationships between pieces, suitable for capturing complex interactions.
- The mixed feature set incorporates chess-specific knowledge (e.g., material balance, pawn structure) alongside network properties, offering a more traditional chess evaluation perspective.
- Features are normalized where appropriate (e.g., percentages for node counts) to ensure consistency.
- The engine evaluation uses a 0.1-second time limit for efficiency, balancing accuracy and processing speed.
- The code assumes the PGN file is valid and the chess engine is properly configured.

This dual approach allows for flexible analysis, enabling comparisons between purely topological and hybrid chess evaluation methods for position strength prediction.