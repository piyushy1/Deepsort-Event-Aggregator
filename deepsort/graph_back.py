import networkx as nx
import pickle as pk
import numpy as np
import itertools
import matplotlib.pyplot as plt

with open("over_results.pkl", 'rb') as f:
	results = pk.load(f)

pkges = []

for res in results:
	G = nx.DiGraph()
	for i in range(len(res['identities'])):
		node = res['identities'][i]
		label = 'car'
		bbox = res['bbox_xyxy'][i]
		G.add_node(node, label=label, bbox=bbox)
	pkges.append(G)

gg = pkges[55]

def check_left(bbox1, bbox2):
	# centre_bbox1_x =  (bbox1[0] + bbox1[2])/2
	# centre_bbox2_x =  (bbox2[0] + bbox2[2])/2
	if bbox1[0] < bbox2[0] and bbox1[2] < bbox2[2]:
		return True
	else:
		return False

def check_right(bbox1, bbox2):
	# centre_bbox1_x =  (bbox1[0] + bbox1[2])/2
	# centre_bbox2_x =  (bbox2[0] + bbox2[2])/2
	if bbox1[0] < bbox2[0] and bbox1[2] < bbox2[2]:
		return False
	else:
		return True

import time

# s = time.time()

for graph in pkges:
	combos = list(itertools.combinations(list(graph), 2))
	for i,j in combos:
		temp = check_left(graph.nodes[i]['bbox'],graph.nodes[j]['bbox'])
		graph.add_edge(i,j, operators={'is_left' : temp, 'is_right': not temp})
		graph.add_edge(j,i, operators={'is_left' : not temp, 'is_right': temp})


# print(time.time() - s)

all_detected_objects = set()

for gs in pkges:
	all_detected_objects.update(list(gs.nodes()))

def compute_operator_event_adjancency_matrix(graph1, operator, all_detected_objects):

	nodes = list(graph1.nodes())
	no_nodes = len(nodes)
	no_all = len(all_detected_objects)
	adj_matrix = np.zeros((no_all+1, no_all+1))

	adj_matrix[0,:] = [-9, *all_detected_objects]
	adj_matrix[:,0] = [-9, *all_detected_objects]

	for i in range(1,no_all+1):
		for j in range(1,no_all+1):
			try:
				if i == j:
					adj_matrix[i][j] = 0
				else:
					adj_matrix[i][j] = graph1.edges[(all_detected_objects[i-1],all_detected_objects[j-1])]['operators'][operator]
			except KeyError as e:
				adj_matrix[i][j] = -9
			# print(adj_matrix[i][j], "  ")
		# print()
	return adj_matrix

gg = pkges[1]
hh = pkges[2]

l1 = compute_operator_event_adjancency_matrix(gg, 'is_left', list(all_detected_objects))
l2 = compute_operator_event_adjancency_matrix(hh, 'is_left', list(all_detected_objects))

common_nodes = list(set.intersection(set(gg.nodes), set(hh.nodes)))
common_nodes_index = list(map(lambda x : list(all_detected_objects).index(x)+1, common_nodes ))
	
swaps = {}

event_aggregated_graph = nx.DiGraph()
for node in all_detected_objects:
	event_aggregated_graph.add_node(node)

combos = list(itertools.combinations(list(event_aggregated_graph), 2))
for i,j in combos:
	event_aggregated_graph.add_edge(j,i, operators={'is_left' : [], 'is_right': [] })
	event_aggregated_graph.add_edge(i,j, operators={'is_left' : [], 'is_right': [] })


for operator in ['is_left', 'is_right']:
	for l in range(len(pkges)-1):
		l1 = compute_operator_event_adjancency_matrix(pkges[l], operator, list(all_detected_objects))
		l2 = compute_operator_event_adjancency_matrix(pkges[l+1], operator, list(all_detected_objects))
		lp = np.logical_not( np.logical_xor(l1,l2) )
		common_nodes = list(set.intersection(set(pkges[l].nodes), set(pkges[l+1].nodes)))
		common_nodes_index = list(map(lambda x : list(all_detected_objects).index(x)+1, common_nodes ))
		swap_indices = np.where(lp[common_nodes_index][:,common_nodes_index] == False)
		# print(swap_indices)
		swappees = list(zip(swap_indices[0], swap_indices[1]))
		# print(swappees)
		for (i,j) in swappees:
			event_aggregated_graph.edges[(common_nodes[i], common_nodes[j])]['operators'][operator].append(0)
			swaps[l] = (common_nodes[i], common_nodes[j])

		for edge in event_aggregated_graph.edges:
			if edge in swappees:
				continue
			event_aggregated_graph.edges[edge]['operators'][operator].append(1)


G = event_aggregated_graph

edgelist = []

for i in G.edges():
	for operator in G.edges[i]['operators']:
		# print(G.edges[i])
		if 0 in G.edges[i]['operators'][operator]:
			edgelist.append(i)

#####################################################################
def process_single_frame_traffic(data):
	G = nx.DiGraph()
	for i in range(len(data['identities'])):
		node = data['identities'][i]
		label = 'car'
		bbox = data['bbox_xyxy'][i]
		G.add_node(node, label=label, bbox=bbox)
	combos = list(itertools.permutations(list(G), 2))
	for i,j in combos:
		# print(G.nodes[j], "in temp")
		# print(G.nodes[j]['bbox'], "in temp")
		temp = (G.nodes[i]['bbox'] , G.nodes[j]['bbox'])
		results = { 'present': 1 }
		G.add_edge(i,j, operators=results)
		# print(G.edges.data())
	return G

def create_aggregated_graph_traffic(all_detected_objects):
	event_aggregated_graph = nx.DiGraph()
	for node in all_detected_objects:
		event_aggregated_graph.add_node(node)
	combos = list(itertools.permutations(all_detected_objects, 2))
	for i,j in combos:
		operators = { 'present':[] }
		event_aggregated_graph.add_edge(i,j, operators=operators)
	return event_aggregated_graph

def update_operations_value_traffic(frames):
	swaps = {}
	all_graphs = [ process_single_frame_traffic(frame) for frame in frames ]
	start = time.time()
	all_detected_objects = calculate_all_objects(all_graphs)
	event_aggregated_graph = create_aggregated_graph_traffic(all_detected_objects)
	print(time.time()- start ,"s Time taken for aggregated graph")
	process_nodes_egdes(all_graphs, event_aggregated_graph)
	process_in_window(frames, operators)
	start = time.time()

	all_combos = list(itertools.permutations(list(event_aggregated_graph), 2))

	for l in all_graphs:

		combos = list(l.edges())
		not_combos = [edge for edge in all_combos if edge not in combos]

		for (i,j) in combos:
			event_aggregated_graph.edges[(i, j)]['operators']['present'].append(1)

		for (i,j) in not_combos:
			event_aggregated_graph.edges[(i, j)]['operators']['present'].append(0)

	print(time.time()- start ,"s Time taken for updated operator values in edges of aggregated graph")
	return event_aggregated_graph, swaps

#####################  Traffic ###################################################
##################################################################################


def calculate_all_objects(event_graphs):
	all_detected_objects = set()
	for graph in event_graphs:
		all_detected_objects.update(list(graph.nodes()))
	return all_detected_objects

# event_aggregated_graph = update_operations_value_traffic(results)


non_edges = [edge for edge in G.edges() if edge not in edgelist]

plt.figure(figsize=(11,7))
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size=1000, node_shape='o', node_color='#92D050')
nx.draw_networkx_labels(G, pos, font_family='monospace', font_weight='bold')
nx.draw_networkx_edges(G, pos, style='dashdot',  edgelist=non_edges, edge_color='black', width=0.15, arrows=True)
nx.draw_networkx_edges(G, pos, style='solid', edgelist=edgelist, edge_color='#ED553B', width=2, arrows=True, arrowsize=25)
nx.draw_networkx_edge_labels(G, pos, font_size=13, edge_labels={ (8,33): G.edges[(8,33)]['operators']['is_left'][::-1][:5] })
plt.savefig('final2.svg', format='svg', dpi=1000)
# plt.show()
# plt.savefig("test.svg")


# for l in range(len(pkges)):
# 	i = pkges[l]
# 	try:
# 		l2 = compute_operator_event_adjancency_matrix(i, 2)

# 		i6 = np.where(l2[0] == 6)[0][0]
# 		i8 = np.where(l2[0] == 8)[0][0]

# 		sub2 = l2[[i6,i8]][:,[i6,i8]]
# 		if not np.logical_not( np.logical_xor(sub,sub2) ).all():
# 			print(sub2, "here")
# 			print(np.logical_not( np.logical_xor(sub,sub2) ))
# 			print("in here ", l)
# 	except Exception as e:
# 		continue