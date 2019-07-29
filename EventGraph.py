import networkx as nx
import pickle as pk
import numpy as np
import itertools
import time
from config import OPERATOR
from Operators import operations


def save_file(filename, file):
	with open(filename, 'wb') as f:
		pk.dump(file,f)
# with open("over_results.pkl", 'rb') as f:
# 	results = pk.load(f)


# for res in results:
# 	G = nx.DiGraph()
# 	for i in range(len(res['identities'])):
# 		node = res['identities'][i]
# 		label = 'car'
# 		bbox = res['bbox_xyxy'][i]
# 		G.add_node(node, label=label, bbox=bbox)
# 	pkges.append(G)

pkges = []


# print(time.time() - s)

def calculate_all_objects(event_graphs):
	all_detected_objects = set()
	for graph in event_graphs:
		all_detected_objects.update(list(graph.nodes()))
	return all_detected_objects

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
	return adj_matrix


def process_single_frame(data, req_operations):
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
		results = { operator: operations[operator](*temp) 	}
		G.add_edge(i,j, operators=results)
		# print(G.edges.data())
	return G		
	
def create_aggregated_graph(all_detected_objects, operators):
	event_aggregated_graph = nx.DiGraph()
	for node in all_detected_objects:
		event_aggregated_graph.add_node(node)
	combos = list(itertools.permutations(all_detected_objects, 2))
	for i,j in combos:
		event_aggregated_graph.add_edge(i,j, operators={ op:[] for op in operators })
	return event_aggregated_graph

# comparison between VEKG and EAG-
# edges and nodes comaprison....
def process_nodes_egdes(all_graphs, event_aggregated_graph):
	no_of_nodes_all_graphs = sum([len(i.nodes()) for i in all_graphs])
	no_of_nodes_aggr_graph = len(event_aggregated_graph.nodes())
	print(no_of_nodes_all_graphs, " all nodes")
	print(no_of_nodes_aggr_graph, " aggr nodes")
	print((no_of_nodes_all_graphs - no_of_nodes_aggr_graph)/no_of_nodes_all_graphs, " percentage reduction in nodes")
	all_edges = sum([len(i.edges()) for i in all_graphs])
	aggr_edges = len(event_aggregated_graph.edges())
	print(all_edges, " all e")
	print(aggr_edges, " aggr e")
	print(( all_edges - aggr_edges)/all_edges, " percentage reduction in edges")



#####################  Traffic start###############################################
##############################################################################
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
	save_file(f'list_graphs_{int(time.time())}.pkl', all_graphs)
	start = time.time()
	all_detected_objects = calculate_all_objects(all_graphs)
	event_aggregated_graph = create_aggregated_graph_traffic(all_detected_objects)
	print(time.time()- start ,"s Time taken for aggregated graph")
	process_nodes_egdes(all_graphs, event_aggregated_graph)
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
	save_file(f'ea_graphs_{int(time.time())}.pkl', event_aggregated_graph)
	return event_aggregated_graph, swaps

#####################  Traffic end ###################################################
##################################################################################


def make_TAG(frames, operators):

	tags = []
	for operator in operators:
		swaps = {}
		all_graphs = [ process_single_frame(frame, operators) for frame in frames ]
		all_detected_objects = calculate_all_objects(all_graphs)
		# save_file('list_graphs.pkl', all_graphs)
		event_aggregated_graph = create_aggregated_graph(all_detected_objects, operators)
		for l in range(len(all_graphs)-1):
			l1 = compute_operator_event_adjancency_matrix( all_graphs[l], operator, list(all_detected_objects))
			l2 = compute_operator_event_adjancency_matrix( all_graphs[l+1], operator, list(all_detected_objects))
			lp = np.logical_not( np.logical_xor(l1,l2) )
			common_nodes = list(set.intersection(set(all_graphs[l].nodes), set(all_graphs[l+1].nodes)))
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
	tags.append(event_aggregated_graph)

	print(operators)
	total_time = 0
	count = 0
	for graph in tags:
		all_detected_objects = list(all_detected_objects)
		first_node, last_node = all_detected_objects[0],all_detected_objects[::-1][0]
		for operator in operators:
			s1 = time.time()
			pp = graph.edges[(first_node, last_node)]['operators'][operator]
			
			try:
				for j in pp:
					count += 1
			except KeyError:
				print("not here??")
			s2 = time.time()
			total_time += (s2-s1)
	print((total_time)*1000, "ms Time for two nodes in TAG   ", count)

	# print(time.time()- start ,"s Time taken for updated operator values in edges of aggregated graph")
	# save_file(f'ea_graphs_{int(time.time())}.pkl', event_aggregated_graph)
	

def process_in_window(frames, operators):
	all_graphs = [ process_single_frame(frame, operators) for frame in frames ]
	all_detected_objects = calculate_all_objects(all_graphs)
		
	all_detected_objects = list(all_detected_objects)
	first_node, last_node = all_detected_objects[0],all_detected_objects[::-1][0]

	bb =  []
	print(operators) 
	count = 0
	start = time.time()

	for i in all_graphs:
		for operator in operators:
			try:
				bb.append(i.edges[(first_node, last_node)])
			except KeyError:
				bb.append(0)
			count += 1
	end = time.time()

	print((end- start)*1000, "ms Time taken for worst case search query between 1st and last node   ", count)

	event_aggregated_graph = create_aggregated_graph(all_detected_objects, operators)
	start = time.time()

	count = 0

	adj_time = 0

	for l in range(len(frames) - 1):
		for operator in operators:
			l1 = compute_operator_event_adjancency_matrix( all_graphs[l], operator, list(all_detected_objects))
			l2 = compute_operator_event_adjancency_matrix( all_graphs[l+1], operator, list(all_detected_objects))
			t1 = time.time()
			lp = np.logical_not( np.logical_xor(l1,l2) )
			common_nodes = list(set.intersection(set(all_graphs[l].nodes), set(all_graphs[l+1].nodes)))
			common_nodes_index = list(map(lambda x : list(all_detected_objects).index(x)+1, common_nodes ))
			swap_indices = np.where(lp[common_nodes_index][:,common_nodes_index] == False)
			t2 = time.time()
			adj_time += t2 -t1
			# print(swap_indices," 	 in process_in_window")
			count += 1
	end = time.time()
	print(f'timew - {adj_time}')
	print(end- start, "s Time taken for all graphs   count-", count)

def update_operations_value(frames, operators=['front', 'front2', 'front3', 'front4']):
	swaps = {}
	all_graphs = [ process_single_frame(frame, operators) for frame in frames ]
	# save_file(f'list_graphs_{int(time.time())}.pkl', all_graphs)
	start = time.time()
	all_detected_objects = calculate_all_objects(all_graphs)
	event_aggregated_graph = create_aggregated_graph(all_detected_objects, operators)
	print(time.time()- start ,"s Time taken for aggregated graph")
	process_nodes_egdes(all_graphs, event_aggregated_graph)
	process_in_window(frames, operators)

	count = 0
	start = time.time()
	adj_time = 0

	# all_edges_list = {i: {edge : [] for edge in event_aggregated_graph.edges} for i in operators}
	all_edges_list = {}
	for i in operators:
		for edge in event_aggregated_graph.edges:
			all_edges_list[i] = {edge : []}

	all_edges_dict = {edge : [] for edge in event_aggregated_graph.edges}
	
	somelist = []
	for l in range(len(frames)-1):
		for operator in operators:
			l1 = compute_operator_event_adjancency_matrix( all_graphs[l], operator, list(all_detected_objects))
			l2 = compute_operator_event_adjancency_matrix( all_graphs[l+1], operator, list(all_detected_objects))
			lp = np.logical_not( np.logical_xor(l1,l2) )
			common_nodes = list(set.intersection(set(all_graphs[l].nodes), set(all_graphs[l+1].nodes)))
			common_nodes_index = list(map(lambda x : list(all_detected_objects).index(x)+1, common_nodes ))
			swap_indices = np.where(lp[common_nodes_index][:,common_nodes_index] == False)
			# # print(swap_indices)
			swappees = zip(swap_indices[0], swap_indices[1])
			t1 = time.time()
			# print(swappees)
			all_edges = list(event_aggregated_graph.edges)
			for (i,j) in swappees:
				# all_edges_list[operator][(common_nodes[i], common_nodes[j])].append(0)
				all_edges_dict[(common_nodes[i], common_nodes[j])].append(0)
				all_edges.remove((common_nodes[i], common_nodes[j]))
				# swaps[l] = (common_nodes[i], common_nodes[j])

			for edge in all_edges:
				# all_edges_list[operator][edge].append(1)
				all_edges_dict[edge].append(1)
				# somelist.append(1)
				# event_aggregated_graph.edges[edge]#['operators'][operator].append(1)
			t2 = time.time()
			# print(all_edges_dict)
			adj_time += t2 -t1
			count += 1
	print(f'timea - {adj_time}')
	print(time.time() - start ,"s Time taken for updated operator values in edges of aggregated graph tag   count-", count)

	all_detected_objects = list(all_detected_objects)
	first_node, last_node = all_detected_objects[0],all_detected_objects[::-1][0]

	print(operators)
	count = 0
	pp = event_aggregated_graph.edges[(first_node, last_node)]['operators']['front']
	s1 = time.time()
	# print(pp)
	# print(f'len of array - {len(pp)}')
	try:
		for operator in operators:
			# if 0 in all_edges_list[operator][(first_node, last_node)]:
			if 0 in all_edges_dict[(first_node, last_node)]:
				pass
			# for j in all_edges_dict[(first_node, last_node)]:
			count += 1
	except KeyError:
		print("not here??")
	s2 = time.time()

	print((s2- s1)*1000, "ms Time for two nodes in EAG   ", count)

	# save_file(f'ea_graphs_{int(time.time())}.pkl', event_aggregated_graph)
	# return event_aggregated_graph, swaps, all_edges_list
	return event_aggregated_graph, swaps, all_edges_list

def update_operations_value_single(frames, operators=['left']):
	swaps = {}
	all_graphs = [ process_single_frame(frame, operators) for frame in frames ]
	all_detected_objects = calculate_all_objects(all_graphs)
	event_aggregated_graph = create_aggregated_graph(all_detected_objects, operators)
	start = time.time()
	for operator in operators:
		for l in range(len(all_graphs)-1):
			l1 = compute_operator_event_adjancency_matrix( all_graphs[l], operator, list(all_detected_objects))
			l2 = compute_operator_event_adjancency_matrix( all_graphs[l+1], operator, list(all_detected_objects))
			lp = np.logical_not( np.logical_xor(l1,l2) )
			common_nodes = list(set.intersection(set(all_graphs[l].nodes), set(all_graphs[l+1].nodes)))
			common_nodes_index = list(map(lambda x : list(all_detected_objects).index(x)+1, common_nodes ))
			swap_indices = np.where(lp[common_nodes_index][:,common_nodes_index] == False)
			# # print(swap_indices)

			# # dontcare = np.where()

			# swappees = list(zip(swap_indices[0], swap_indices[1]))
			# # print(swappees)
			# for (i,j) in swappees:
			# 	event_aggregated_graph.edges[(common_nodes[i], common_nodes[j])]['operators'][operator].append(0)
			# 	swaps[l] = (common_nodes[i], common_nodes[j])

			# for edge in event_aggregated_graph.edges:
			# 	if edge in swappees:
			# 		continue
			# 	event_aggregated_graph.edges[edge]['operators'][operator].append(1)

	save_file(f'single_ea_graphs_{int(time.time())}.pkl', event_aggregated_graph)
	return event_aggregated_graph, swaps