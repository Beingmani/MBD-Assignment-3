import argparse
import time
import sys
import networkx as nx
import operator
from multiprocessing import Pool
import json


def cal_no_of_lines(filename='web-Google.txt'):

    with open(filename) as f:
        for i,l in enumerate(f):
            pass 
    return i + 1

#------------------------------------------Progress Bar------------------------------------------

def update_progress(progress):
    barLength = 20 
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
#------------------------------------------------------------------------------------------------------------
def ReadFile(inputGraph,filename='web-Google.txt'):
    
    print("\nReading File for Graph creation - Start\n")
    try:
        file_size=  cal_no_of_lines()
        i = 0
        input_file = open(filename, "r")
        for line in input_file:
            line = line.rstrip()
            if line[0] == "#":
                continue
            line_arr = line.split("\t")
            Node_toGraph(inputGraph, int(line_arr[0]))
            Node_toGraph(inputGraph, int(line_arr[1]))
            Edge_toGraph(inputGraph,int(line_arr[0]),int(line_arr[1]))
            update_progress(i/file_size)
            i += 1
    except FileNotFoundError:
        print("Wrong file or file path")
    finally:
        input_file.close()
    print("\nReading File for Graph creation - End\n")

def input_graph_creation():
    print("\nInput Graph creation - Start\n")
    inputGraph = nx.DiGraph()
    print("\nInput Graph creation - End\n")
    return inputGraph
    

def Node_toGraph(inputGraph, ID_Node):
    if inputGraph.has_node(ID_Node):
        pass
    else:
       inputGraph.add_node(ID_Node)

def Edge_toGraph(inputGraph, FromId, ToId):
    if inputGraph.has_edge(FromId,ToId):
        pass
    else:
        inputGraph.add_edge(FromId, ToId)
#Pagerank algorithm
def pagerank_implementation(inputGraph, probability=0.85,
             max_iter=2000,start_vector=None,pers_vector=None,dangling=None):
    print("\nPagerank Implementation - Start\n")
    if len(inputGraph) == 0:
        print("\nInput Graph Empty - Exiting page rank\n")
        return {}
    
    sto_graph = nx.stochastic_graph(inputGraph, weight='weight')
    nodeCount = sto_graph.number_of_nodes()


    print("\nCreating first vector\n")
    start_vector=assign_default_if_none_provided(sto_graph,nodeCount,start_vector)
    
    print("Intializing vectors...")
    pers_vector=assign_default_if_none_provided(sto_graph,nodeCount,pers_vector)


    if dangling is None:
        dangling_weights = pers_vector
    else:
        sums = float(sum(dangling.values()))
        dangling_weights = dict((key, val / sums) for key, val in dangling.items())
    dangling_nodes = [n for n in sto_graph if sto_graph.out_degree(n, weight='weight') == 0.0]

    cur_itr = 0
    print("power iterating")
    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):  
        update_progress( cur_itr/max_iter *120)
        last_v = start_vector
        start_vector = dict.fromkeys(last_v.keys(), 0)
        danglesum = probability * sum(last_v[n] for n in dangling_nodes)
        for n in start_vector:
            # this matrix multiply looks odd because it is
            # doing a left multiply x^T=last_v^T*sto_graph
            for nbr in sto_graph[n]:
                start_vector[nbr] =start_vector[nbr] +probability * last_v[n] * sto_graph[n][nbr]['weight']
            start_vector[n] =start_vector[n]+ danglesum * dangling_weights.get(n, 0) + (1.0 - probability) * pers_vector.get(n, 0)
        # check convergence, l1 norm
        err = sum([abs(start_vector[n] - last_v[n]) for n in start_vector])
        if err < nodeCount * 1.0e-9:
            return start_vector
        cur_itr+=1
    raise nx.PowerIterationFailedConvergence(max_iter)

print("\nPagerank Implementation - End\n")

def assign_default_if_none_provided(graph,nodeCount,vector):
    if vector is None:
        # Assign uniform vector vector if not given
        output_dict = dict.fromkeys(graph, 1.0 / nodeCount)
    else:
        summation = float(sum(vector.values()))
        output_dict = dict((k, v / summation) for k, v in vector.items())
    return output_dict

def main():

    process_begins = time.time()
    #Constants Starts
    ifilename ="web-Google.txt"
    #Constants Ends
    
    inputGraph = input_graph_creation()
    ReadFile(inputGraph)
    pr = pagerank_implementation(inputGraph)

    print("\nResult:")
    print("\nSaving all results to pagerankresults.json\n")
    #printing all results to pagerankresults.json
    result = sorted(pr.items(), key=operator.itemgetter(1), reverse = True)
    with open('pageRankResults.json', 'w', encoding ='utf8') as json_file:
        json.dump(result, json_file, ensure_ascii = True)
    print("\nPrinting Top10 Results\n")
    print(result[:10])
    
    print("--- Total run time: %s seconds ---" % (time.time() - process_begins))

if __name__ == "__main__":
    main()
