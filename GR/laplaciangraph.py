##########################################################################
# module: dendro
# author: Pasindu Tennage
# email:  pasindu.13@cse.mrt.ac.lk
#
# python module to generate dependency graph
#
# (c) 2018 University of Utah, All rights reserved.
##########################################################################

from sympy import *
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import scipy.sparse as sparse
import csv
import numpy as np
from sklearn.cluster import *

def printGraph(G, original_dependencies, clusters, n_clusters=7):
    colors = {0: "black", 1: "white",2: "yellow", 3: "red", 4: "green", 5: "blue", 6: "gray", 7: "purple" }
    original_dependencies = list(original_dependencies.keys())
    cluster_labels = clusters.labels_.tolist()
    labels = {}
    for i in range(len(original_dependencies)):
        labels[original_dependencies[i]] = cluster_labels[i]
    color_map = []
    for node in G:
        if node not in original_dependencies:
            color_map.append(colors[7])
        else:
            color_map.append(colors[labels[node]])
    nx.draw(G, node_color=color_map, with_labels=True)
    plt.show()

def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def RepresentsFloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def getReducedArguments(expression, result):
    args = expression.args
    func = expression.func
    if("grad" in str(func)):
        result.append(str(expression))
    elif("Symbol" in str(func)):
        result.append(str(expression))
    else:
        for arg in args:
            if(len(arg.args)==0):
                if((not RepresentsFloat(str(arg))) and (not RepresentsInt(str(arg))) and (str(arg) not in result)):
                    result.append(str(arg))
            else:
                getReducedArguments(arg, result)
    return result


def makeGraph(_v):
    G=nx.DiGraph()
    dependencies = {}
    counter = 1
    for i in _v[0]:
        result = []
        result = getReducedArguments(i[1], result)
        #print(str(i[0])+"="+str(result))
        #print(str(i[0])+"="+str(i[1]))
        #print()
        dependencies[str(i[0])] = len(result)
        G.add_node(str(i[0]))
        for arg in result:
            if(arg not in dependencies.keys()):
                dependencies[arg] = 0
            G.add_node(str(arg))
            G.add_edge(str(i[0]),str(arg))
    for i in _v[1]:
        G.add_node("Equation" + str(counter))
        result = []
        result = getReducedArguments(i, result)
        dependencies["Equation" + str(counter)] = len(result)
        #print("Equation" + str(counter)+"="+str(result))
        #print("Equation" + str(counter)+"="+str(i))
        #print()
        for arg in result:
            if (arg not in dependencies.keys()):
                dependencies[arg] = 0
            G.add_node(str(arg))
            G.add_edge("Equation" + str(counter),str(arg))
        counter = counter+1
    return (G)

def getNumNodes(G):
    return len(G.nodes())

def getDimensions(matrix):
    return matrix.shape

def getAdjecencyMatrix(G):
    return nx.to_numpy_matrix(G)

def getGraphLaplacian(G):
    return nx.laplacian_matrix(G)

def doSpectralClustering(adj_mtr):
    # Cluster
    sc = SpectralClustering(n_clusters = 4, assign_labels = "kmeans")
    sc.fit(adj_mtr)
    return sc

def printSpectralClustering(sc, graph, dependencies):
    # print(len(dependencies.keys()))
    # print(len(list(graph.nodes())))
    # print(str(sc.labels_.size))

    print('spectral clustering')
    nodes = list(graph.nodes())
    labels = sc.labels_
    data = []
    data.append(["Node name", "Label", "Number of dependencies"])
    for i in range(len(nodes)):
        data.append([str(nodes[i]), str(labels[i]), str(dependencies[str(nodes[i])])])
    myFile = open('SpectralClustering.csv', 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(data)

def getEigenValuesandVectors(matrix):
    return sparse.linalg.eigs(matrix)