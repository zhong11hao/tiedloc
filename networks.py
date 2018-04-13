import networkx as nx
import os
import agents
import random
from collections import OrderedDict

class CPS_network(nx.Graph):
    bctlist = []
    distmat = {}
    name = ''
    def __init__(self, parameters):
        nx.Graph.__init__(self)
        for key, val in  parameters['CPS'].iteritems():
            setattr(self, key,  val)
        
        if "picklePack_edge_data" in parameters['CPS']:
            self.add_nodes_from(range(self.num_of_nodes))
            self.add_edges_from(parameters['CPS']["picklePack_edge_data"])
        else:
            self.loadData()   
            self.loadDistMat()
            self.calcCentrality()
            if parameters["response_protocol"]["auxiliary"]:
                self.calcAux()
            degrees = list(self.degree().values())
            self.average_degree = float(sum(degrees))/len(degrees)
        
        for node_id in self.nodes_iter(data=False):
            self.node[node_id] = agents.Node(node_id)
        for vertex1, vertex2 in self.edges_iter(data=False):
            self[vertex1][vertex2] = agents.Link(vertex1, vertex2)
        
         
    def pack(self, picklePack = None): 
        if  picklePack is None:
            picklePack = {}
        packData={"picklePack_edge_data":self.edges()}
        for a in ["auxThreshold","average_degree","distmat", "name","num_of_nodes", "bctlist"]:
            try:
                packData[a] = getattr(self,a)
            except AttributeError:
                continue
        picklePack["CPS"] = packData     
        return picklePack
    
    def calcCentrality(self):
        if not self.bctlist:      
            if hasattr(self, "betweenness_centrality"):  
                print "Loading Centrality" 
                infile = open(os.path.join('data', self.betweenness_centrality),'r')
                for line in infile:
                    nums = line.split(" ")
                    self.bctlist.append([float(nums[0]), int(nums[1])])
                infile.close()
            else:
                print "Calculating Centrality...may take a while"     
                bct = nx.betweenness_centrality(self)
                self.bctlist =[[nval,nid] for nid,nval in bct.items()] 
                self.bctlist.sort(reverse=True)         
            #--- for testing save the centrality
            #outfile = open(os.path.join('data', "Centrality.txt"),'a')
            #for row in self.bctlist:
            #    outfile.write(str(row[0])+" "+str(row[1]+"\n"))
            #outfile.close()
            
    def calcAux(self):
        print "Calculating Aux Threshold...may take a while"
        self.auxThreshold = [x*y for x, y in zip(nx.clustering(self).values(), self.degree().values())]
        
    def failureModel(self, parameters, env, R):
        if parameters['failure_model']['name'] == "Watts cascade":
            for node_id in self.nodes_iter(data=False):
                env.process(self.node[node_id].node_fail_watts(env, float(parameters['failure_model']['phi']),float(parameters['failure_model']['fail_speed']), self, R))
            for vertex1, vertex2 in self.edges_iter(data=False):
                env.process(self[vertex1][vertex2].edge_fail_watts(env, float(parameters['failure_model']['fail_speed']), self, R))
    
    def initialFailures(self, env, parameters):
        tnds = self.number_of_nodes()
        for node_id in random.sample(xrange(0, tnds-1), int(parameters['failure_model']['initial_failures'])):
            print node_id
            self.failed(env, self.node[node_id])
            
    def failed(self, env, f1):
        f1.state =  1 
        f1.failedTime = env.now
        env.failedElements+=1
        env.failures.append(f1)
        env.totalFailures+=1
        
    def restored(self, env, f1):
        f1.state = 0
        latency = env.now - f1.failedTime
        env.latency.append(latency)
        env.totalLatency+=latency
        env.failedElements-=1
             
    def loadData(self):
        savename = self.name
        if self.name == "Power Grid of Western States of USA": 
            self.num_of_nodes = 4941   
            self.add_nodes_from(range(self.num_of_nodes))
            linkflag = 0
            fromnode = -1
            fobj = open(os.path.join('data', 'power.txt'),'r')
            for line in fobj:
                try:
                    line = int(line)
                except ValueError:
                    continue
                if linkflag == 0:
                    fromnode = line
                    linkflag = 1
                else:
                    linkflag = 0
                    self.add_edge(fromnode,line)
            fobj.close()
        elif self.name == "Barabasi Albert Scale-Free Network":
            nx.Graph.__init__(self, nx.barabasi_albert_graph(int(self.num_of_nodes),int(self.new_node_to_existing_nodes)) )# scale-free BA model
        elif self.name == "Binomial Graph":
            nx.Graph.__init__(self, nx.binomial_graph(int(self.num_of_nodes),int(self.average_degree)*1.0/int(self.num_of_nodes)) )# Random Graph
        elif self.name == "Watts-Strogatz Small-World Model":
            nx.Graph.__init__(self, nx.watts_strogatz_graph(int(self.num_of_nodes),int(self.average_degree),0.3) )# WS model    
        self.name=savename 
        
    def loadDistMat(self):
        if hasattr(self, "distance_matrix"):  
            print "Loading distance matrix" 
            infile = open(os.path.join('data', self.distance_matrix),'r')
            lineid = 0
            for line in infile:
                self.distmat[lineid] = {}
                nums = line.split(" ")
                for key, x in enumerate(nums):
                    try:
                        self.distmat[lineid][key] = float(x)
                    except ValueError:
                        pass
                #self.distmat[lineid] = OrderedDict(sorted(self.distmat[lineid].items(), key=lambda t: t[1]))
                
                lineid+=1
            infile.close()
        else:
            print "Calculating Distance Matrix...may take a while"     
            self.distmat=nx.all_pairs_shortest_path_length(self)
            maxLength = max([max([self.distmat[x][y] for y in self.distmat[x]]) for x in self.distmat])
 
            for a in self.nodes_iter():
                if not (a in self.distmat):
                    self.distmat[a] = {}
                for b in self.nodes_iter():
                    if not (b in self.distmat[a]):
                        self.distmat[a][b] = maxLength
                #self.distmat[a] = OrderedDict(sorted(self.distmat[a].items(), key=lambda t: t[1]))
                
            #--- for testing save the distMat
            #outfile = open(os.path.join('data', "distMat.txt"),'a')
            #for row in self.distmat:
            #    for a in self.distmat[row]:
            #        outfile.write(str(self.distmat[row][a])+" ")
            #    outfile.write("\n")
            #outfile.close()
            #exit()                 
        
        
class Service_team(nx.Graph):
    def __init__(self, parameters, env):
        nx.Graph.__init__(self)
        for key, val in  parameters['service_team'].iteritems():
            setattr(self, key,  val)
            
        if self.name == "regular graph":
            nx.Graph.__init__(self, nx.random_regular_graph(parameters['service_team']['team_degree'], parameters['service_team']['team_members']))
        
        for node_id in self.nodes_iter(data=False):
            self.node[node_id] = agents.Server(node_id, env, self)
        
        self.lastDispatch = -1
            
    def allocation(self,parameters,cpsnetwork):
        if parameters['service_team']['initial_allocation'] == "Centrality": # other allocation methods may apply
                self.centrality_allocation(parameters, cpsnetwork)
    
    def centrality_allocation(self, parameters, cpsnetwork):
        cpsnetwork.calcCentrality()
        for node_id in self.nodes_iter(data=False):
            self.node[node_id].allocateTo(cpsnetwork.node[cpsnetwork.bctlist[node_id%cpsnetwork.number_of_nodes()][1]])

        
        
        
                   
            