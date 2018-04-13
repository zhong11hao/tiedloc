import networkx as nx
import simpy

class Node():
    id = -1
    state = 0
        
    def __init__(self, id):
        self.id = id
        self.hosting = []
        
    def node_fail_watts(self, env, phi, tstep, G, R): # using Watts model
        if self.state == 0:
            failededge = 0
            auxAdjuster = 0
            preventer = 0
            for neighbor in G.neighbors_iter(self.id):
                iflinkfailed = 0
                try:
                    iflinkfailed = G[self.id][neighbor].state
                    thislink = G[self.id][neighbor]
                except AttributeError:
                    iflinkfailed = G[neighbor][self.id].state
                    thislink = G[neighbor][self.id]
                    
                if iflinkfailed == 1:
                    if hasattr(thislink, "auxJustIn"):
                        if thislink.auxJustIn == True:
                            auxAdjuster += 1
                            continue
                    failededge += 1
                    breakflag = 0
                    for ai in self.hosting:# preventability
                        for a_nei in R.neighbors_iter(ai):
                            if R.node[a_nei].current_pos == neighbor:
                                failededge -= 1
                                breakflag = 1
                                preventer = 1 #just prevent a cause...
                                break
                        if breakflag == 1:
                            break
                            
            if (nx.degree(G, self.id) - auxAdjuster) == 0:
                phyi = 0
            else:
                phyi = failededge * 1.0 / (nx.degree(G, self.id) - auxAdjuster)
            if phyi > phi:
                G.failed(env, self)
                #print "+1", self
            else:
                env.prevented += preventer
                
        yield env.timeout(tstep)
        env.process(self.node_fail_watts(env, phi, tstep, G, R))
            
            
class Link():
    state = 0
    def __init__(self,vertex1, vertex2):
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        
    def edge_fail_watts(self, env, tstep, G, R):
        if self.state == 0:
            if ((G.node[self.vertex1].state==1) and (G.node[self.vertex2].state==0)):
                if len(G.node[self.vertex1].hosting)==0:#preventability
                    self.state=1
                else:
                    env.prevented+=1

            elif((G.node[self.vertex2].state==1)and (G.node[self.vertex1].state==0)):
                if len(G.node[self.vertex2].hosting)==0:#preventability
                    self.state=1
                else:
                    env.prevented+=1

            elif((G.node[self.vertex2].state==1)and (G.node[self.vertex1].state==1)):
                self.state=1
                
                breakflag = 0
                for ai in G.node[self.vertex1].hosting:# preventability
                    for a_nei in R.neighbors_iter(ai):
                        if R.node[a_nei].current_pos==self.vertex2:
                            self.state=0
                            env.prevented += 1
                            breakflag = 1
                            break
                    if breakflag == 1:
                        break
            if self.state == 1:
                G.failed(env,self)
                #print "+1", self
        yield env.timeout(tstep)
        env.process(self.edge_fail_watts(env, tstep, G, R))
                           
class Server():
    id = -1
    #current_pos = -1
    #waiting_list 

    def __init__(self,id, env, R):
        self.id = id
        self.agent_travel_speed = R.agent_travel_speed
        self.repairing_time = R.repairing_time
        
        self.waiting_list = []
        
        self.state = 0
        self.resource = simpy.Resource(env, capacity=1)
        self.freedAgentEvent = env.event() # no need of callbacks
        
    def allocateTo(self, cpsnode):
        self.current_pos = cpsnode.id
        cpsnode.hosting.append(self.id)
        
    def bidding(self, env, G, target):
        ag_from = self.current_pos
        ag_departingTime = env.now
        if hasattr(env, 'agent_schedule'):
            if len(env.agent_schedule[self.id]) > 0:
                ag_from = env.agent_schedule[self.id][-1]["node"]
                ag_departingTime = max(ag_departingTime,  env.agent_schedule[self.id][-1]["finishingTime"])
            
        pathlen = G.distmat[ag_from][target]  
        cost = pathlen/float(self.agent_travel_speed)      
        cost+=self.repairing_time
        cost+=ag_departingTime
        ag_arrivalTime = cost - self.repairing_time
        return (cost, ag_departingTime, ag_arrivalTime)
 
    def provideService(self,env, G, job):
        yield job
        self.state = 1
        if env.now < job.params["arrivalTime"]:
            G.node[self.current_pos].hosting.remove(self.id)
            self.current_pos = float('nan')
            # start moving
            moveTimve = job.params["arrivalTime"] - env.now
            yield env.timeout(moveTimve)
            env.totalDistance+= moveTimve*float(self.agent_travel_speed)
            # arrival
        self.current_pos = job.params["node"]
        if not self.id in G.node[self.current_pos].hosting:
            G.node[self.current_pos].hosting.append(self.id)
        

        if env.now < job.params["finishingTime"]:
                yield env.timeout(job.params["finishingTime"] - env.now)
            
        self.state = 0
        #print "-1", self, f1
        self.resource.release(job)
        if (not self.resource.users) and (not self.resource.queue):
            self.freedAgentEvent.succeed()
            self.freedAgentEvent = env.event()
                
        if isinstance(job.params["failure"], Link):
            if self.current_pos == job.params["failure"].vertex1:
                if hasattr(job.params["failure"],"auxJustIn"):
                    job.params["failure"].auxJustIn = False # flip the flag once
                return # only recover once
        G.restored(env, job.params["failure"])

            
        

        
    
        
        

        
        

    
            
        
        
        


    