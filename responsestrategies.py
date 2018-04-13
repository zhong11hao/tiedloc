import sys
import agents
import math
import networkx as nx

def init(env, parameters, G, R):
    G.newFailures = env.event()
    env.process(maintenanceCheck(env, parameters, G, R))
    while True:
        if R.lastDispatch > -1:
            events = [R.node[a].freedAgentEvent for a in R.nodes_iter()]
            events.append(G.newFailures)
            yield env.any_of(events)#conditional event on (time-out with free agents) or newly freed agents
            #results.
        if R.lastDispatch >= env.now:
            continue
        R.lastDispatch = env.now
        
        if parameters["response_protocol"]["auxiliary"]==True:
            if env.now == 5:
                wireaux(env,parameters,G,R)
            #else:
            #    wireaux(env,parameters,G,R)
			
        response_protocol = parameters["response_protocol"]["name"]
        if response_protocol in ["nearest", "activity"]:
            #for f1 in env.failures:
            #    calcNearness(f1, env, G, R)
            nNSchedule(env, G, R, response_protocol)
        else:               
            env.failures.sort(reverse = True, cmp = getattr(sys.modules[__name__], response_protocol))
            dispatcher(env, G, R) # build up the queues for resources 
        #improvement
           # GA or others
           # schedule = GA(schedule, env)
        # implementation
        implementSchedule(R, G, env) # only implement the first in schedule return others back to env.failures
        # 1.trim all request beyond now, 2. assigning process to queued jobs.
def maintenanceCheck(env, parameters, G, R):
    
    check_base =  hash(str(env.failures))
    while True:
        checker = hash(str(env.failures))
        if checker != check_base:
            for agentId in R.nodes_iter():
                if not R.node[agentId].waiting_list:
                    G.newFailures.succeed()
                    G.newFailures = env.event()
                    break
                elif R.node[agentId].waiting_list[-1]["finishingTime"] <= env.now:
                    G.newFailures.succeed()
                    G.newFailures = env.event()
                    break
        check_base = checker
        yield env.timeout(float(parameters["response_protocol"]["frequency"]))
        
def FCFS(a,b):
    return cmp(a.failedTime - b.failedTime, 0)

def nearest(a, b):
    return cmp(a[1] - b[1], 0)

def activity(a, b):
    activity = cmp(b[0] - a[0], 0)
    if activity == 0:
        return cmp(a[1] - b[1], 0)
    else:
        return activity
    
def calcActivity(f1, env, G, R):
    #calculate and return priority
    if hasattr(f1, "activity"):
        if f1.activity['time'] == env.now:
            return f1.activity['value']

    if isinstance(f1, agents.Link):
        f1.activity = {"time":env.now, "value":min(calcActivity(G.node[f1.vertex1], env, G, R), calcActivity(G.node[f1.vertex2], env, G, R))}
        return f1.activity['value']
    
    node_id = f1.id
    phyi = 0
    failededge = 0
    auxAdjuster = 0
    for neighbor in G.neighbors_iter(node_id):
        try:
            curstate = G[node_id][neighbor].state
            thislink = G[node_id][neighbor]
        except AttributeError:
            curstate = G[neighbor][node_id].state
            thislink = G[neighbor][node_id]
        if curstate==1:
            if hasattr(thislink, "auxJustIn"):
                if thislink.auxJustIn == True:
                    auxAdjuster+=1
                    continue
            failededge+=1
            breakflag = 0
            for ai in f1.hosting:# preventability
                for a_nei in R.neighbors_iter(ai):
                    if R.node[a_nei].current_pos == neighbor:
                        failededge -= 1# see here difference
                        breakflag = 1
                        break
                if breakflag == 1:
                    break
                            
    if (G.degree(node_id)-auxAdjuster)==0:
        phyi = 0
    else:
        phyi = failededge * 1.0 / (G.degree(node_id) - auxAdjuster)

    prio = 1 - phyi
    f1.activity = {"time":env.now, "value":prio} 
    return f1.activity['value']

def calcNearness(f1, env, G, R):
    #calculate priority 
    # for each failure, find out the closest available 
    if hasattr(f1, "nearness"):
        if f1.nearness['time'] == env.now: # last updated time
            return
        
    if isinstance(f1, agents.Link):
        (leastAgent, job) = findCostlessAgent(env, G, R, f1, f1.vertex1)
        (leastAgent2, job2) = findCostlessAgent(env, G, R, f1, f1.vertex2, leastAgent)
        close_to_finish = max(job['finishingTime'], job2['finishingTime'])
    else:   
        (leastAgent, job) = findCostlessAgent(env, G, R, f1, f1.id)
        close_to_finish = job['finishingTime']
    
    f1.nearness = {"time":env.now, "value":close_to_finish}
    

def aliveLinks(G, node_id):
    alivelinks = 0
    for neighbor in G.neighbors_iter(node_id):
        try:
            curstate = G[node_id][neighbor].state
        except AttributeError:
            curstate = G[neighbor][node_id].state
        alivelinks += 1 - curstate
    return alivelinks

def protectionDensity(G, node_id):
    protected = 0
    tnodes=G.degree(node_id)
    for neighbor in G.neighbors_iter(node_id):
        if G.node[neighbor].hosting:
            protected+=1
            for nei2 in G.neighbors_iter(neighbor):
                if nei2!=node_id:
                    if G.node[nei2].hosting:
                        protected+=1
                    tnodes+=1
    return protected
    
def wireaux(env, parameters, G, R):
    for fdummy in env.failures:
        
        if isinstance(fdummy, agents.Node):
            target = fdummy
            if G.node[target.id].hosting:
                continue
            #if protectionDensity(G, target.id) > 0:
            #    continue
            for neighbor in G.neighbors_iter(target.id):
                breakflag = False
                if G.node[neighbor].state == 1:
                    continue
                if G.degree(neighbor) >= G.average_degree:
                    continue
                #if aliveLinks(G, neighbor)<=1:
                #    continue
                #if protectionDensity(G, neighbor) > 0:
                #    continue
                if not G.node[neighbor].hosting:
                    for nei2 in G.neighbors_iter(neighbor):
                        if (target.id == nei2):
                            continue
                        if G.node[nei2].hosting:
                            continue
                        if G.node[nei2].state == 1:
                            continue
                        #if aliveLinks(G, nei2)<=1:
                        #    continue
                        if (G.has_edge(target.id, nei2) == False):
                            q = 0.0
                            for neiOne in G.neighbors_iter(target.id):
                                if G.has_edge(neiOne, nei2):
                                    q += 1.0
                            if G.degree(nei2)>(G.average_degree*q):
                                continue
                            if (q > G.auxThreshold[target.id]):
                            #if (q > transNetworkAuxThreshold(target.id, G, R)):
                                G.add_edge(target.id, nei2)
                                G[target.id][nei2] = agents.Link(target.id,nei2)
                                env.process(G[target.id][nei2].edge_fail_watts(env, float(parameters['failure_model']['fail_speed']), G, R))
                                
                                G.failed(env, G[target.id][nei2])    
                                
                                G[target.id][nei2].auxJustIn = True
                                env.auxlines+=1
                                #print "aux", env.now
                                breakflag = True
                                break
                if breakflag == True:
                    break        
                                
def transNetworkAuxThreshold(node_id, G,R): # not used
    e=0
    neighbors = G.neighbors(node_id)
    degree = len(neighbors)
    if degree>1:
        e = nx.clustering(G, node_id)*degree*(degree-1)/2.0
    for agid in G.node[node_id].hosting:
        for agneb in R.neighbors_iter(agid): 
            if R.node[agneb].current_pos != node_id:
                degree+=1
            elif R.node[agneb].current_pos in neighbors:
                e+=1
    if degree > 1:
        return e * 1.0 / (degree - 1)
    else:
        return 0

def dispatcher(env, G, R): # this is a centralized scheduler
    
    if not hasattr(env, 'agent_schedule'):
        env.agent_schedule = [[job.params for job in R.node[nid].resource.users]
                + [job.params for job in R.node[nid].resource.queue] for nid in R.nodes_iter()]
    
    #if env.now == 15:
    #        pass
    
    reAssign = []
    nextIteration = []
       
    while env.failures:
        f1 = env.failures.pop(0) # remove and return the first
        v1 = -1
        if isinstance(f1, agents.Link):
            v1 = f1.vertex1
        else:
            v1 = f1.id
        
        (leastAgent, job) = findCostlessAgent(env, G, R, f1, v1)
        env.agent_schedule[leastAgent].append(job)#
                
        #handling finishing time and collaboration       
        if isinstance(f1, agents.Link):
                        
            v2 = f1.vertex2
            (v2Agent, job2) = findCostlessAgent(env, G, R, f1, v2, leastAgent)
                                
            if job["finishingTime"] < job2["finishingTime"]:
                #delays = v2LeastCost-leastCost # will start moving without delays
                job["finishingTime"] = job2["finishingTime"]     # test if it is updated
                #schedule[leastAgent][-1]["departingTime"]+= delays
                #schedule[leastAgent][-1]["arrivalTime"]+= delays
            else:
                job2["finishingTime"] = job["finishingTime"]
                
            job["collaborator"] = v2Agent                                          
            job2["collaborator"]=leastAgent
            env.agent_schedule[v2Agent].append(job2)
            
            if job["assign"]!=job2["assign"]:
                reAssign.append(f1)
                job["assign"]= False
                job2["assign"] = False
            elif job["assign"] == False:
                nextIteration.append(f1)
        elif job["assign"] == False:
            nextIteration.append(f1)
        
    # filter out jobs that are not current
    for agentId in R.nodes_iter():
        env.agent_schedule[agentId] = [ job for job in env.agent_schedule[agentId] if job["assign"]]

                
    # re-assign collaboration
    while reAssign:
        f1 = reAssign.pop(0)
        v1 = f1.vertex1
        (leastAgent, job) = findCostlessAgent(env, G, R, f1, v1)
        env.agent_schedule[leastAgent].append(job)
        v2 = f1.vertex2
        (v2Agent, job2) = findCostlessAgent(env, G, R, f1, v2, leastAgent)
        if job["finishingTime"] < job2["finishingTime"]:
            job["finishingTime"] = job2["finishingTime"]     # test if it is updated
        else:
            job2["finishingTime"] = job["finishingTime"]
        job["collaborator"] = v2Agent                                          
        job2["collaborator"]=leastAgent
        env.agent_schedule[v2Agent].append(job2)
        job["assign"] = True
        job2["assign"]=True
  
    env.failures = nextIteration

def findCostlessAgent(env, G, R, f1, target, collaborator = None):
    leastCost = -1
    leastAgent = -1
    departingTime = -1
    
    elligibleNodes = R.nodes_iter()
    if not collaborator is None:
        elligibleNodes = R.neighbors_iter(collaborator)
        
    for agent_id in elligibleNodes: # the inner part is an agent function, can be moved to agents/server
        (cost, ag_departingTime, ag_arrivalTime) = R.node[agent_id].bidding(env, G, target)
        
        if (cost < leastCost) or (leastCost < 0):
            leastCost = cost
            leastAgent = agent_id
            departingTime = ag_departingTime
            arrivalTime = ag_arrivalTime
    
    job = {"node":target, "failure":f1, "departingTime":departingTime,
                    "arrivalTime":arrivalTime, "finishingTime":leastCost, 
                    "processAssigned":False, "assign":(departingTime == env.now)}
    return (leastAgent, job)

def nNSchedule(env, G, R, response_protocol):
    # nearest neighbor and related protocols
    # calcuate distance matrix for idle agents and failures
    idleAgents = []
    for agent_id in R.nodes_iter():
        idelone = {'agent_id':agent_id, 'pos':R.node[agent_id].current_pos}
        if hasattr(env, 'agent_schedule'):
            if len(env.agent_schedule[agent_id]) > 0:
                if env.now < env.agent_schedule[agent_id][-1]["finishingTime"]:
                    #idelone['pos'] = env.agent_schedule[agent_id][-1]['node']
                    #idelone['finishingTime'] = env.agent_schedule[agent_id][-1]["finishingTime"]
                    continue
        idleAgents.append(idelone)
    
    idel_mat = []
    for f1 in env.failures:
        activity = calcActivity(f1, env, G, R)
        for idler in idleAgents:
            dist = 0
            if isinstance(f1, agents.Link):
                dist = min(G.distmat[idler['pos']][f1.vertex1],
                           G.distmat[idler['pos']][f1.vertex2]) 
            else:
                dist = G.distmat[idler['pos']][f1.id]
            #dist += (idler['finishingTime'] - env.now)
            idel_mat.append((activity, dist, idler['agent_id'], f1))

    
    idel_mat.sort(cmp = getattr(sys.modules[__name__], response_protocol))
    
    if not hasattr(env, 'agent_schedule'):
        env.agent_schedule = [[job.params for job in R.node[nid].resource.users]
                + [job.params for job in R.node[nid].resource.queue] for nid in R.nodes_iter()] 
       
    while idel_mat:
        (activity, dist, agent_id, f1) = idel_mat.pop(0) # 
        env.failures.remove(f1)
        v1 = -1
        v2 = -1
        if isinstance(f1, agents.Link):
            cur_Dist = G.distmat[R.node[agent_id].current_pos][f1.vertex1]
            if cur_Dist <= dist:
                v1 = f1.vertex1
                v2 = f1.vertex2
            else:
                v1 = f1.vertex2
                v2 = f1.vertex1
        else:
            v1 = f1.id
        (cost, ag_departingTime, ag_arrivalTime) = R.node[agent_id].bidding(env, G, v1)
        
        job = {"node":v1, "failure":f1, "departingTime":ag_departingTime,
                    "arrivalTime":ag_arrivalTime, "finishingTime":cost, 
                    "processAssigned":False, "assign":(ag_departingTime == env.now)}
    
        env.agent_schedule[agent_id].append(job)#
                
        #handling finishing time and collaboration  
        v2Agent = -1     
        if isinstance(f1, agents.Link):
            # find available agent only
            leastCost = -1
            v2Agent = -1
            departingTime = -1
    
            for agentId in R.neighbors_iter(agent_id):
                (cost, ag_departingTime, ag_arrivalTime) = R.node[agentId].bidding(env, G, v2)
                if ag_departingTime > env.now:
                    continue
                if (cost < leastCost) or (leastCost < 0):
                    leastCost = cost
                    v2Agent = agentId
                    departingTime = ag_departingTime
                    arrivalTime = ag_arrivalTime
            if leastCost != -1:
                job2 = {"node":v2, "failure":f1, "departingTime":departingTime,
                    "arrivalTime":arrivalTime, "finishingTime":leastCost, 
                    "processAssigned":False, "assign":(departingTime == env.now)}
                if job["finishingTime"] < job2["finishingTime"]:
                    job["finishingTime"] = job2["finishingTime"]     # test if it is updated
                else:
                    job2["finishingTime"] = job["finishingTime"]
                
                job["collaborator"] = v2Agent                                          
                job2["collaborator"] = agent_id
                env.agent_schedule[v2Agent].append(job2)
                job["assign"] = True
                job2["assign"] = True
                # scheduling successful
                idel_mat = [(activity, dist, agentId, fx) for (activity, dist, agentId, fx) in idel_mat 
                            if  (agentId != agent_id) and (agentId != v2Agent) and (fx != f1)] 
				# 
            else:
                # scheduling unsuccessful
                job["assign"] = False
                env.agent_schedule[agent_id].pop()
                env.failures.append(f1)
                

        #elif job["assign"] == False:
            # scheduling unsuccessful
        #    env.failures.append(f1)
        #    env.agent_schedule[agent_id].pop() 
        else:
        # scheduling successful
            job["assign"] = True
        # remove busy agents and f1 from the idel_mat
            idel_mat = [(activity, dist, agentId, fx) for (activity, dist, agentId, fx) in idel_mat 
                        if (agentId != agent_id) and (fx != f1)] 
        # ( agentId != v2Agent) and  
    # filter out jobs that are not current
    for agentId in R.nodes_iter():
        env.agent_schedule[agentId] = [ job for job in env.agent_schedule[agentId] if job["assign"]]   

def implementSchedule(R, G, env):
############# implementation #################   
    for agentId in R.nodes_iter():
        for job in env.agent_schedule[agentId]:
            if job["processAssigned"]:
                continue
            if job["assign"]:
                job["processAssigned"] = True
                req = R.node[agentId].resource.request()
                req.params = job
                env.process(R.node[agentId].provideService(env, G, req))
                #print agentId, job['node']    
                
                   
        