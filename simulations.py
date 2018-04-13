import simpy
import random
import multiprocessing
import networks
import responsestrategies
import numpy
import json
import gc

import pickle

def init():
    env = simpy.Environment()
    # inprocessSupport
    env.failedElements = 0
    env.failures = []
    
    # results
    env.prevented = 0
    env.auxlines = 0
    env.totalFailures  = 0
    env.failedAtStep = [] # for Recoverability, restoration ratio
    env.totalDistance = 0
    env.totalLatency = 0
    env.latency = [] # to represent QoS
    
    return env

def simEachTimeUnit(env):
    #logfile = open("log.txt",'a')
    while True:
        #logfile.write("time:"+str(env.now))
        #logfile.write(" failed: "+str(env.failedElements))
        #logfile.write(" prevented: "+str(env.prevented))
        #logfile.write("\n")
        
        env.failedAtStep.append(env.failedElements)
   
        yield env.timeout(1)
        
def simRandomSeed(parameters):
    random.seed(int(parameters["simulation_param"]["seed"]))
    
def start(env, parameters, repnum):
    print "Start of Simulation replication "+str(repnum)
    env.run(until=float(parameters["simulation_param"]["simulation_length"]))
    
def parallelDispatch(parameters, G):
    rnums = [int(1000 * random.random()) for i in xrange(int(parameters["simulation_param"]["replications"]))]
    #manager = multiprocessing.Manager()
    parameters['gpack'] = G.pack()
    
    
    gc.collect()

    if (int(parameters["simulation_param"]["processors"])> 1):
        
        pool = multiprocessing.Pool(processes = int(parameters["simulation_param"]["processors"]))
        return pool.map(worker, [ (parameters, rint, repnum) for repnum, rint in enumerate(rnums)], )
    elif (int(parameters["simulation_param"]["processors"])==1):
        return map(worker, [ (parameters, rint, repnum) for repnum, rint in enumerate(rnums)], )

 
def worker(args):
    [parameters, rint, repnum] = args
    random.seed(rint)
    # create simulation
    env = init()
    
    cpsnetwork = networks.CPS_network(parameters['gpack'])
    cpsnetwork.initialFailures(env, parameters) 
    serviceteam = networks.Service_team(parameters, env)
    serviceteam.allocation(parameters, cpsnetwork)
    cpsnetwork.failureModel(parameters, env, serviceteam)
    
# set response strategy
    env.process(responsestrategies.init(env,parameters, cpsnetwork, serviceteam))

# set display
    env.process(simEachTimeUnit(env))
    start(env, parameters, repnum)
    print rint
    
    # add failures haven't fixed.
    for node_id in cpsnetwork.nodes_iter(data=False):
        if cpsnetwork.node[node_id].state == 1:
            env.totalLatency+=env.now-cpsnetwork.node[node_id].failedTime
    for vertex1, vertex2 in cpsnetwork.edges_iter(data=False):
        latency = 0
        try:
            if 1 == cpsnetwork[vertex1][vertex2].state:
                latency = env.now-cpsnetwork[vertex1][vertex2].failedTime
        except AttributeError:
            if 1 == cpsnetwork[vertex2][vertex1].state:
                latency = env.now-cpsnetwork[vertex2][vertex1].failedTime
        env.totalLatency+=latency

    
    envPack ={}
    for a in ["totalLatency","latency","totalFailures", "prevented","totalDistance", "auxlines","failedAtStep"]:
        envPack[a] = getattr(env,a)
    del cpsnetwork, serviceteam
    gc.collect()
    return envPack

# output handling

def statistics(results):
    # the results is a list of env
    stats = {"Total_Latency":[],
              "Mean_Latency_QoS": [],
              "Mean_Recovery_Time":[],
              "Mean_Recovery_Time_withINF":[],
             "Total_Failures":[],
             "Preventability":[],
             "Total_Distance_Traveled_by_Agent":[],
             "Aux_Lines":[]
             }
    for i in range(len(results[0]["failedAtStep"])):
        stats["Step_"+str(i).zfill(3)]=[]
    Recoverability = 0
    
    for envPack in results:
        stats["Total_Latency"].append(envPack["totalLatency"])
        stats["Mean_Latency_QoS"]+=envPack["latency"]
        stats["Total_Failures"].append(envPack["totalFailures"])
        stats["Preventability"].append(float(envPack["prevented"])/(envPack["prevented"] + envPack["totalFailures"]))
        stats["Total_Distance_Traveled_by_Agent"].append(envPack["totalDistance"])
        stats["Aux_Lines"].append(envPack["auxlines"])
        for i, x in enumerate(envPack["failedAtStep"]):
            stats["Step_"+str(i).zfill(3)].append(x)
        recoverFlag = findRecoveryTime(envPack["failedAtStep"])
        stats["Mean_Recovery_Time_withINF"].append(recoverFlag)
        if not numpy.isposinf(recoverFlag):
            stats["Mean_Recovery_Time"].append(recoverFlag)
            Recoverability += 1

    
    statResults = {"Recoverability":Recoverability/float(len(stats["Total_Latency"])),
                   "Median_Recovery_Time": numpy.median(stats["Mean_Recovery_Time_withINF"])}

    for  key, val in stats.iteritems():     
        statResults[key] = {"Mean":numpy.mean(val), "SEM": numpy.std(val)/numpy.sqrt(len(val))}# mean and standard error of mean
    return (statResults, stats)

def findRecoveryTime(failedList):
    k = next((i for i,k in enumerate(failedList) if k == 0), -1)
    if k==-1:
        maxVal = max(failedList)
        #print maxVal
        maxList = [i for i,x in enumerate(failedList) if (x == maxVal)]
        if maxList[-1] == (len(failedList)-1):
            k = numpy.inf
        else:
            coef = numpy.polyfit(range(maxList[-1], len(failedList)), failedList[maxList[-1]:len(failedList)], 1)
            if coef[0] >=0: # heightest order first
                k = numpy.inf
            else:
                k = -coef[1]/coef[0]
    return k

def saveResults(pathToInputFile, statResults, samples):
    (pathToOutput, sep, jster)= pathToInputFile.partition('.')
    
    statPath=pathToOutput+"OUTPUT"+sep+jster
    outfile = open(statPath,"w")
    outfile.write(json.dumps(statResults, sort_keys=True, indent=4, separators=(',', ': ')))
    outfile.close()
    
    samplePath=pathToOutput+"SAMPLES"+sep+jster
    outfile=open(samplePath,"w")
    outsamples = {key:val for key, val in samples.iteritems() if key in ["Total_Latency", "Preventability", "Total_Distance_Traveled_by_Agent", "Total_Failures"]}
    outfile.write(json.dumps(outsamples, sort_keys=True, indent=4, separators=(',', ': ')))
    outfile.close()
    
