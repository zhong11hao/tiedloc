import json
import argparse

import networks
import simulations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get input-file')
    parser.add_argument('pathToInputFile', default='input.json', metavar='pathToInputFile', type=str, help='path to the input file',  nargs='?')
    args = parser.parse_args()
#load input
    json_data=open(args.pathToInputFile)
    inputs = json.load(json_data)
    json_data.close()

#check input here
    print "Input:"
    print json.dumps(inputs, sort_keys=True, indent=4, separators=(',', ': '))


#Experiment setup
    simulations.simRandomSeed(inputs)
    cpsnetwork = networks.CPS_network(inputs)

#create parallel processing 

    results = simulations.parallelDispatch(inputs, cpsnetwork)
    
# results handling    
    (statResults, samples) = simulations.statistics(results)
    print "Results:"
    print json.dumps(statResults, sort_keys=True, indent=4, separators=(',', ': '))
    simulations.saveResults(args.pathToInputFile, statResults, samples)
    