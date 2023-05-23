# Words2Wires: GenerateDevice.py
# This is a basic example of using the Words2Wires library to generate device specifications
# including bills-of-materials, pinouts, schematics (as netlists), and microcontroller code.
# This wrapper reads the text device specifications from a TSV file. 

import json
import argparse

from Words2Wires import Words2Wires
    
# Load a TSV file with the task descriptions.
# The first line of the file is a header.
# Should return a list of dictionaries, where each dictionary is a row in the file.  The key names are the column names from the header.
def loadTaskDescriptions(filenameIn:str):
    tasks = []

    with open(filenameIn, 'r') as f:
        header = f.readline()
        header = header.strip()
        header = header.split('\t')
        for line in f:
            line = line.strip()
            if (line == ''):
                continue
            if (line[0] == '#'):
                continue
            line = line.split('\t')

            packed = {}
            for idx, value in enumerate(line):
                packed[header[idx]] = value

            tasks.append(packed)            

    return tasks


def generateForOneTask(t2d, task, args):
    platformStr = task['platform']
    taskNameStr = task['taskName']
    taskDescriptionStr = task['taskDescription']
    generationMode = args['generationMode']
    temperature = args['temperature']
    engineStr = args['engine']

    print("----------------------------------------")
    print("Task: " + taskNameStr)
    print("Platform: " + platformStr)
    print("Description: " + taskDescriptionStr)
    print("Generation mode: " + generationMode)
    print("Temperature: " + str(temperature))
    print("----------------------------------------")

    # Generate the code
    packet = t2d.generate(microcontrollerPlatform=platformStr, task=taskDescriptionStr, temperature=temperature, generationMode=generationMode)

    # Also save any keys from 'task' in 'packet' (so that any information from the TSV file is preserved in the output)
    for key in task:
        if (key not in packet):
            packet[key] = task[key]
    # Also save any keys from 'args' in 'packet' (so that any information from the command line is preserved in the output)
    for key in args:
        if (key not in packet):
            packet[key] = args[key]            

    # Write the response to a JSON file (using pretty print)
    pathOut = args['pathOut']
    filenameOut = pathOut + "generated-" + str(taskNameStr) + "." + engineStr + ".gen" + str(args['genNum']) + ".json"
    print ("Saving generation to " + filenameOut + "...")
    with open(filenameOut, 'w') as f:
        json.dump(packet, f, indent=4)

    # Also save as plain text    
    filenameOut = pathOut + "generated-" + str(taskNameStr) + "." + engineStr + ".gen" + str(args['genNum']) + ".txt"
    print ("Saving generation to " + filenameOut + "...")
    # For each key in the packet, save it to the file
    with open(filenameOut, 'w') as f:
        for key in packet:
            if (key == "responses"):
                f.write("responses:\n")
                for idx, response in enumerate(packet[key]):
                    f.write("----------------------------------------\n")
                    f.write("response " + str(idx) + ":\n")
                    f.write(str(response) + "\n\n")
                    f.write("----------------------------------------\n")
                f.write("\n")
            elif (key == "prompts"):
                f.write("prompts:\n")
                for idx, prompt in enumerate(packet[key]):
                    f.write("----------------------------------------\n")
                    f.write("prompt " + str(idx) + ":\n")
                    f.write(str(prompt) + "\n\n")
                    f.write("----------------------------------------\n")
                f.write("\n")
            else:
                f.write(key + ":\n")
                f.write(str(packet[key]) + "\n\n")


def main(args):        
    # Argument checks:
    print("Command-line Arguments: " + str(args))

    if (args['task'] is None and args['tasks'] is None and args['all'] is False):
        print ("")
        print ("ERROR: Either --task or --tasks or --all must be specified.")
        return
    
    # Create a Words2Wires object
    t2d = Words2Wires(engineStr = args['engine'])

    # Load a TSV file with the task descriptions
    filenameIn = args['filenameIn']
    tasks = loadTaskDescriptions(filenameIn)
    print ("Loaded " + str(len(tasks)) + " tasks.")
    print ("")

    # Case 1: Do all the tasks    
    if (args['all']):
        for taskIdx in range(len(tasks)):
            task = tasks[taskIdx]
            for genNum in range(args['numGenerations']):
                args['genNum'] = genNum
                print("Task " + str(taskIdx+1) + " of " + str(len(tasks)) + " (Generation " + str(genNum+1) + " of " + str(args['numGenerations']) + ")")
                generateForOneTask(t2d, task, args)
    # Case 2: Do one named task
    elif (args['task'] is not None):
        taskName = args['task']
        for task in tasks:
            if (task['taskName'] == taskName):
                for genNum in range(args['numGenerations']):
                    args['genNum'] = genNum
                    print("Generation " + str(genNum+1) + " of " + str(args['numGenerations']))
                    generateForOneTask(t2d, task, args)
                break
        else:
            print ("ERROR: Could not find task (" + taskName + ") in the task file (" + filenameIn + ").")
            return  
    # Case 3: Do a list of tasks
    elif (args['tasks'] is not None):        
        taskNames = args['tasks'].split(',')
        errors = []
        for taskName in taskNames:
            for task in tasks:
                if (task['taskName'] == taskName):
                    for genNum in range(args['numGenerations']):
                        args['genNum'] = genNum
                        print("Generation " + str(genNum+1) + " of " + str(args['numGenerations']))
                        generateForOneTask(t2d, task, args)
                    break
            else:
                errorStr = "ERROR: Could not find task (" + taskName + ") in the task file (" + filenameIn + ")."
                print (errorStr)
                errors.append(errorStr)
        if (len(errors) > 0):
            print("")
            print("ERRORS:")
            for error in errors:
                print(error)
            
                

#
#   Parse command line arguments
#
def parseArgs():
    desc = "Run Words2Wires to generate an electrical device specification (e.g. schematics and code) using a language model."
    parser = argparse.ArgumentParser(desc)
    parser.add_argument("--task", type=str,
                        help="Run one task with a specific name")
    parser.add_argument("--tasks", type=str,
                        help="A comma-delimited list of tasks to run")
    parser.add_argument("--filenameIn", type=str, default="experiment3-opengeneration/words2wires-task-descriptions.tsv",
                        help="A TSV file with the task descriptions. Default: %(default)s")
    parser.add_argument("--pathOut", type=str, default="",
                        help="The output path (where files will be stored). Default: %(default)s")
    parser.add_argument("--all", action='store_true', default=False,
                        help="Run ALL the tasks in the task file. Note, this could be expensive. Default: %(default)s")    
    parser.add_argument("--numGenerations", type=int, default=1,
                        help="The number of generations to do for each task. Default: %(default)s")
    # Add engine (default gpt-4)
    parser.add_argument("--engine", type=str, default="gpt-4",
                        help="The engine to use. Options: gpt-4, claude-v1. Default: %(default)s")
    # Generation temperature (float)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="The generation temperature. Default: %(default)s")

    # Generation mode (ALL_AT_ONCE or PIECEWISE)
    # This feature is in development -- it is not currently recommended to change it. 
    parser.add_argument("--generationMode", type=str, default="ALL_AT_ONCE",
                        help="The generation mode. Options: ALL_AT_ONCE or PIECEWISE. This feature is in development, and it is not currently recommended to change it from the default seeting. Default: %(default)s")

    args = parser.parse_args()
    params = vars(args)
    return params

#
#   Main
#
if __name__ == '__main__':
    print("Words2Wires: GenerateDevice")
    # Parse command line arguments
    args = parseArgs()
    #print(args)
    #exit(1)
    main(args)