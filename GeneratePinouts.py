# EvaluatePinouts.py

import time
import json
import argparse

from Words2Wires import engineGPT4, engineGPT35Turbo, engineClaudeV1


# Load the device names, from a TSV file. 
# The first column is the device name, the second column (optional) is the device description, the third column is the pinout.
# The data should be stored in a list of dictionaries. 
def loadDeviceNames(filenameIn:str):
    devices = []

    with open(filenameIn, 'r') as f:
        print ("* Loading device names from " + filenameIn + "...")
        for line in f:
            line = line.strip()
            if (line == ''):
                continue
            if (line[0] == '#'):
                continue
            fields = line.split('\t')

            deviceName = fields[0]            
            deviceDescription = fields[1]
            devicePinout = fields[2]

            packed = {
                'name': deviceName,
                'description': deviceDescription,
                'pinout': devicePinout
            }
            devices.append(packed)            

        print("* Found " + str(len(devices)) + " device names.")

    return devices




def doPinoutGeneration(deviceName:str, engine, temperature=0.0, maxTokensOut = 1000):

    # Step 1: Create prompt
    prompt = f"""Your task is to generate a description and pinout for an electronic component. 
The specific electronic component to generate this output for is: {deviceName}
The output format is JSON, between code blocks, as shown in the example below:
```
{{
    "7479": {{
        "description": "Dual D positive-edge triggered flip flop, asynchronous preset and clear", 
        "pinout:"["#R1", "D1", "CLK1", "#PR1", "Q1", "#Q1", "VSS", "#Q2", "Q2", "#PR2", "CLK2", "D2", "#R2", "VDD"]
    }}
}}
```
"""

    # Step 2: Generate the output    
    response, success, numTokensSent, numTokensReceived = engine.getResponse(prompt, temperature, maxTokensOut)        

    # Step 3: Pack the output
    packed = {
        'deviceName': deviceName,
        'prompt': prompt,
        'response': response,
        'success': success,
        'numTokensSent': numTokensSent,
        'numTokensReceived': numTokensReceived
    }

    return packed

#
#   Parse command line arguments
#
def parseArgs():
    desc = "Run Words2Wires"
    parser = argparse.ArgumentParser(desc)
    # Add engine (default gpt-4)
    parser.add_argument("--engine", type=str, default="gpt-3.5-turbo",
                        help="The engine to use. Options: gpt-3.5-turbo, gpt-4, claude-v1. Default: %(default)s")

    args = parser.parse_args()
    params = vars(args)
    return params


#
#   Main
#

def main(args):
    engineStr = args['engine']

    filenameIn = "experiment1-pinouts/pins100-benchmark.tsv"
    filenameOut = "generated.pinouts." + engineStr + ".json"

    # Load the device names
    devices = loadDeviceNames(filenameIn)

    # Instantiate the GPT4 engine
    engine = None
    if (engineStr == "gpt-4"):
        engine = engineGPT4()
    elif (engineStr == "gpt-3.5-turbo"):
        engine = engineGPT35Turbo()
    elif (engineStr == "claude-v1"):
        engine = engineClaudeV1()    
    else:
        print("Error: Unknown engine " + engineStr)
        return

    print("Using engine: " + engineStr + " (" + engine.name + ")")
    print("")
    time.sleep(2)

    responses = []    

    # For each device, generate a pinout
    for idx, device in enumerate(devices):
        deviceName = device['name']
        deviceDescription = device['description']

        print(f"Generating pinout for device {idx+1} of {len(devices)}: {deviceName}")

        packedResponse = doPinoutGeneration(deviceName, engine)
        responses.append(packedResponse)

        # Add any keys from the device dictionary to the response dictionary
        for key in device:
            if (key not in packedResponse):
                packedResponse[key] = device[key]                

        responseStr = packedResponse['response']
        print(responseStr)
        print("")

        packedOut = {
            "engine:": engine.name,
            "responses": responses
        }

        # Save the responses to a JSON file (pretty print)
        with open(filenameOut, 'w') as f:
            json.dump(packedOut, f, indent=4)
        
    print("Completed.")



if __name__ == "__main__":
    # Parse command line arguments
    args = parseArgs()
    print(args) 
    main(args)
