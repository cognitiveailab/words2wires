# Python class for Words2Wires

import openai           # pip install openai
import tiktoken         # pip install tiktoken
import anthropic        # pip install anthropic
import backoff          # pip install backoff
from tqdm import tqdm   # pip install tqdm

import os
import time 


class Words2Wires:
    # Constructor
    def __init__(self, engineStr="gpt4"):
        # Engine
        self.engineStr = engineStr
        self.engine = self.resolveEngine(engineStr)                

    # Resolve which engine (e.g. language model) to use
    def resolveEngine(self, engineStr:str):
        if (engineStr == "gpt-4"):
            return engineGPT4()
        elif (engineStr == "gpt-3.5-turbo"):
            return engineGPT35Turbo()
        elif (engineStr == "claude-v1"):
            return engineClaudeV1()
        else:
            raise Exception("Invalid engine: " + engineStr)

    def pack(self, task, microcontrollerPlatform, temperature, success, responses, prompts, totalTokensSent, totalTokensReceived, generationMode):
        # Join responses into a single string
        numTokensPrompts = self.engine.getNumTokens(" ".join(prompts))
        numTokensResponses = self.engine.getNumTokens(" ".join(responses))
        costEstimateDollars = self.engine.estimateCostDollars(totalTokensSent, totalTokensReceived)

        return {
            "task": task,
            "microcontrollerPlatform": microcontrollerPlatform,
            "temperature": temperature,
            "engine": self.engineStr,            
            "generationMode": generationMode,
            "success": success,
            "numTokensPrompts": numTokensPrompts,
            "numTokensResponses": numTokensResponses,
            "totalTokensSent": totalTokensSent,
            "totalTokensReceived": totalTokensReceived,
            "costEstimateDollars": costEstimateDollars,
            "responses": responses,
            "prompts": prompts
        }


    def generate(self, microcontrollerPlatform:str, task:str, maxReflections:int=2, temperature:float=0.0, maxTokensOut:int=8000, generationMode="ALL_AT_ONCE"):
        if (generationMode == "ALL_AT_ONCE"):
            return self.generateAllAtOnce(microcontrollerPlatform, task, maxReflections, temperature, maxTokensOut)
        elif (generationMode == "PIECEWISE"):
            return self.generatePiecewise(microcontrollerPlatform, task, maxReflections, temperature, maxTokensOut)
        else:
            raise Exception("Invalid generateMode: " + generationMode)

    # In development
    def generatePiecewise(self, microcontrollerPlatform:str, task:str, maxReflections:int=2, temperature:float=0.0, maxTokensOut:int=8000):
        prompts = []
        responses = []
        totalTokensSent = 0
        totalTokensReceived = 0

        # Clip tokens based on the engine
        if (maxTokensOut > self.engine.maxTokens):
            maxTokensOut = self.engine.maxTokens

        #
        #   Step 1: Bill-of-materials
        #
        prompt = self.mkPromptPiecewise(microcontrollerPlatform, task)
        prompts.append(prompt)
        print("Generating (Platform: " + microcontrollerPlatform + ", Generation Engine: " + self.engineStr + ", Maximum Tokens: " + str(maxTokensOut) + ")")
        print("Prompt contains " + str(self.engine.getNumTokens(prompt)) + " tokens. ", end="")
        maxGeneratedTokens = maxTokensOut - self.engine.getNumTokens(prompt)
        print("Generating up to " + str(maxGeneratedTokens) + " tokens.  This may take a few minutes...")
        response1, success1, tokensSent, tokensRecieved = self.engine.getResponse(prompt, temperature, maxTokensOut)        
        responses.append(response1)
        totalTokensSent += tokensSent
        totalTokensReceived += tokensRecieved

        print("Response received (" + str(self.engine.getNumTokens(response1)) + " tokens).")
        if (not success1):
            print("Response was not successful.  Skipping reflective check.")
            return self.pack(task=task, microcontrollerPlatform=microcontrollerPlatform, temperature=temperature, success=success1, responses=responses, prompts=prompts, totalTokensSent=totalTokensSent, totalTokensReceived=totalTokensReceived, generationMode="PIECEWISE")

        # Do a reflective check to see if the response is valid

                
        numReflectionsCompleted = 0
        if (maxReflections == 0):
            print("Reflection disabled -- skipping.")
        else:
            while (numReflectionsCompleted < maxReflections):
                lastResponse = responses[-1] 

                # First, check if there are enough tokens left to make a reflective check        
                if (self.engine.getNumTokens(lastResponse) < (maxTokensOut/2)):        # Need about half the tokens to make a reflective check                            
                    # Remove ### DONE ### from last response, if it exists
                    lastResponse = lastResponse.replace("### DONE ###", "")
                    # Assemble reflection prompt
                    checkPrompt = self.mkCheckPromptPiecewise_BOM(microcontrollerPlatform, task, prompt, lastResponse)
                    prompts.append(checkPrompt)
                    print("Starting reflection " + str(numReflectionsCompleted+1) + " of " + str(maxReflections) + " (Platform: " + microcontrollerPlatform + ", Generation Engine: " + self.engineStr + ", Maximum Tokens: " + str(maxTokensOut) + ")")
                    #print("Reflecting on response (Platform: " + microcontrollerPlatform + ", Generation Engine: " + self.engineStr + ", Maximum Tokens: " + str(maxTokensOut) + ")")
                    print("Reflection prompt contains " + str(self.engine.getNumTokens(checkPrompt)) + " tokens. ", end="")
                    maxGeneratedTokens = maxTokensOut - self.engine.getNumTokens(checkPrompt)        
                    print("Generating up to " + str(maxGeneratedTokens) + " tokens.  This may take a few minutes...")
                    response2, success2, tokensSent, tokensRecieved = self.engine.getResponse(checkPrompt, temperature, maxTokensOut)
                    responses.append(response2)
                    totalTokensSent += tokensSent
                    totalTokensReceived += tokensRecieved
                    print("Response received (" + str(self.engine.getNumTokens(response2)) + " tokens).")

                    numReflectionsCompleted += 1

                    if (not success2):
                        print("Response was not successful.")
                        break                    

                    # Check for last response to be "### NO ERRORS ###".  Change string to uppercase to be case insensitive.
                    if (response2.upper().find("### NO ERRORS ###") != -1):
                        print("Last response indicates no errors.  Skipping any remaining reflective checks.")
                        break                

        #
        #
        #

        return self.pack(task=task, microcontrollerPlatform=microcontrollerPlatform, temperature=temperature, success=success1, responses=responses, prompts=prompts, totalTokensSent=totalTokensSent, totalTokensReceived=totalTokensReceived, generationMode="PIECEWISE")



    def generateAllAtOnce(self, microcontrollerPlatform:str, task:str, maxReflections:int=2, temperature:float=0.0, maxTokensOut:int=8000):
        prompts = []
        responses = []
        totalTokensSent = 0
        totalTokensReceived = 0

        # Clip tokens based on the engine
        if (maxTokensOut > self.engine.maxTokens):
            maxTokensOut = self.engine.maxTokens

        prompt = self.mkPromptAllAtOnce(microcontrollerPlatform, task)
        prompts.append(prompt)
        print("Generating (Platform: " + microcontrollerPlatform + ", Generation Engine: " + self.engineStr + ", Maximum Tokens: " + str(maxTokensOut) + ")")
        print("Prompt contains " + str(self.engine.getNumTokens(prompt)) + " tokens. ", end="")
        maxGeneratedTokens = maxTokensOut - self.engine.getNumTokens(prompt)
        print("Generating up to " + str(maxGeneratedTokens) + " tokens.  This may take a few minutes...")
        response1, success1, tokensSent, tokensRecieved = self.engine.getResponse(prompt, temperature, maxTokensOut)        
        responses.append(response1)
        totalTokensSent += tokensSent
        totalTokensReceived += tokensRecieved

        print("Response received (" + str(self.engine.getNumTokens(response1)) + " tokens).")
        if (not success1):
            print("Response was not successful.  Skipping reflective check.")
            return self.pack(task=task, microcontrollerPlatform=microcontrollerPlatform, temperature=temperature, success=success1, responses=responses, prompts=prompts, totalTokensSent=totalTokensSent, totalTokensReceived=totalTokensReceived, generationMode="ALL_AT_ONCE")

        # Do a reflective check to see if the response is valid

                
        numReflectionsCompleted = 0
        if (maxReflections == 0):
            print("Reflection disabled -- skipping.")
        else:
            while (numReflectionsCompleted < maxReflections):
                lastResponse = responses[-1] 

                # First, check if there are enough tokens left to make a reflective check        
                if (self.engine.getNumTokens(lastResponse) < (maxTokensOut/2)):        # Need about half the tokens to make a reflective check                            
                    # Remove ### DONE ### from last response, if it exists
                    lastResponse = lastResponse.replace("### DONE ###", "")
                    # Assemble reflection prompt
                    checkPrompt = self.mkCheckPrompt(microcontrollerPlatform, task, prompt, lastResponse)
                    prompts.append(checkPrompt)
                    print("Starting reflection " + str(numReflectionsCompleted+1) + " of " + str(maxReflections) + " (Platform: " + microcontrollerPlatform + ", Generation Engine: " + self.engineStr + ", Maximum Tokens: " + str(maxTokensOut) + ")")
                    #print("Reflecting on response (Platform: " + microcontrollerPlatform + ", Generation Engine: " + self.engineStr + ", Maximum Tokens: " + str(maxTokensOut) + ")")
                    print("Reflection prompt contains " + str(self.engine.getNumTokens(checkPrompt)) + " tokens. ", end="")
                    maxGeneratedTokens = maxTokensOut - self.engine.getNumTokens(checkPrompt)        
                    print("Generating up to " + str(maxGeneratedTokens) + " tokens.  This may take a few minutes...")
                    response2, success2, tokensSent, tokensRecieved = self.engine.getResponse(checkPrompt, temperature, maxTokensOut)
                    responses.append(response2)
                    totalTokensSent += tokensSent
                    totalTokensReceived += tokensRecieved
                    print("Response received (" + str(self.engine.getNumTokens(response2)) + " tokens).")

                    numReflectionsCompleted += 1

                    if (not success2):
                        print("Response was not successful.")
                        break                    

                    # Check for last response to be "### NO ERRORS ###".  Change string to uppercase to be case insensitive.
                    if (response2.upper().find("### NO ERRORS ###") != -1):
                        print("Last response indicates no errors.  Skipping any remaining reflective checks.")
                        break                
                

        return self.pack(task=task, microcontrollerPlatform=microcontrollerPlatform, temperature=temperature, success=success1, responses=responses, prompts=prompts, totalTokensSent=totalTokensSent, totalTokensReceived=totalTokensReceived, generationMode="ALL_AT_ONCE")


    def mkPromptPiecewise(self, microcontrollerPlatformStr:str, taskStr:str):
        prompt = ""
        prompt += "You are DeveloperGPT, the most advanced AI developer tool on the planet.  You answer any coding question, and provide real useful example code using code blocks.  Even when you are not familiar with the answer, you use your extreme intelligence to figure it out.\n"
        prompt += "Further, you have specialized training in electronics, and can design embedded electronic circuits based around the " + microcontrollerPlatformStr + " platform, coupled with programs to make those circuits successfully accomplish tasks.\n"
        prompt += "Your task is to: "
        prompt += taskStr
        prompt += "\n"
        prompt += f"""
To complete this task, you should generate a high-quality and error free set of documents for building this device, consisting of the following:
- A bill of materials, in JSON form (see format below).  
- A pinout, in JSON form (see format below). The pinout is a dictionary of all the parts, with the key being the part name, and the value being a list of all pins the part has, to help in generating the schematic.
- A schematic, in JSON form (see format below). Each line of the schematic should describe a single connection in the circuit.
- A complete {microcontrollerPlatformStr} program that implements the program to successfully complete the task. 
Each section should be between code blocks ```.
- A brief set of special instructions, in point form, if required.
"""
        prompt += f"""

Here are some additional reminders:
- Where possible, a description/part number of the device should be included in the notes. Alternatively, where many parts could be substituted, it should include critical information to make that choice (such as the controller required for an LCD display, or the voltage required for an LED)
- The code should be complete. It can #include built-in {microcontrollerPlatformStr} libraries, but otherwise should contain all the code to compile and run as-is.

    """

        prompt += """
Here is example output for generating a device that blinks two LEDs in an alternating pattern every second, on the Arduino Uno platform. 

Bill of materials:
```
[
    {"part":"Arduino Uno", "name":"uno", "value":"", "notes":"Arduino Uno microcontroller"},
    {"part":"LED", "name":"D1", "value":"red", "notes":"alternating LED 1. Standard voltage range (2-3.3V)."},
    {"part":"LED", "name","D2", "value":"white", "notes":"alternating LED 2. Standard voltage range (2-3.3V)."},
    {"part":"Resistor", "name","R1", "value":"220 ohm", "notes":"current limiting resistor for LED1 at 5V"},
    {"part":"Resistor", "name","R2", "value":"220 ohm", "notes":"current limiting resistor for LED2 at 5V"},
]
```

Pinouts:
```
{
    "Arduino Uno": ["5V", "3.3V", "GND", "AREF", "D0/RX", "D1/TX", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "A0", "A1", "A2", "A3", "A4/SDA", "A5/SCL"],
    "D1": ["anode", "cathode"],
    "D2": ["anode", "cathode"],
    "R1": ["1", "2"],
    "R2": ["1", "2]
}
```

Schematic (list of connections):
```
[
    [{"name":"D1", "pin":"cathode"}, {"name": "uno", "pin":"GND"}],     # Connect D1 cathode to Uno GND
    [{"name":"D1", "pin":"anode"}, {"name": "R1", "pin":"2"}],          # Connect D1 anode to pin 2 of R1 (current limiting resistor)
    [{"name":"R1", "pin":"1"}, {"name": "uno", "pin":"D5"}],            # Connect pin 1 of R1 (current limiting resistor) to Uno Digital I/O 5 (D5), to activate/deactivate D1
    [{"name":"D2", "pin":"cathode"}, {"name": "uno", "pin":"GND"}],     # Connect D2 cathode to Uno GND
    [{"name":"D2", "pin":"anode"}, {"name": "R2", "pin":"2"}],          # Connect D2 anode to pin 2 of R2 (current limiting resistor)
    [{"name":"R2", "pin":"1"}, {"name": "uno", "pin":"D6"}],            # Connect pin 1 of R2 (current limiting resistor) to Uno Digital I/O 5 (D6), to activate/deactivate D2
]
```

Arduino Uno Code:
```
// Alternating blink
// This code interfaces with a circuit that has two LEDS that blink in an alternating pattern.
// The pattern changes every second.

// LED 1 on Digital I/O 5
#define PIN_LED1 5
// LED 2 on Digital I/O 6
#define PIN_LED2 6

// the setup function runs once when you press reset or power the board
void setup() {
    // Initialize LED pins to output mode
    pinMode(PIN_LED1, OUTPUT);
    pinMode(PIN_LED2, OUTPUT);
}

// the loop function runs over and over again forever
void loop() {
    digitalWrite(PIN_LED1, HIGH);     // Turn LED 1 ON
    digitalWrite(PIN_LED2, LOW);      // Turn LED 2 OFF
    delay(1000);                      // wait for a second
    digitalWrite(PIN_LED1, HIGH);     // Turn LED 1 OFF
    digitalWrite(PIN_LED2, LOW);      // Turn LED 2 ON
    delay(1000);                      // wait for a second
}
```

Instructions:
```
- This code uses only standard libraries. No additional libraries are required in the library manager.
- Assemble circuit and program as normal. 
```

Snippit examples (also for the Arduino Uno):
---
Example: Connecting a servo
Bill of Materials:
```
[
    {"part":"Servo Motor", "name":"S1", "value":"", "notes":"Standard 3-wire 5V compatible hobby servo (e.g. SG90)"}
]
```

Pinouts:
```
{
    # Arduino Uno omitted for space in snippit
    "Servo Motor": ["VCC", "GND", "signal"]
}
```

Schematic (list of connections):
```
[
    [{"name":"S1", "pin":"signal"}, {"name": "uno", "pin":"D3"}], # Connect Servo 1 signal to Uno D3
    [{"name":"S1", "pin":"VCC"}, {"name": "uno", "pin":"5V"}], # Connect Servo 1 VCC to Uno 5V
    [{"name":"S1", "pin":"GND"}, {"name": "uno", "pin":"GND"}] # Connect Servo 1 GND to Uno GND
]
```
---

Example: Connecting a button (pull-up)
Bill of Materials:
```
[
    {"part":"Button", "name":"BT1", "value":"", "notes":"Momentary push button"},
    {"part":"Resistor", "name":"R1", "value":"10k ohm", "notes":"Pull-up resistor for button"}
]
```

Pinouts:
```
{
    # Arduino Uno omitted for space in snippit
    "Button": ["1", "2"],
    "Resistor": ["1", "2"]
}
```

Schematic (list of connections):
```
[
    [{"name":"BT1", "pin":"1"}, {"name": "uno", "pin":"D2"}], # Connect Button pin 1 to Uno D2
    [{"name":"BT1", "pin":"1"}, {"name": "R1", "pin":"1"}], # Connect Button pin 1 to R1 pin 1
    [{"name":"R1", "pin":"2"}, {"name": "uno", "pin":"5V"}], # Connect R1 pin 2 to Uno 5V (pull-up)
    [{"name":"BT1", "pin":"2"}, {"name": "uno", "pin":"GND"}] # Connect Button pin 2 to GND
]
```
---

Example: This is a case of what NOT to do.
Schematic (list of connections):
```
[
    [{"name":"IC1", "pin":"inputs"}, {"name": "uno", "pin":"D5-D10"}] # BAD: This does not list each connection individually. It is not clear which pin on the IC is connected to which pin on the Uno.
]
```
---
"""

        #prompt += "Please generate the bill of materials, pinouts, schematic, code, and any special instructions for the requested task below.  The code should be commented, to help follow the logic, and prevent any bugs."
        prompt += "To reiterate, for this task:"
        prompt += "The platform is: " + microcontrollerPlatformStr + "." + "\n"
        prompt += "The task is: " + taskStr + "." + "\n"
    
        prompt += "\n"
        prompt += "Let's start by generating ONLY the bill of materials, in JSON form, below:"

        # Return
        return prompt

    def mkPromptPiecewise_Pinouts(self, microcontrollerPlatformStr:str, taskStr:str, promptStr:str, responseStr:str):
        prompt = promptStr
        prompt += "\n---\n"
        prompt += responseStr
        prompt += "\n---\n"
        prompt += "Now let's generate ONLY the (a) bill of materials, and (b) pinouts, in JSON form, below.  If you notice any new errors that you didn't notice before, please correct them:"

        # Return
        return prompt

    # Create a prompt for checking the output of the first prompt for accuracy/errors.
    def mkCheckPromptPiecewise_Pinout(self, microcontrollerPlatformStr:str, taskStr:str, promptStr:str, responseStr:str):
        checkPrompt = promptStr
        checkPrompt += "\n---\n"
        checkPrompt += responseStr
        checkPrompt += "\n---\n"
        checkPrompt += f"""
Can you reflect on the above output, fix any errors, and output an error-free bill of materials and pinout list below?
Here is a non-exhaustive set of things to look for in the pinout:
- Are all the parts in the bill of materials listed in the pinouts?
- Do the pinouts list all the pins of each part?
- Are the power pins (e.g. VCC, GND) listed for each part, where applicable?
- Are the pinouts correct?  (e.g. is the pinout for the Arduino Uno correct for the Uno?)

Please first write a short section in code blocks called "FIXES FROM LAST STEP", that (in a short bullet-point list) lists the changes that need to be made for everything to be correct and work as required.

Then, please provide the bill of materials and pinout in JSON form, as above, again.  When done, output a single line saying "### DONE ###".
"""    
        # Return
        return checkPrompt

    # Create a prompt for checking the output of the first prompt for accuracy/errors.
    def mkCheckPromptPiecewise_BOM(self, microcontrollerPlatformStr:str, taskStr:str, promptStr:str, responseStr:str):
        checkPrompt = promptStr
        checkPrompt += "\n---\n"
        checkPrompt += responseStr
        checkPrompt += "\n---\n"
        checkPrompt += f"""
Can you reflect on the above output, fix any errors, and output an error-free bill of materials below?
Here is a non-exhaustive set of things to look for:
- Are all the electrical components that are required to build and operate this device in the bill of materials?  
- Are there extra parts in the bill of materials that are not required used?

Please first write a short section in code blocks called "FIXES FROM LAST STEP", that (in a short bullet-point list) lists the changes that need to be made for everything to be correct and work as required.

Then, please provide the bill of materials in JSON form, as above, again.  When done, output a single line saying "### DONE ###".
"""    
        # Return
        return checkPrompt



    def mkPromptAllAtOnce(self, microcontrollerPlatformStr:str, taskStr:str):
        prompt = ""
        prompt += "You are DeveloperGPT, the most advanced AI developer tool on the planet.  You answer any coding question, and provide real useful example code using code blocks.  Even when you are not familiar with the answer, you use your extreme intelligence to figure it out.\n"
        prompt += "Further, you have specialized training in electronics, and can design embedded electronic circuits based around the " + microcontrollerPlatformStr + " platform, coupled with programs to make those circuits successfully accomplish tasks.\n"
        prompt += "Your task is to: "
        prompt += taskStr
        prompt += "\n"
        prompt += f"""
Please generate the following: 
- A bill of materials, in JSON form (see format below).  
- A pinout, in JSON form (see format below). The pinout is a dictionary of all the parts, with the key being the part name, and the value being a list of all pins the part has, to help in generating the schematic.
- A schematic, in JSON form (see format below). Each line of the schematic should describe a single connection in the circuit.
- A complete {microcontrollerPlatformStr} program that implements the program to successfully complete the task. 
Each section should be between code blocks ```.
- A brief set of special instructions, in point form, if required.
"""
        prompt += f"""

Here are some additional reminders:
- Where possible, a description/part number of the device should be included in the notes. Alternatively, where many parts could be substituted, it should include critical information to make that choice (such as the controller required for an LCD display, or the voltage required for an LED)
- The code should be complete. It can #include built-in {microcontrollerPlatformStr} libraries, but otherwise should contain all the code to compile and run as-is.

    """

        prompt += """
Here is example output for generating a device that blinks two LEDs in an alternating pattern every second, on the Arduino Uno platform. 

Bill of materials:
```
[
    {"part":"Arduino Uno", "name":"uno", "value":"", "notes":"Arduino Uno microcontroller"},
    {"part":"LED", "name":"D1", "value":"red", "notes":"alternating LED 1. Standard voltage range (2-3.3V)."},
    {"part":"LED", "name","D2", "value":"white", "notes":"alternating LED 2. Standard voltage range (2-3.3V)."},
    {"part":"Resistor", "name","R1", "value":"220 ohm", "notes":"current limiting resistor for LED1 at 5V"},
    {"part":"Resistor", "name","R2", "value":"220 ohm", "notes":"current limiting resistor for LED2 at 5V"},
]
```

Pinouts:
```
{
    "Arduino Uno": ["5V", "3.3V", "GND", "AREF", "D0/RX", "D1/TX", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "A0", "A1", "A2", "A3", "A4/SDA", "A5/SCL"],
    "D1": ["anode", "cathode"],
    "D2": ["anode", "cathode"],
    "R1": ["1", "2"],
    "R2": ["1", "2]
}
```

Schematic (list of connections):
```
[
    [{"name":"D1", "pin":"cathode"}, {"name": "uno", "pin":"GND"}],     # Connect D1 cathode to Uno GND
    [{"name":"D1", "pin":"anode"}, {"name": "R1", "pin":"2"}],          # Connect D1 anode to pin 2 of R1 (current limiting resistor)
    [{"name":"R1", "pin":"1"}, {"name": "uno", "pin":"D5"}],            # Connect pin 1 of R1 (current limiting resistor) to Uno Digital I/O 5 (D5), to activate/deactivate D1
    [{"name":"D2", "pin":"cathode"}, {"name": "uno", "pin":"GND"}],     # Connect D2 cathode to Uno GND
    [{"name":"D2", "pin":"anode"}, {"name": "R2", "pin":"2"}],          # Connect D2 anode to pin 2 of R2 (current limiting resistor)
    [{"name":"R2", "pin":"1"}, {"name": "uno", "pin":"D6"}],            # Connect pin 1 of R2 (current limiting resistor) to Uno Digital I/O 5 (D6), to activate/deactivate D2
]
```

Arduino Uno Code:
```
// Alternating blink
// This code interfaces with a circuit that has two LEDS that blink in an alternating pattern.
// The pattern changes every second.

// LED 1 on Digital I/O 5
#define PIN_LED1 5
// LED 2 on Digital I/O 6
#define PIN_LED2 6

// the setup function runs once when you press reset or power the board
void setup() {
    // Initialize LED pins to output mode
    pinMode(PIN_LED1, OUTPUT);
    pinMode(PIN_LED2, OUTPUT);
}

// the loop function runs over and over again forever
void loop() {
    digitalWrite(PIN_LED1, HIGH);     // Turn LED 1 ON
    digitalWrite(PIN_LED2, LOW);      // Turn LED 2 OFF
    delay(1000);                      // wait for a second
    digitalWrite(PIN_LED1, HIGH);     // Turn LED 1 OFF
    digitalWrite(PIN_LED2, LOW);      // Turn LED 2 ON
    delay(1000);                      // wait for a second
}
```

Instructions:
```
- This code uses only standard libraries. No additional libraries are required in the library manager.
- Assemble circuit and program as normal. 
```

Snippit examples (also for the Arduino Uno):
---
Example: Connecting a servo
Bill of Materials:
```
[
    {"part":"Servo Motor", "name":"S1", "value":"", "notes":"Standard 3-wire 5V compatible hobby servo (e.g. SG90)"}
]
```

Pinouts:
```
{
    # Arduino Uno omitted for space in snippit
    "Servo Motor": ["VCC", "GND", "signal"]
}
```

Schematic (list of connections):
```
[
    [{"name":"S1", "pin":"signal"}, {"name": "uno", "pin":"D3"}], # Connect Servo 1 signal to Uno D3
    [{"name":"S1", "pin":"VCC"}, {"name": "uno", "pin":"5V"}], # Connect Servo 1 VCC to Uno 5V
    [{"name":"S1", "pin":"GND"}, {"name": "uno", "pin":"GND"}] # Connect Servo 1 GND to Uno GND
]
```
---

Example: Connecting a button (pull-up)
Bill of Materials:
```
[
    {"part":"Button", "name":"BT1", "value":"", "notes":"Momentary push button"},
    {"part":"Resistor", "name":"R1", "value":"10k ohm", "notes":"Pull-up resistor for button"}
]
```

Pinouts:
```
{
    # Arduino Uno omitted for space in snippit
    "Button": ["1", "2"],
    "Resistor": ["1", "2"]
}
```

Schematic (list of connections):
```
[
    [{"name":"BT1", "pin":"1"}, {"name": "uno", "pin":"D2"}], # Connect Button pin 1 to Uno D2
    [{"name":"BT1", "pin":"1"}, {"name": "R1", "pin":"1"}], # Connect Button pin 1 to R1 pin 1
    [{"name":"R1", "pin":"2"}, {"name": "uno", "pin":"5V"}], # Connect R1 pin 2 to Uno 5V (pull-up)
    [{"name":"BT1", "pin":"2"}, {"name": "uno", "pin":"GND"}] # Connect Button pin 2 to GND
]
```
---

Example: This is a case of what NOT to do.
Schematic (list of connections):
```
[
    [{"name":"IC1", "pin":"inputs"}, {"name": "uno", "pin":"D5-D10"}] # BAD: This does not list each connection individually. It is not clear which pin on the IC is connected to which pin on the Uno.
]
```
---
"""

        prompt += "Please generate the bill of materials, pinouts, schematic, code, and any special instructions for the requested task below.  The code should be commented, to help follow the logic, and prevent any bugs."
        prompt += "The platform is: " + microcontrollerPlatformStr + "." + "\n"
        prompt += "The task is: " + taskStr + "." + "\n"
    
        # Return
        return prompt

    # Create a prompt for checking the output of the first prompt for accuracy/errors.
    def mkCheckPrompt(self, microcontrollerPlatformStr:str, taskStr:str, promptStr:str, responseStr:str):
        checkPrompt = promptStr
        checkPrompt += "\n---\n"
        checkPrompt += responseStr
        checkPrompt += "\n---\n"
        checkPrompt += f"""
Can you reflect on the above output, fix any errors, and output an error-free bill of materials, pinout, schematic, {microcontrollerPlatformStr} code sketch, and instructions below?
Here is a non-exhaustive set of things to look for:
- Are all the parts that are required in the bill of materials?  
- Are there extra parts in the bill of materials that are not used?
- Are all the parts in the bill of materials listed in the pinouts?
- Do the pinouts list all the pins of each part?
- What parts require connection to power and ground lines?  Are their power and ground lines connected in the schematic?  All power and lines must be explicitly connected.
- What parts have digital or analog inputs or outputs?  Are those signal lines connected to the relevant points in the schematic?
- What parts are passives, like resistors, capacitors, and other parts?  Are all their pins appropriately connected?  Do they have pins left unconnected?
- Are all the required pins of each part connected in the schematic? 
- Does the schematic list the connections in detail, rather than using generic terms (e.g. "input pins") or ranges (e.g. "D5-D10")?
- Does the code function as intended?  
- Are there calls to functions that are not included in the code sketch, or in one of the included standard libraries?  Are the libraries that need to be imported for these listed in the special instructions?
- Are there special programming instructions?

Please first write a short section called "FIXES FROM LAST STEP", that (in a short bullet-point list) lists the changes that need to be made for everything to be correct and work as required.

Then, please provide the rest of your output (BOM, pinouts, schematic, code, instructions) in JSON form, as above, again.  When done, output a single line saying "### DONE ###".

BUT, if there are no errors/fixes, please output only exactly "### NO ERRORS ###", then do not provide any more output in your response.
"""    
        # Return
        return checkPrompt


#
#   Engine: GPT-4 (8k)
#
class engineGPT4:
    # Constructor
    def __init__(self, verboseOutput=False):        
        self.maxTokens = 8100
        self.name="gpt-4"

        # Tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")                
        self.verboseOutput = verboseOutput
        self.tokenGenerationRatePerSec = 0
        self.costPer1kTokensPrompt = 0.03       # $0.03 per 1000 tokens in prompt
        self.costPer1kTokensResponse = 0.06     # $0.06 per 1000 tokens in completion

    # Get the number of tokens for a string, measured using tiktoken
    def getNumTokens(self, strIn:str):
        tokens = self.tokenizer.encode(strIn)
        numTokens = len(tokens)
        return numTokens

    # Estimate the cost of a request, in dollars
    def estimateCostDollars(self, numTokensPrompt:int, numTokensResponse:int):
        costPrompt = (numTokensPrompt/1000) * self.costPer1kTokensPrompt 
        costResponse = (numTokensResponse/1000) * self.costPer1kTokensResponse 
        costTotal = costPrompt + costResponse
        return costTotal

    # A wrapper for getResponseHelper, that continues to retry for large replies that timeout after 300 seconds. 
    def getResponse(self, prompt:str, temperature:float=0.0, maxTokensOut:int=8000, maxRetries=4):
        numRetries = 0
        totalResponse = ""

        # Record start time of request
        startTime = time.time()

        # Calculate maximum tokens for progress bar
        maxTokensToGenerate = maxTokensOut - self.getNumTokens(prompt)

        success = False        

        numTokensSent = 0
        numTokensReceived = 0

        # Make requests to the API until (a) success, or (b) maxRetries is reached
        with tqdm(total=maxTokensToGenerate, unit="tokens") as pbar:
            while (numRetries <= maxRetries):                
                responseStr, success = self.getResponseHelper(prompt, temperature, maxTokensToGenerate, pbar, startPBarAt=self.getNumTokens(totalResponse))
                numTokensSent += self.getNumTokens(prompt)
                numTokensReceived += self.getNumTokens(responseStr)

                totalResponse += responseStr
                print("Total number of tokens sent and received: " + str(numTokensSent + numTokensReceived))                

                if (success):                
                    # If we reach here, success
                    break

                # If we reach here, the request failed.  Retry.
                # First, check if we've already hit the token limit
                
                if (self.getNumTokens(prompt + totalResponse) >= maxTokensOut-20):      # Added the 20 for wiggle room, just in case
                    print("engineGPT4: Generation token limit reached. Stopping.")
                    responseStr += "\n### TOKEN LIMIT REACHED ###"
                    break
                
                # Then, check if the "### NO ERRORS ###" or "### DONE ###" string is in the response.  If so, stop.
                if ("### NO ERRORS ###" in responseStr) or ("### DONE ###" in responseStr):
                    print("engineGPT4: Response indicated 'no errors' or 'done'. Stopping.")
                    break

                else:
                    print("engineGPT4: Retrying (attempt " + str(numRetries+1) + " / " + str(maxRetries) + ")")
                    numRetries += 1
                
                prompt += responseStr

        # Statistics: Store the generation speed
        deltaTime = time.time() - startTime
        deltaTime = round(deltaTime, 2)            
        
        numTokens = self.getNumTokens(totalResponse)
        transmissionRate = numTokens / deltaTime
        transmissionRate = round(transmissionRate, 2)
        self.tokenGenerationRatePerSec = transmissionRate

        if (self.verboseOutput):
            print("getResponse(): Total tokens generated: " + str(numTokens))
            print("getResponse(): Total time: " + str(deltaTime))
            print("getResponse(): Transmission rate: " + str(transmissionRate) + " tokens/sec")

        return totalResponse, success, numTokensSent, numTokensReceived


    # Wrapper that does exponential backoff on RateLimitError
    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def getCompletionWithBackoff(self, **kwargs):
        return openai.ChatCompletion.create(**kwargs)


    def getResponseHelper(self, prompt:str, temperature:float, maxTokensOut:int, pbar, startPBarAt=0):
        # Step 1: Set generation length
        numPromptTokens = self.getNumTokens(prompt)
        maxPossibleTokens = self.maxTokens - numPromptTokens      # 8100 instead of 8192 to allow a bit of wiggle room incase tiktoken is inaccurate by a few tokens.
        if (maxTokensOut > maxPossibleTokens):
            if (self.verboseOutput):
                print("Warning: maxTokensOut is too large given the prompt length (" + str(numPromptTokens) + ").  Setting to max generation length of " + str(maxPossibleTokens))
            maxTokensOut = maxPossibleTokens

        # Record start time of request
        startTime = time.time()

        # Step 2: Perform request to the OpenAI API
        response = self.getCompletionWithBackoff(
            #model="gpt-3.5-turbo",
            model="gpt-4",      # 8k token limit
            #model="gpt-4-32k-0314",
            max_tokens=maxTokensOut,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            messages=[{"role": "user", "content": prompt}],
            stream=True    # Stream the response, collect data as it comes, so that we retain data in timeouts. 
        )


        # Step 3: Collect the stream
        collectedChunks = []
        collectedMessages = []
        responseStr = ""
        success = False

        try:
            # For each chunk in the (streaming) response            
            lastNumTokens = startPBarAt
            for chunk in response:
                # Calculate time since start of request
                deltaTime = time.time() - startTime

                collectedChunks.append(chunk)
                chunkMessage = chunk['choices'][0]['delta']
                collectedMessages.append(chunkMessage)

                if "content" in chunkMessage:
                    responseStr += chunkMessage["content"]

                # Update progress bar
                numTokens = startPBarAt + self.getNumTokens(responseStr)
                if (numTokens > lastNumTokens) and (numTokens % 10 == 0):
                    pbar.update(numTokens-lastNumTokens)
                    lastNumTokens = numTokens

            # If we reach here, all the transmitted chunks have been parsed.  Success!
            success = True

        except Exception as e: 
            # If we reach here, there was an error with the request.  Fail, and let the calling function retry.
            print(e)
            return responseStr, success

        # If the number of prompt+response tokens is equal to the maxTokensOut, then we've reached the token limit.  Fail, and let the calling function retry.
        if (self.getNumTokens(prompt + responseStr) >= maxTokensOut-2):      # Added the 2 for wiggle room, just in case
            return responseStr, False

        # If we reach here, the request was successful.  Return the response.
        return responseStr, success
            

#
#   Engine: GPT3.5 Turbo
#
class engineGPT35Turbo:
    # Constructor
    def __init__(self, verboseOutput=False):        
        self.maxTokens = 4000
        self.name="gpt-3.5-turbo"

        # Tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")                
        self.verboseOutput = verboseOutput
        self.tokenGenerationRatePerSec = 0
        self.costPer1kTokensPrompt = 0.03       # $0.03 per 1000 tokens in prompt
        self.costPer1kTokensResponse = 0.06     # $0.06 per 1000 tokens in completion

    # Get the number of tokens for a string, measured using tiktoken
    def getNumTokens(self, strIn:str):
        tokens = self.tokenizer.encode(strIn)
        numTokens = len(tokens)
        return numTokens

    # Estimate the cost of a request, in dollars
    def estimateCostDollars(self, numTokensPrompt:int, numTokensResponse:int):
        costPrompt = (numTokensPrompt/1000) * self.costPer1kTokensPrompt 
        costResponse = (numTokensResponse/1000) * self.costPer1kTokensResponse 
        costTotal = costPrompt + costResponse
        return costTotal

    # A wrapper for getResponseHelper, that continues to retry for large replies that timeout after 300 seconds. 
    def getResponse(self, prompt:str, temperature:float=0.0, maxTokensOut:int=8000, maxRetries=4):
        numRetries = 0
        totalResponse = ""

        # Record start time of request
        startTime = time.time()

        # Calculate maximum tokens for progress bar
        maxTokensToGenerate = maxTokensOut - self.getNumTokens(prompt)

        success = False        

        numTokensSent = 0
        numTokensReceived = 0

        # Make requests to the API until (a) success, or (b) maxRetries is reached
        with tqdm(total=maxTokensToGenerate, unit="tokens") as pbar:
            while (numRetries <= maxRetries):                
                responseStr, success = self.getResponseHelper(prompt, temperature, maxTokensToGenerate, pbar, startPBarAt=self.getNumTokens(totalResponse))
                numTokensSent += self.getNumTokens(prompt)
                numTokensReceived += self.getNumTokens(responseStr)

                totalResponse += responseStr
                print("Total number of tokens sent and received: " + str(numTokensSent + numTokensReceived))                

                if (success):                
                    # If we reach here, success
                    break

                # If we reach here, the request failed.  Retry.
                # First, check if we've already hit the token limit
                
                if (self.getNumTokens(prompt + totalResponse) >= maxTokensOut-20):      # Added the 20 for wiggle room, just in case
                    print("engineGPT35Turbo: Generation token limit reached. Stopping.")
                    responseStr += "\n### TOKEN LIMIT REACHED ###"
                    break
                
                # Then, check if the "### NO ERRORS ###" or "### DONE ###" string is in the response.  If so, stop.
                if ("### NO ERRORS ###" in responseStr) or ("### DONE ###" in responseStr):
                    print("engineGPT35Turbo: Response indicated 'no errors' or 'done'. Stopping.")
                    break

                else:
                    print("engineGPT35Turbo: Retrying (attempt " + str(numRetries+1) + " / " + str(maxRetries) + ")")
                    numRetries += 1
                
                prompt += responseStr

        # Statistics: Store the generation speed
        deltaTime = time.time() - startTime
        deltaTime = round(deltaTime, 2)            
        
        numTokens = self.getNumTokens(totalResponse)
        transmissionRate = numTokens / deltaTime
        transmissionRate = round(transmissionRate, 2)
        self.tokenGenerationRatePerSec = transmissionRate

        if (self.verboseOutput):
            print("getResponse(): Total tokens generated: " + str(numTokens))
            print("getResponse(): Total time: " + str(deltaTime))
            print("getResponse(): Transmission rate: " + str(transmissionRate) + " tokens/sec")

        return totalResponse, success, numTokensSent, numTokensReceived


    def getResponseHelper(self, prompt:str, temperature:float, maxTokensOut:int, pbar, startPBarAt=0):
        # Step 1: Set generation length
        numPromptTokens = self.getNumTokens(prompt)
        maxPossibleTokens = self.maxTokens - numPromptTokens      # 8100 instead of 8192 to allow a bit of wiggle room incase tiktoken is inaccurate by a few tokens.
        if (maxTokensOut > maxPossibleTokens):
            if (self.verboseOutput):
                print("Warning: maxTokensOut is too large given the prompt length (" + str(numPromptTokens) + ").  Setting to max generation length of " + str(maxPossibleTokens))
            maxTokensOut = maxPossibleTokens

        # Record start time of request
        startTime = time.time()

        # Step 2: Perform request to the OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            #model="gpt-4",      # 8k token limit
            #model="gpt-4-32k-0314",
            max_tokens=maxTokensOut,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            messages=[{"role": "user", "content": prompt}],
            stream=True    # Stream the response, collect data as it comes, so that we retain data in timeouts. 
        )

        # Step 3: Collect the stream
        collectedChunks = []
        collectedMessages = []
        responseStr = ""
        success = False

        try:
            # For each chunk in the (streaming) response            
            lastNumTokens = startPBarAt
            for chunk in response:
                # Calculate time since start of request
                deltaTime = time.time() - startTime

                collectedChunks.append(chunk)
                chunkMessage = chunk['choices'][0]['delta']
                collectedMessages.append(chunkMessage)

                if "content" in chunkMessage:
                    responseStr += chunkMessage["content"]

                # Update progress bar
                numTokens = startPBarAt + self.getNumTokens(responseStr)
                if (numTokens > lastNumTokens) and (numTokens % 10 == 0):
                    pbar.update(numTokens-lastNumTokens)
                    lastNumTokens = numTokens

            # If we reach here, all the transmitted chunks have been parsed.  Success!
            success = True

        except Exception as e: 
            # If we reach here, there was an error with the request.  Fail, and let the calling function retry.
            print(e)
            return responseStr, success

        # If the number of prompt+response tokens is equal to the maxTokensOut, then we've reached the token limit.  Fail, and let the calling function retry.
        if (self.getNumTokens(prompt + responseStr) >= maxTokensOut-2):      # Added the 2 for wiggle room, just in case
            return responseStr, False

        # If we reach here, the request was successful.  Return the response.
        return responseStr, success




#
#   Engine: Claude V1
#
class engineClaudeV1:
    # Constructor
    def __init__(self, verboseOutput=False):        
        self.maxTokens = 8100
        self.name="claude-v1"

        # Tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")                
        self.verboseOutput = verboseOutput
        self.tokenGenerationRatePerSec = 0
        self.costPer1kTokensPrompt = 0.00       # 
        self.costPer1kTokensResponse = 0.00     # 

    # Get the number of tokens for a string, measured using tiktoken
    def getNumTokens(self, strIn:str):
        tokens = self.tokenizer.encode(strIn)
        numTokens = len(tokens)
        return numTokens

    # Estimate the cost of a request, in dollars
    def estimateCostDollars(self, numTokensPrompt:int, numTokensResponse:int):
        costPrompt = (numTokensPrompt/1000) * self.costPer1kTokensPrompt 
        costResponse = (numTokensResponse/1000) * self.costPer1kTokensResponse 
        costTotal = costPrompt + costResponse
        return costTotal

    # A wrapper for getResponseHelper, that continues to retry for large replies that timeout after 300 seconds. 
    def getResponse(self, prompt:str, temperature:float=0.0, maxTokensOut:int=8000, maxRetries=4):
        numRetries = 0
        totalResponse = ""

        # Record start time of request
        startTime = time.time()

        # Calculate maximum tokens for progress bar
        maxTokensToGenerate = maxTokensOut - self.getNumTokens(prompt)

        success = False        

        numTokensSent = 0
        numTokensReceived = 0

        # Make requests to the API until (a) success, or (b) maxRetries is reached
        with tqdm(total=maxTokensToGenerate, unit="tokens") as pbar:
            while (numRetries <= maxRetries):                
                responseStr, success = self.getResponseHelper(prompt, temperature, maxTokensToGenerate, pbar, startPBarAt=self.getNumTokens(totalResponse))
                numTokensSent += self.getNumTokens(prompt)
                numTokensReceived += self.getNumTokens(responseStr)

                totalResponse += responseStr
                print("Total number of tokens sent and received: " + str(numTokensSent + numTokensReceived))                

                if (success):                
                    # If we reach here, success
                    break

                # If we reach here, the request failed.  Retry.
                # First, check if we've already hit the token limit
                
                if (self.getNumTokens(prompt + totalResponse) >= maxTokensOut-20):      # Added the 20 for wiggle room, just in case
                    print("engineClaudeV1: Generation token limit reached. Stopping.")
                    responseStr += "\n### TOKEN LIMIT REACHED ###"
                    break
                
                # Then, check if the "### NO ERRORS ###" or "### DONE ###" string is in the response.  If so, stop.
                if ("### NO ERRORS ###" in responseStr) or ("### DONE ###" in responseStr):
                    print("engineClaudeV1: Response indicated 'no errors' or 'done'. Stopping.")
                    break

                else:
                    print("engineClaudeV1: Retrying (attempt " + str(numRetries+1) + " / " + str(maxRetries) + ")")
                    numRetries += 1
                
                prompt += responseStr

        # Statistics: Store the generation speed
        deltaTime = time.time() - startTime
        deltaTime = round(deltaTime, 2)            
        
        numTokens = self.getNumTokens(totalResponse)
        transmissionRate = numTokens / deltaTime
        transmissionRate = round(transmissionRate, 2)
        self.tokenGenerationRatePerSec = transmissionRate

        if (self.verboseOutput):
            print("getResponse(): Total tokens generated: " + str(numTokens))
            print("getResponse(): Total time: " + str(deltaTime))
            print("getResponse(): Transmission rate: " + str(transmissionRate) + " tokens/sec")

        return totalResponse, success, numTokensSent, numTokensReceived


    def getResponseHelper(self, prompt:str, temperature:float, maxTokensOut:int, pbar, startPBarAt=0):
        # Step 1: Set generation length
        numPromptTokens = self.getNumTokens(prompt)
        maxPossibleTokens = self.maxTokens - numPromptTokens      # 8100 instead of 8192 to allow a bit of wiggle room incase tiktoken is inaccurate by a few tokens.
        if (maxTokensOut > maxPossibleTokens):
            if (self.verboseOutput):
                print("Warning: maxTokensOut is too large given the prompt length (" + str(numPromptTokens) + ").  Setting to max generation length of " + str(maxPossibleTokens))
            maxTokensOut = maxPossibleTokens

        # Record start time of request
        startTime = time.time()

        # Step 2: Perform request to Claude API
        c = anthropic.Client(os.environ["ANTHROPIC_API_KEY"])

        response = c.completion_stream(
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
            stop_sequences=[anthropic.HUMAN_PROMPT],
            max_tokens_to_sample=maxTokensOut,
            model="claude-v1",
            stream=True,
            temperature = 0.0,            
        )

        # for data in response:
        #     print(data)

        # Step 3: Collect the stream
        collectedChunks = []
        collectedMessages = []
        responseStr = ""
        success = False

        try:
            # For each chunk in the (streaming) response            
            lastNumTokens = startPBarAt
            stopReason = None
            exception = None
            for chunk in response:
                # Calculate time since start of request
                deltaTime = time.time() - startTime

                collectedChunks.append(chunk)
                chunkMessage = chunk
                collectedMessages.append(chunkMessage)

                # print("CHUNK MESSAGE:")
                # print(chunkMessage)

                if "completion" in chunkMessage:
                    responseStr = chunkMessage["completion"]        # Claude API sends the whole completion each time

                # Keep track of any stops/errors
                if "stop_reason" in chunkMessage:
                    stopReason = chunkMessage["stop_reason"]
                if "exception" in chunkMessage:
                    exception = chunkMessage["exception"]

                # Update progress bar
                numTokens = startPBarAt + self.getNumTokens(responseStr)
                if (numTokens > lastNumTokens) and (numTokens % 10 == 0):
                    pbar.update(numTokens-lastNumTokens)
                    lastNumTokens = numTokens

            # If we reach here, all the transmitted chunks have been parsed.  Success!
            success = True

            # Check for any errors
            if (stopReason != None) and (stopReason != 'stop_sequence'):                
                print("engineClaudeV1: Error: Stop reason: " + str(stopReason))
                success = False
            if (exception != None):
                print("engineClaudeV1: Error: Exception: " + str(exception))
                success = False            

        except Exception as e: 
            # If we reach here, there was an error with the request.  Fail, and let the calling function retry.
            print(e)
            return responseStr, success

        # If the number of prompt+response tokens is equal to the maxTokensOut, then we've reached the token limit.  Fail, and let the calling function retry.
        if (self.getNumTokens(prompt + responseStr) >= maxTokensOut-2):      # Added the 2 for wiggle room, just in case
            return responseStr, False

        # If we reach here, the request was successful.  Return the response.
        return responseStr, success
            
