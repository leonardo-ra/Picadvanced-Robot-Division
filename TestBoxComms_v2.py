# Commands to Communicate with the Test Box:
    # '1 X' - Connect System
    # '2 X' - Read Modules
    # '3 X' - Start Test
    # '4 X' - Tracking Results

# TCP communication configuration with the Test Box (default):
    # IP Address: 127.0.0.1
    # Port Number: 13000
    # Transfer data buffer(size): 8192 bytes

import time
import os
import json
import socket
import re

class TestBox:

    def __init__(self, language = 'EN', host = '127.0.0.1', port = 13000):
        """
        Pameters that are initialized when the class is created:
        :param: language - Language of the Test Box
        :param: host - IP Address of the Test Box
        :param: port - Port Number of the Test Box
        """
        self.host = host
        self.port = port
        self.socket = None
        os.system('start .\Comms_Test_Box\SW_CalBoard\App.exe')
        time.sleep(5)
        with open('Comms_Test_Box/language.json', 'r') as f:
            lang = json.load(f)[language]
            self.stringTCP1 = lang[0]    
            self.stringTCP2 = lang[1]    
            self.stringTCP3 = lang[2]    
            self.stringTCP4 = lang[3]   
        self.box_status = {}   # dictionary to store the status of the Test Box {'module': [(slot number, test result), ...], ...}
 
    def socket_connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        
    def socket_close(self):
        if self.socket:
            self.socket.close()
            self.socket = None

    def connect(self, module):
        """
        This function connects to a specific module of the Test Box. Has to connect in 10 seconds. 
        Sends TCP command ('1 X').
        :param: module
        :return: True or False depending on whether the connection was successful or not.
        """

        if not self.socket:
            self.socket_connect()

        message = '1 ' + module
        self.socket.sendall(message.encode())    # send message
        response = self.socket.recv(4096)
        recv = response.decode()
        if self.stringTCP1 not in recv:
            print("Failed to connect to Test Box")
            self.socket_close()   
            return False          
        else:
            print("Connected to Test Box")
            self.socket_close()  
            return True
    
    def read_modules(self, module):
        """
        This function initializes the read modules of the Test Box.
        Sends TCP command ('2 X'). It checks if the module was read or not. If the module was read it saves the slot and serial number.
        :param: module
        :return: True or False depending on whether the read modules was successful or not.
        """

        if not self.socket:
            self.socket_connect()

        message = '2 ' + module
        self.socket.sendall(message.encode())    # send message
        response = self.socket.recv(4096)
        recv = response.decode()
        print("Reading module " + module)
        if recv == "Error: Module " + module +" is not connected!":
            print("You have to connect to the Test Box first!")
            self.socket_close() 
            return False
        elif recv == "Error: No valid XFPs found!":
            print("No valid XFPs found! Maybe some Transceivers are missing?")
            self.socket_close() 
            return False
        else:
            self.socket_close() 
            return True
    
    def start_test(self, module):
        """
        This function initializes tests of the Test Box.
        Sends TCP command ('3 X'). It can only be started after the read_modules function has been called.
        :param: module
        :return: True or False depending on whether the tests were started or not.
        """

        if not self.socket:
            self.socket_connect()

        message = '3 ' + module
        self.socket.sendall(message.encode())    # send message
        response = self.socket.recv(4096)
        recv = response.decode()
        if self.stringTCP3 not in recv:
            print("Failed to start test")
            self.socket_close()   
            return False          
        else:
            print("Tests started on module " + module + "!")

            self.socket_close()  
            return True
        
    def tracking(self, module):
        """
        This function allows to track the results of the tests of the Test Box.
        Sends TCP command ('4 X'). It can only be sended after the start_test function has been called.
        :param: module
        :return: list of tuples containing the slot number and the test result for that module.
        """

        if not self.socket:
            self.socket_connect()

        message = '4 ' + module
        self.socket.sendall(message.encode())    # send message
        response = self.socket.recv(4096)
        recv = response.decode()
        self.process_tracking(module, recv)
        self.socket_close()
        return self.box_status[module]
        
    def process_tracking(self, module, result):
        """
        This function processes the tracking results of the Test Box.
        :param: module
        :param: result - the string containing the tracking results received from the Test Box.
        It saves the slot and the test result in the dictionary with the key 'module'
        """

        # Split the input string into separate module blocks
        module_blocks = re.split('\r\n\r\n', result)
        
        # Initialize an empty list to hold the results
        results = []

        # Loop over each module block
        for block in module_blocks[:-1]:
            # Split the block into separate lines
            lines = block.split('\r\n')
            
            # Initialize variables to hold the slot and final result values
            slot = None
            final_result = None
            # Loop over each line in the block
            for line in lines:
                # Check if the line contains a slot or final result value
                if line.startswith('Slot:'):
                    slot = line.split(': ')[1].strip()
                elif line.startswith('Final Result:'):
                    final_result = line.split(':')[1].strip()
                    
            
            # Add a tuple of the slot and final result values to the results list
            results.append((slot, final_result))
    
        self.box_status[module] = results


        
