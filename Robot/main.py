import argparse
import cv2
import select
import socket
from Comms_Test_Box import TestBoxComms as Tbox
from time import sleep
import threading

class robot:
    def __init__(self, args):
        self.host_robot = args["host_ip_robot"]
        self.port_robot = args["port_robot"]
        self.host_box = args["host_ip_box"]
        self.port_box = args["port_box"]
        self.camera_index = args["camera_index"]
        self.cam = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)  # Create video stream
        self.cam.set(3, 1280)
        self.cam.set(4, 720)
        self.box_status = {}
        for i in range(1, args["number_of_test_box"]+1):
            self.box_status[i] = (4,1)
        self.present_box = "A"
        self.started_tests_flags = [0] * args["number_of_test_box"]

    def robot_communication(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host_robot, self.port_robot))                                  # Bind to the port  
        s.listen()                                                                  # Wait for client connection
        print("Listening for incoming connections")
        self.conn_robot, addr = s.accept()                                          # Establish connection with client
        print(f"Connection established with the robot at the address: {addr}")

    def box_communication(self):
        i = 1
        self.conn_box_flag = [False, False, False, False]
        self.box = Tbox.TestBox('PT', self.host_box, self.port_box)
        while self.conn_box_flag != [True, True, True, True]:
            if i > 10:
                break
            for j in range(self.number_of_test_box):
                self.conn_box_flag[j] = self.box.connect(chr(j+65))
            i +=1       
         
    def message_to_send_to_robot(self, tuplo):
        string = str(tuplo)
        string2 = string.encode()
        self.conn_robot.send(string2)
        print("Sent: " + string)  

    def Box_Calibration(self):
        self.message_to_send((1,1))

    def Table_Calibration(self):
        self.message_to_send((1,1))       

    def check_if_test_done(self):
        for i in range(0, len(self.started_tests_flags)):
                self.test_done = True
                if self.started_tests_tests[i] == False:        
                    continue                                # Go through only the modules that are testing
                else:
                    print("Tracking results on module " + chr(i+65))
                    results = self.box.tracking(chr(i+65))  # Receives a list of tuples (slot, Final Result)
                    for tup in results:                     # Check if any slot is still testing = '-'
                        if tup[1] == '-':
                            self.test_done = False
                            break
                    if self.test_done == True:
                        flags = (chr(i+65), 0, 1)
                        self.message_to_send_to_robot(flags)
                        self.flag = True
                        break 

    def check_procedures(self):
        self.flag = False
        if any(self.started_tests_flags):   # if some test has been started
            self.check_if_test_done()
            if all(self.started_tests_flags):
                while self.flag == False:
                    self.check_if_test_done()
            else:
                present_index = ord(self.present_box) - 65
                while self.started_tests_flags[present_index] == False:
                    present_index = (present_index + 1)%len(self.started_tests_flags)
                flags = (chr(present_index+65), 4, 0)
                self.message_to_send_to_robot(flags)    
        else:
            print("No tests are running")
            print(f"Working on the present box: {self.present_box}")
            flags = (ord(self.present_box)-64, *self.box_status[ord(self.present_box) - 64]) # flags = (Test Box "A", Slots available, test done)
            self.message_to_send_to_robot(flags)

    def main(self):
        """
        Start connection with robot Polyscope
        """
        print("--------------------------------Starting program----------------------------------------")
        self.robot_communication()
        self.box_communication()   
    
        while True:
            try:
                ready, _, _ = select.select([self.conn_robot], [], [], 3600)
                if ready:
                    data = self.conn_robot.recv(1024)
                    message = data.decode()
                    if message:
                        print(f"Received from robot: {message}")

                        if message == "TestBox Connected?":
                            if self.conn_box_flag == [True, True, True, True]:
                                self.message_to_send_to_robot((1,1))
                            else:
                                self.message_to_send_to_robot()

                        elif message == "Box Calibrated?":
                            self.Box_Calibration()

                        elif message == "Table Calibrated?":
                            self.Table_Calibration()

                        elif message == "Started Tests?":
                            self.check_procedures()

                        elif message == "Start Tests":
                            aux = self.box.read_modules(self.present_box)                   # Read modules
                            if aux:
                                self.box.start_test(self.present_box)                       # Start test
                                self.started_tests_flags[ord(self.present_box)-65] = True   # Set flag that test has started on that module
                                self.message_to_send_to_robot((1,1))                        # Indicates to the robot that test has started on that module

                    else:
                        print("! Polyscope program was stopped ! Ctrl+C on the terminal, Reboot the system")
                        sleep(5)

            except socket.error as socketerror:
                print("Error: ", socketerror)
                break 

            sleep(0.1) # Pauses the execution of the script for 0.1 seconds, to avoid overloading the server.

        self.conn_robot.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Add command-line arguments
    parser.add_argument("-hpr", "--host_ip_robot", type=str, help="Host IP used to make TCP connection to the robot polyscope", default="192.1.1.2")
    parser.add_argument("-hpb", "--host_ip_box", type=str, help="Host IP used to make TCP connection to the test box", default="127.0.0.1")
    parser.add_argument("-port_rob", "--port_robot", type=int, help="Port used to make TCP connection to the robot polyscope", default=6000)
    parser.add_argument("-port_box", "--port_box", type=int, help="Port used to make TCP connection to the test box", default=13000)
    parser.add_argument("-cam_ind", "--camera_index", type=int, help="Index of camera we want to access", default=1)
    parser.add_argument("-tbox_num", "--number_of_test_box", type=int, help = "Number of test box we want to access", default = 4)
    # Parse command-line arguments
    args = vars(parser.parse_args())
    
    # Create an instance of the program and run the main method
    program = robot(args)
    program.main()