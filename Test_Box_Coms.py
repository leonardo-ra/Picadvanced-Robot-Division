from Comms_Test_Box import TestBoxComms_v2 as TBox 
import time

host_testbox = '127.0.0.1'
port_testbox = 13000
connection = False
start = False
i = 1
module = "A"

box = TBox.TestBox('PT', host_testbox, port_testbox)

# while connection == False:
#     if i > 10:
#         break
#     connection = box.connect(module)
#     i +=1

# box.read_modules(module)

# i = 0

# while start == False:
#     if i > 10:
#         break
#     start = box.start_test(module)
#     i +=1
# start_time = time.time()

test_done = False

print("Tracking results on module " + module)
while test_done == False:
    # time.sleep(5)
    test_done = True
    results = box.tracking(module)
    for tup in results:                     # Check if any slot is still testing = '-'
        if tup[1] == '-':
            test_done = False
            print("Test not completed")
            print(results)
            break

# print(f"Test completed in {time.time()-start_time} seconds")
print(results)
