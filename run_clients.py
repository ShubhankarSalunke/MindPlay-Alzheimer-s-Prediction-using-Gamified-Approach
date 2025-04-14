import os

os.system("python clients/client_1.py")
os.system("python clients/client_2.py")

# Aggregate models
os.system("python server.py")
