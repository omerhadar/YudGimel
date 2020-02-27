import socket
from threading import Thread
import itertools

IP = "127.0.0.1"
PORT = 6666

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((IP, PORT))
client_socket.send(bytes("hi", "ascii"))
print(client_socket.recv(1024))
client_socket.close()
