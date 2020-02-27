import socket
from threading import Thread
import itertools

IP = "127.0.0.1"
PORT = 6666


def on_new_client(cl_socket, addr, ready, id):
    req = cl_socket.recv(1024)
    print(id)
    cl_socket.send(req)


def wait_for_client(sv_socket, ret_list):
    ret_list.append(sv_socket.accept())


def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((IP, PORT))
    server_socket.listen(10)

    players = 0
    ready = []
    threads = []
    accept_list = []

    accept_thread = Thread(target=wait_for_client, args=(server_socket, accept_list))
    accept_thread.start()

    while 1:
        if not accept_thread.is_alive():
            ready.append(0)
            c, addr = accept_list.pop(0)
            t = Thread(target=on_new_client, args=(c, addr, ready, players))
            threads.append(t)
            t.start()
            players += 1
            if players < 8 and sum(ready) < players:
                accept_thread = Thread(target=wait_for_client, args=(server_socket, accept_list))
                accept_thread.start()
            elif players == 8 or sum(ready) == players:
                break
    server_socket.close()


if __name__ == '__main__':
    main()
