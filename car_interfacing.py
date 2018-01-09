import json
import socket
from PIL import Image
from io import BytesIO
import numpy as np
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CarConnection(object):
    def __init__(self, machine_name='tigu6'):
        remote_ip = socket.gethostbyname(machine_name)
        send_server = (remote_ip, 22241)
        receive_server = (remote_ip, 22242)

        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.send_sock.connect(send_server)

        # Temp removed until fixes are made
        #self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #self.recv_sock.connect(receive_server)

    def send_commands_to_car(self, commands):
        """
        Method to send the movement commands to the relevant RCSnail car
        :param commands: a list of commands, in order steering (double), braking (double), throttle (double), gear (int)
        :return:
        """
        assert len(commands) == 4, 'The input array must have all 4 parameters'

        data = {
            'steering': commands[0],
            'braking': commands[1],
            'throttle': commands[2],
            'gear': commands[3]
        }

        json_body = json.dumps(data)

        self.send_sock.sendall((json_body + '\n').encode('utf-8'))


    def receive_data_from_stream(self):
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.recv_sock.connect((socket.gethostbyname('tigu6'), 22242))

        header = self.recv_sock.recv(4)
        bytes_to_read = int.from_bytes(header, byteorder='little')
        bitmap_image = self.recv_sock.recv(bytes_to_read)

        img = Image.open(BytesIO(bitmap_image))
        img_array = np.asarray(img.convert('RGB'))

        img_array = np.flip(img_array, axis=2)
        # Temporarily shut it down every time it asks due to issues
        self.recv_sock.shutdown(socket.SHUT_WR)
        self.recv_sock.close()

        return img_array


    def close(self):
        self.send_sock.shutdown(socket.SHUT_WR)
        self.send_sock.close()

        # Temp removed because it's closed every time it opens
        #self.recv_sock.shutdown(socket.SHUT_WR)
        #self.recv_sock.close()
