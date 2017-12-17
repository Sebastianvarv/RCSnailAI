import json
import socket


class CarConnection(object):
    def __init__(self, machine_name='tigu6'):
        remote_ip = socket.gethostbyname(machine_name)
        send_server = (remote_ip, 22241)

        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.send_sock.connect(send_server)

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
        pass

    def close(self):
        self.send_sock.shutdown(socket.SHUT_WR)
        self.send_sock.close()
