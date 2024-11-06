import numpy as np

from sock import SocketClient
from controller import Controller
import zmq
import signal
import pickle
import loguru

# NOTE This is the ip and port of the pc host connected to vr
GLOBAL_IP = "192.168.12.198"
GLOBAL_PORT = "34565"


logger = loguru.logger


def main():
    context = zmq.Context()
    sock = context.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, 1)
    sock.bind(f"tcp://*:{GLOBAL_PORT}")

    def sig_handler(sig, frame):
        logger.info("Exit")
        sock.close()
        context.term()
        exit(0)

    signal.signal(signal.SIGINT, sig_handler)

    controller = Controller()
    actions = controller.get_action()
    for idx, data in enumerate(actions):
        bin_data = pickle.dumps(data)
        sock.send(bin_data)
        logger.info(f"Sent data packet {idx+1}")



if __name__ == '__main__':
    # test()
    main()
