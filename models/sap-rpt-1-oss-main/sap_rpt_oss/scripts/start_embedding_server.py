# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import multiprocessing
import pickle
import sys
from subprocess import Popen
from typing import Optional

import zmq

from sap_rpt_oss.constants import ZMQ_PORT_DEFAULT, embedding_model_to_dimension_and_pooling


def is_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def start_embedding_server(sentence_embedding_model_name: str, gpu_idx: Optional[int] = None, device_ids: Optional[list] = None):
    """Start embedding server(s) for sentence embedding generation.
    
    Args:
        sentence_embedding_model_name: Name of the sentence embedding model
        gpu_idx: Single GPU index (deprecated, use device_ids instead)
        device_ids: List of GPU device IDs to use (if None, uses gpu_idx or defaults to 0)
    """
    # Support both old gpu_idx and new device_ids parameter
    if device_ids is None:
        if gpu_idx is None:
            device_ids = [0]
        else:
            device_ids = [gpu_idx]
    
    processes = []
    
    # Start a server for each GPU
    for idx, gpu_id in enumerate(device_ids):
        zmq_port = ZMQ_PORT_DEFAULT + idx
        print(f'Start async server to compute embedding on port {zmq_port} for GPU {gpu_id}')
        if is_port_in_use(zmq_port):
            print(f'Port {zmq_port} already in use, not starting again.')
        else:
            commands = [
                sys.executable, '-m', 'sap_rpt_oss.scripts.zmq_server', '--port',
                str(zmq_port), '--gpu_idx',
                str(gpu_id), '-semn', sentence_embedding_model_name
            ]
            process = Popen(commands)
            processes.append((process, zmq_port))

        # Register a function that kills the subprocess when the main process exits.
        # This is needed because otherwise the subprocess would keep running even
        # after the main process exits.
        def kill_subprocess():
            for process, port in processes:
                if process is not None:
                    process.terminate()
                    print(f'Subprocess on port {port} terminated')

        atexit.register(kill_subprocess)


def wait_forever_until_done(gpu_idx: Optional[int] = None):
    if gpu_idx is None:
        gpu_idx = 0
    zmq_port = ZMQ_PORT_DEFAULT + gpu_idx
    # Wait until the server is ready - send a test message without timeout
    print('\n\nWaiting for server to start on port...', zmq_port)
    socket = zmq.Context().socket(zmq.REQ)
    socket.connect(f'tcp://localhost:{zmq_port}')
    socket.send(pickle.dumps(['hello world']))
    socket.recv()
    socket.close()
    del socket
    print('Done!')


def wait_until_done(gpu_idx: Optional[int] = None, timeout=60):
    p = multiprocessing.Process(target=wait_forever_until_done, args=(gpu_idx, ))
    p.start()

    # Wait for `timeout` seconds or until process finishes
    p.join(timeout)

    # If thread is still active
    if p.is_alive():
        print('Timeout expired, killing process...')

        # Terminate - may not work if process is stuck for good
        p.terminate()
        # OR Kill - will work for sure, no chance for process to finish nicely however
        # p.kill()

        p.join()

        raise TimeoutError('Process did not finish in time - zmq server starting failed, probably?')


def test(sentence_embedding_model_name, gpu_idx: Optional[int] = None):
    if gpu_idx is None:
        gpu_idx = 0
    zmq_port = ZMQ_PORT_DEFAULT + gpu_idx
    print(f'Running test embedding job on port {zmq_port}...')
    socket = zmq.Context().socket(zmq.REQ)
    socket.connect(f'tcp://localhost:{zmq_port}')
    # Timeout after 10 seconds
    socket.setsockopt(zmq.RCVTIMEO, 10000)
    socket.setsockopt(zmq.LINGER, 0)

    serialized_data = pickle.dumps(['hello', 'world'])
    socket.send(serialized_data)

    try:
        response = socket.recv()
    except zmq.error.Again as e:
        raise ValueError('No response from server, it did not start correctly.') from e

    response = pickle.loads(response)
    assert len(response) == 2
    for i in range(2):
        assert len(response[i]) == 2 * embedding_model_to_dimension_and_pooling[sentence_embedding_model_name][0]
    print('Test passed!')


if __name__ == '__main__':
    start_embedding_server('sentence-transformers/all-MiniLM-L6-v2')
    wait_until_done()
    test('sentence-transformers/all-MiniLM-L6-v2')
