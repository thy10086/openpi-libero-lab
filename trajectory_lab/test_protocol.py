"""Test the OpenPI protocol against a local deterministic policy server."""

from threading import Thread
import unittest

import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.server import serve

from .libero_backend import RemotePolicy


class ProtocolTest(unittest.TestCase):
    def test_read_image_state_and_return_chunk(self):
        received = []

        def handler(connection):
            packer = msgpack_numpy.Packer()
            connection.send(packer.pack({"kind": "test-policy"}))
            received.append(msgpack_numpy.unpackb(connection.recv()))
            connection.send(packer.pack({"actions": np.zeros((10, 7), dtype=np.float32)}))

        with serve(handler, "127.0.0.1", 0, compression=None) as server:
            port = server.socket.getsockname()[1]
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            client = RemotePolicy(f"ws://127.0.0.1:{port}", 8)
            try:
                image = np.zeros((256, 256, 3), dtype=np.uint8)
                result = client.predict({
                    "raw": {"agentview_image": image, "robot0_eye_in_hand_image": image,
                            "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
                            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                            "robot0_gripper_qpos": np.array([0.02, -0.02])},
                    "prompt": "place the bowl on the plate",
                })
                self.assertEqual(result.shape, (10, 7))
                self.assertEqual(len(received), 1)
                self.assertEqual(received[0]["observation/image"].shape, (224, 224, 3))
                self.assertEqual(received[0]["observation/wrist_image"].shape, (224, 224, 3))
                self.assertEqual(received[0]["observation/state"].shape, (8,))
                self.assertEqual(received[0]["prompt"], "place the bowl on the plate")
            finally:
                client.close()
                server.shutdown()
                thread.join(timeout=5)
