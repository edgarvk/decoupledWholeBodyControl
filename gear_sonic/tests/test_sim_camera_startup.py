"""Exercise camera startup and transport without DDS, a controller, or a robot."""

from multiprocessing import shared_memory
import socket
import time
import unittest

import numpy as np
import zmq

from gear_sonic.camera.sensor_server import ImageMessageSchema
from gear_sonic.utils.mujoco_sim.image_publish_utils import ImagePublishProcess


class TestSimulationCameraStartup(unittest.TestCase):
    def make_publisher(self, port):
        return ImagePublishProcess({"ego_view": {"height": 16, "width": 24}}, image_dt=0.02, zmq_port=port)

    def assert_memory_released(self, names):
        for name in names:
            with self.assertRaises(FileNotFoundError):
                shared_memory.SharedMemory(name=name)

    def test_occupied_port_fails_startup_and_releases_memory(self):
        with socket.socket() as occupied:
            occupied.bind(("0.0.0.0", 0))
            occupied.listen()
            port = occupied.getsockname()[1]
            publisher = self.make_publisher(port)
            names = [info["name"] for info in publisher.shared_memory_info.values()]
            try:
                with self.assertRaisesRegex(RuntimeError, f"port {port}"):
                    publisher.start_process()
                self.assertFalse(publisher.process.is_alive())
                self.assert_memory_released(names)
            finally:
                publisher.stop()

    def test_ready_publisher_delivers_rgb_images_to_camera_schema(self):
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        publisher = self.make_publisher(port)
        names = [info["name"] for info in publisher.shared_memory_info.values()]
        context = zmq.Context()
        subscriber = context.socket(zmq.SUB)
        subscriber.setsockopt(zmq.SUBSCRIBE, b"")
        subscriber.setsockopt(zmq.LINGER, 0)
        frame = np.full((16, 24, 3), [230, 100, 20], np.uint8)
        try:
            publisher.start_process()
            self.assertTrue(publisher.started_event.is_set())
            subscriber.connect(f"tcp://127.0.0.1:{port}")
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                publisher.update_shared_memory({"ego_view_image": frame})
                if subscriber.poll(100):
                    import msgpack

                    message = msgpack.unpackb(subscriber.recv(), raw=False)
                    received = ImageMessageSchema.deserialize(message)
                    np.testing.assert_allclose(received.images["ego_view"], frame, atol=4)
                    self.assertGreater(received.timestamps["ego_view"], 0)
                    break
            else:
                self.fail("Camera startup succeeded but no image arrived")
        finally:
            subscriber.close()
            context.term()
            publisher.stop()
        self.assert_memory_released(names)


if __name__ == "__main__":
    unittest.main()
