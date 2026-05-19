import threading
from queue import Queue, Empty, Full

class PoseWorker:
    def __init__ (self, pose_input_queue : Queue):
        self._pose_input_queue = pose_input_queue
        self._euler_angle = Queue(maxsize=1)
        self._running = False
        self._process_pose_thread : threading.Thread = None

    def start(self):
        self._running = True
        self._process_pose_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_pose_thread.start()

    def stop(self):
        self._running = False
        self._process_pose_thread.join()

    def put_processed_pose(self, pose):
        try:
            self._processed_pose_queue.put_nowait(pose)
        except Full:
            try:
                self._processed_pose_queue.get_nowait()
            except Empty:
                pass
        
        self._processed_pose_queue.put_nowait(pose)

    def _process_loop(self):
        while self._running:
            try:
                pose = self._pose_input_queue.get(timeout=1.0)
            except Empty:
                continue
            except ValueError:
                continue
            

    def process_pose(self, pose): #orignal obj pose in camera frame
        
        #Convert obj pose in cam frame to gripper frame

        #Calculate cross product and angle

        #Calculate rotation matrix via Rodrigues

        #Calculate ZYX angle from rotation matrix using atan2

        #Return robot RX RY RZ angles
        
        pass


    @property
    def processed_pose_queue(self):
        return self._processed_pose_queue


if __name__ == "__main__":
    pass