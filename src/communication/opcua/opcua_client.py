from asyncua.sync import Client, ThreadLoop
from asyncua import ua
import time
import logging
import yaml
from pathlib import Path
# ────────────────────────────────────────────────────────────────
# 🧠 Base class for all OPC UA devices
# ────────────────────────────────────────────────────────────────


class OPCUAClient:
    def __init__(self, url, auto_start=True):
        self.url = url
        self.client : Client
        self.tloop = None
        self.client_logger = logging.getLogger(__name__)
        if auto_start:
            self.start_communication()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_communication()
        return False

    def start_communication(self):
        """Initialize ThreadLoop and connect to OPC UA server."""
        self.tloop = ThreadLoop()
        self.tloop.daemon = True
        self.client = Client(self.url, tloop=self.tloop)
        self.tloop.start()
        #Remove some verbose
        logging.getLogger("asyncua.client.client").setLevel(logging.WARNING)
        self.client.connect()
        self.client_logger.info(f"✅ Connected to OPC UA server: {self.url}")

    def node(self, nodeid: str):
        """Shortcut for get_node."""
        return self.client.get_node(nodeid)
    
    def set_node_value(self, node, value, variant_type=None):
        try:
            if variant_type:
                dv = ua.DataValue(ua.Variant(value, variant_type))
                node.set_value(dv)
            else:
                node.set_value(value)
        except Exception as e:
            self.client_logger.exception("Set node '%s' failed: %s", node, e)

    def get_node_value(self, node, default=None):
        try:
            return node.get_value()
        except Exception as e:
            self.client_logger.exception("OPC UA read failed: %s", node)
            return default

    def stop_communication(self):
        """Gracefully disconnect and stop loop."""
        if self.client:
            self.client.disconnect()
        if self.tloop:
            self.tloop.stop()
        self.client_logger.info(f"🔌 Disconnected from {self.url}")

        


# ────────────────────────────────────────────────────────────────
# ⚙️ PLC Client (inherits base)
# ────────────────────────────────────────────────────────────────

class PLCClient(OPCUAClient):
    def __init__(self, url, auto_start=True):
        super().__init__(url, auto_start)
        if auto_start:
            self.init_nodes()

    def init_nodes(self):
    

        self.node_trigger = self.get_node('ns=4;i=5')
        self.node_break_loop = self.get_node('ns=3;s="MotoMotion_Instance"."BreakLoop"')
        self.node_step_z = self.get_node('ns=3;s="MotoMotion_Instance"."Step_Z"')
        self.node_closegripper = self.get_node('ns=3;s="MotoMotion_Instance"."CloseGrip"')
        self.node_opengripper = self.get_node('ns=3;s="MotoMotion_Instance"."OpenGrip"')

        """Movement Nodes"""
        self.node_x0 = self.get_node('ns=3;s="MotoLocal"."PosTCP"."TCPPosition"[0]')
        self.node_y0 = self.get_node('ns=3;s="MotoLocal"."PosTCP"."TCPPosition"[1]')
        self.node_z0 = self.get_node('ns=3;s="MotoLocal"."PosTCP"."TCPPosition"[2]')

        self.node_x1 = self.get_node('ns=3;s="MotoLocal"."PosTCP1"."TCPPosition"[0]')
        self.node_y1 = self.get_node('ns=3;s="MotoLocal"."PosTCP1"."TCPPosition"[1]')
        self.node_z1 = self.get_node('ns=3;s="MotoLocal"."PosTCP1"."TCPPosition"[2]')

        self.node_x2 = self.get_node('ns=3;s="MotoLocal"."PosTCP2"."TCPPosition"[0]')
        self.node_y2 = self.get_node('ns=3;s="MotoLocal"."PosTCP2"."TCPPosition"[1]')
        self.node_z2 = self.get_node('ns=3;s="MotoLocal"."PosTCP2"."TCPPosition"[2]')
        
        self.node_x3 = self.get_node('ns=3;s="MotoLocal"."PosTCP3"."TCPPosition"[0]')
        self.node_y3 = self.get_node('ns=3;s="MotoLocal"."PosTCP3"."TCPPosition"[1]')
        self.node_z3 = self.get_node('ns=3;s="MotoLocal"."PosTCP3"."TCPPosition"[2]')
        self.node_ry3 = self.get_node('ns=3;s="MotoLocal"."PosTCP3"."TCPPosition"[4]')
        self.state_job = self.get_node('ns=3;s="MotoMotion_Instance"."stateJob"')

    def send_coordinates0(self, x, y, z):
        for node, val in zip((self.node_x0, self.node_y0, self.node_z0), (x, y, z)):

            #node.set_value(ua.Variant(val, ua.VariantType.Float))

            #TEST

            dv = ua.DataValue(ua.Variant(val,ua.VariantType.Float))
            node.set_value(dv)
        
        print(f"📤 Sent coordinates to PosTCP0 PLC: ({x:.3f}, {y:.3f}, {z:.3f})")

    def send_coordinates1(self, x, y, z):
        for node, val in zip((self.node_x1, self.node_y1, self.node_z1), (x, y, z)):
            dv = ua.DataValue(ua.Variant(val,ua.VariantType.Float))
            node.set_value(dv)
        print(f"📤 Sent coordinates to PosTCP1 PLC: ({x:.3f}, {y:.3f}, {z:.3f})")
    
    def send_coordinates2(self, x, y, z):
        for node, val in zip((self.node_x2, self.node_y2, self.node_z2), (x, y, z)):
            dv = ua.DataValue(ua.Variant(val,ua.VariantType.Float))
            node.set_value(dv)
        print(f"📤 Sent coordinates to PosTCP2 PLC: ({x:.3f}, {y:.3f}, {z:.3f})")

    def send_coordinates3(self, x, y, z, ry = 0):
        for node, val in zip((self.node_x3, self.node_y3, self.node_z3, self.node_ry3), (x, y, z, ry)):
            dv = ua.DataValue(ua.Variant(val,ua.VariantType.Float))
            node.set_value(dv)
        print(f"📤 Sent coordinates to PosTCP3 PLC: ({x:.3f}, {y:.3f}, {z:.3f})")

    def set_trigger(self, value: bool):
        self.node_trigger.set_value(ua.Variant(value, ua.VariantType.Boolean))
        print("Sent Step")

    def set_breakloop(self, value: bool):
        self.node_break_loop.set_value(ua.DataValue(ua.Variant(value, ua.VariantType.Boolean)))

    def set_stepz(self, value: bool):
        self.node_step_z.set_value(ua.DataValue(ua.Variant(value, ua.VariantType.Boolean)))

    def set_closegripper(self, value: bool):
        self.node_closegripper.set_value(ua.DataValue(ua.Variant(value, ua.VariantType.Boolean)))
    
    def set_opengripper(self, value: bool):
        self.node_opengripper.set_value(ua.DataValue(ua.Variant(value, ua.VariantType.Boolean)))

    def get_state_job(self):
        return self.state_job.get_value()


# ────────────────────────────────────────────────────────────────
# 🤖 Yaskawa Robot Client (inherits base)
# ────────────────────────────────────────────────────────────────

class Yaskawa_YRC1000(OPCUAClient):
    def __init__(self, url, auto_start=True):
        super().__init__(url, auto_start)
        if auto_start:
            self.init_nodes()

    def init_nodes(self):
        """Initialize Yaskawa-specific nodes."""
        self.running_var = self.get_node(
            "ns=5;s=MotionDeviceSystem.Controllers.Controller_1.ParameterSet.IsRunning")
        self.controller_obj = self.client.nodes.root.get_child([
            "0:Objects",
            "2:DeviceSet",
            "4:MotionDeviceSystem",
            "4:Controllers",
            "4:Controller_1",
            "5:Methods"
        ])
        print("✅ Robot nodes initialized")

    def get_available_jobs(self):
        return self.controller_obj.call_method("5:GetAvailableJobs")

    def set_servo(self, enable: bool):
        return self.controller_obj.call_method("5:SetServo", enable)

    def start_job(self, job_name, block=True):
        print(f"▶️ Starting job: {job_name}")
        self.controller_obj.call_method("5:StartJob", job_name)
        time.sleep(0.1)
        if block:
            running = self.running_var.get_value()
            print("running job: ",job_name)
            while running == True:
                running = self.running_var.get_value()
            print("finished job: ", job_name)


class PLCNodeMap:
    def __init__(self, client: OPCUAClient, yaml_path: str):
        self.client = client

        with open(yaml_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.nodes = self._build_nodes(self.config)

    def _build_nodes(self, cfg):
        result = {}

        for key, value in cfg.items():

            if isinstance(value, dict):
                result[key] = {
                    k: self.client.node(v) for k, v in value.items()
                }
            else:
                result[key] = self.client.node(value)

        return result

class PLCInterface:
    def __init__(self, node_map : PLCNodeMap , plc_opcua_client : OPCUAClient):
        self.nodes = node_map.nodes
        self.io = plc_opcua_client

    def set_pregrasp_tcp(self, value: list[float]):
        self.io.set_node_value(self.nodes['plc']['pregrasp_tcp'], value, ua.VariantType.Float)

    def set_wrist_rotation_tcp(self, value: list[float]):
        self.io.set_node_value(self.nodes['plc']['wrist_rotate_tcp'], value, ua.VariantType.Float)

    def set_approach_tcp(self, value: list[float]):
        self.io.set_node_value(self.nodes['plc']['approach_tcp'], value, ua.VariantType.Float)

    def get_state_motion(self):
        return self.io.get_node_value(self.nodes['plc']['state_motion'])  

    def get_bool_6D_pose_data(self):
         return self.io.get_node_value(self.nodes['plc']['get_6D_pose_data'])   
# ────────────────────────────────────────────────────────────────
# 🚀 Main program
# ────────────────────────────────────────────────────────────────
def test_robot_comm():
        try:
            robot_url = "opc.tcp://192.168.0.56:16448"
            robot = Yaskawa_YRC1000(robot_url)
            robot.set_servo(True)
            # time.sleep(1)
            # robot.start_job('', block=True)
            time.sleep(1)
            robot.set_servo(False)
        finally:
            robot.stop_communication()
            print("🔚 Program ended.")

def test_plc_comm():
    try:
        plc_url = "opc.tcp://192.168.0.1:4840"
        plc_client = OPCUAClient(plc_url)

        node_map = PLCNodeMap(plc_client, r"C:\Users\lin40269\Desktop\Linh (Desktop)\01_Python\realsense\config\plc_opcua_nodes.yaml")
        
        plc_io = PLCInterface(node_map,plc_client)

        print(plc_io.nodes)
        print(plc_client.get_node_value(plc_io.nodes['plc']['pregrasp_tcp']))
        plc_client.set_node_value(plc_io.nodes['plc']['pregrasp_tcp'], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], ua.VariantType.Float)
        print(plc_client.get_node_value(plc_io.nodes['plc']['pregrasp_tcp']))

        plc_io.set_wrist_rotation_tcp([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        print("STATE_MOTION: ",plc_io.get_state_motion())
    finally:

        plc_client.stop_communication()


    


def main():
    test_plc_comm()



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()