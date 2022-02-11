import atexit
from shutil import move
from turtle import left
from unittest import result
import cv2
import mediapipe as mp
import math

from kivy.core.window import Window
Window.size = (960,540)

from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.properties import StringProperty
from kivy.properties import ObjectProperty
from kivy.properties import NumericProperty

mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder

import configparser
import os
import errno

class CameraPreview(Image):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = cv2.VideoCapture(1)
        Clock.schedule_interval(self.update, 1.0 / 30)

        self.mp_pors = mp.solutions.pose
        self.pose = self.mp_pors.Pose()
        self.mp_draw = mp.solutions.drawing_utils

        self.sender = None
        self.isSend = False
        self.highlightID = 32

    def update(self, dt):
        success, self.frame = self.capture.read()

        self.frame = cv2.cvtColor(cv2.flip(self.frame, 1), cv2.COLOR_BGR2RGB)
        self.frame.flags.writeable = False
        self.results = self.pose.process(self.frame)
        self.frame.flags.writeable = True
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
        if self.results.pose_landmarks:
            """highlights_landmark = self.results.pose_landmarks.landmark[self.highlightID]
            h, w, c = self.frame.shape
            cx, cy = int(highlights_landmark.x*w), int(highlights_landmark.y*h)
            cv2.circle(self.frame, (cx,cy), 15, (255, 0, 255), cv2.FILLED)"""
            self.mp_draw.draw_landmarks(self.frame, self.results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        buf = cv2.flip(self.frame, 0).tostring()
        texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt = 'bgr', bufferfmt = 'ubyte')
        self.texture = texture
        
        if self.isSend:
            self.sender.update(self.results)

        #self.test()

    def set_sender(self):
        self.sender = OSC_Sender(self.results)
        self.isSend = True
        print("set_sender")

    def test(self):
        world_landmark = self.results.pose_world_landmarks.landmark
        landmark_point = [[None] * 3 for i in range(33)]
        for index, landmark in enumerate(world_landmark):
            landmark_point[index] = [landmark.x, landmark.y, landmark.z]
        print(landmark_point[32][0])

    def update_config_param(self, num):
        if self.isSend:
            self.sender.bias_foot_z = num
        else:
            pass

class OSC_Sender():
    def __init__(self, results):
        IP = '127.0.0.1'
        PORT = 39570
        self.client = udp_client.UDPClient(IP, PORT)

        self.landmark = results.pose_landmarks.landmark
        self.world_landmark = results.pose_world_landmarks.landmark
        x = (self.landmark[23].x + self.landmark[24].x) / 2
        y = (self.landmark[23].y + self.landmark[24].y) / 2
        self.origin_pixel = [x, y, 0]
        self.distance_origin_foot = (self.world_landmark[31].y + self.world_landmark[32].y) / 2 
        self.distance_per_pixel = getConfigParams('PARAMS','distance_per_pixel')
        self.bias_foot_z = getConfigParams('PARAMS','bias_foot_z')
        self.heel_misalignment = (self.world_landmark[29].z + self.world_landmark[30].z) / 2


        self.world_landmark_point = [[None] * 3 for i in range(33)]
        for index, landmark in enumerate(self.world_landmark):
            self.world_landmark_point[index] = [landmark.x, landmark.y, landmark.z]
        self.pixel_landmark_point = [[None] * 3 for i in range(33)]
        for index, landmark in enumerate(self.landmark):
            self.pixel_landmark_point[index] = [landmark.x, landmark.y, landmark.z]

        default_rotation_right = self.calculate_rotation(self.world_landmark_point[31],self.world_landmark_point[29])
        default_rotation_left = self.calculate_rotation(self.world_landmark_point[32], self.world_landmark_point[30])
        self.default_rotation_foot = [default_rotation_right, default_rotation_left]

    def update(self, results):
        self.update_landmark(results)
        self.send_osc()

    def update_landmark(self, results):
        self.landmark = results.pose_landmarks.landmark
        self.world_landmark = results.pose_world_landmarks.landmark
        for index, landmark in enumerate(self.world_landmark):
            self.world_landmark_point[index] = [landmark.x, landmark.y, landmark.z]
        for index, landmark in enumerate(self.landmark):
            self.pixel_landmark_point[index] = [landmark.x, landmark.y, landmark.z]

    def calculate_waist_point(self):
        x = (self.pixel_landmark_point[23][0] + self.pixel_landmark_point[24][0]) / 2
        y = (self.pixel_landmark_point[23][1] + self.pixel_landmark_point[24][1]) /2
        now_waist = [x, y, 0]#現在の腰のlandmark(pixel)
        for index, i in enumerate(now_waist):
            now_waist[index] = i - self.origin_pixel[index]#原点のと差
        now_waist[1] = now_waist[1] * -1#mediapipeとSteamVRではYの符号が反対のため
        now_waist = [n * self.distance_per_pixel for n in now_waist]#原点との差にピクセルあたりの距離を掛ける
        now_waist[1] = self.distance_origin_foot + now_waist[1]#mediapipeでは原点が腰にあるため、足との距離を足して足の長さ分浮かせる
        now_waist[2] = (((self.world_landmark_point[29][2] + self.world_landmark_point[30][2]) / 2) - self.heel_misalignment) * -1#かかとの位置から腰のZを計算
        return now_waist

    def calculate_rotation(self, my_pos, target_pos):
        rot_x = 0
        rot_y = math.atan2(target_pos[0] - my_pos[0], target_pos[2] - my_pos[2])
        rot_z = 0
        return [math.degrees(rot_x), math.degrees(rot_y), math.degrees(rot_z)]

    """def calculate_diff(self):
        return_dict = {}
        right_foot = self.pixel_landmark_point[31]
        left_foot = self.pixel_landmark_point[32]
        waist = self.calculate_waist_point
        return_dict["right_foot"] = [self.origin[0] - right_foot.x, self.origin[1] - right_foot.y, self.origin[2] - right_foot.z]
        return_dict["left_foot"] = [self.origin[0] - left_foot.x, self.origin[1] - left_foot.y, self.origin[2] - left_foot.z]
        return_dict["waist"] = [self.origin[0] - waist[0], self.origin[1] - waist[1], self.origin[2] - waist[2]]
        return return_dict"""

    def send_osc(self):
        """osc送信実行関数
        座標データを処理してVRCに合った形に加工してから実際の送信関数に渡す"""

        """move_waist_y = self.positionMemory_right.ifSameSign_y() and self.positionMemory_left.ifSameSign_y()
        move_waist_z = self.positionMemory_right.ifSameSign_z() and self.positionMemory_left.ifSameSign_z()"""
        waist = self.calculate_waist_point()

        landmarks = [31,32]
        new_foot = [[0],[0]]
        for index, land_index in enumerate(landmarks):
            land = self.world_landmark_point[land_index]
            pos_y = self.distance_origin_foot - land[1] + (waist[1] - self.distance_origin_foot)
            if pos_y < 0:
                pos_y = 0.0
            default_rotation = self.default_rotation_foot[index]
            rotation = self.calculate_rotation(land, self.world_landmark_point[index - 2])
            new_foot[index] = [land[0] + waist[0], pos_y, land[2] + waist[2] + self.bias_foot_z, rotation[0] - default_rotation[0], rotation[1] - default_rotation[1], rotation[2] - default_rotation[2]]
        print(new_foot[0][4])
        self.sender(0, waist[0], waist[1], waist[2], 0, 0, 0)

        for index, x in enumerate(new_foot):
            self.sender(index + 1, x[0], x[1], x[2], x[3], x[4], x[5])
        """
        if move_waist_y:
            for index, x in enumerate(new_foot):
                self.sender(index + 1, x[0], self.current_foot[index][1], x[2], 0, 0, 0)
        else:
            for index, x in enumerate(new_foot):
                self.sender(index + 1, x[0], x[1], x[2], 0, 0, 0)
            self.current_foot = new_foot"""
        
        

    def sender(self, index, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z):
        """osc送信関数
        引数をVMTへとoscで送信する
        回転はオイラー角かクォータニオンに変換される"""
        enable = 1
        timeoffset = 0.0
        rot_x, rot_y, rot_z, rot_w = EulerAnglesToQuaternion(rot_x, rot_y, rot_z)
        data = [index, enable, timeoffset, pos_x, pos_y, pos_z * -1, rot_x, rot_y, rot_z, rot_w]

        msg = OscMessageBuilder(address = '/VMT/Raw/Unity')
        for i in data:
            msg.add_arg(i)
        m = msg.build()

        self.client.send(m)

def EulerAnglesToQuaternion(x, y, z):
    roll, pitch, yaw = math.radians(x), math.radians(y), math.radians(z)
    cosRoll, sinRoll = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cosPitch, sinPitch = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cosYaw, sinYaw = math.cos(yaw / 2.0), math.sin(yaw / 2.0)

    q0 = cosRoll * cosPitch * cosYaw + sinRoll * sinPitch * sinYaw
    q1 = sinRoll * cosPitch * cosYaw - cosRoll * sinPitch * sinYaw
    q2 = cosRoll * sinPitch * cosYaw + sinRoll * cosPitch * sinYaw
    q3 = cosRoll * cosPitch * sinYaw - sinRoll * sinPitch * cosYaw
    
    return q0, q1, q2, q3

def getConfigParams(section,key):
    config_ini = configparser.ConfigParser()
    config_ini_path = 'config.ini'

    if not os.path.exists(config_ini_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_ini_path)

    config_ini.read(config_ini_path, encoding='utf-8')
    return config_ini.getfloat(section, key)

def setConfigParams(section,key,value):
    config = configparser.ConfigParser()
    config_ini_path = 'config.ini'
    config.read(config_ini_path, encoding='utf-8')
    config[section][key] = str(value)
    with open('Config.ini','w') as configfile:
        config.write(configfile)

class Timer(BoxLayout):

    text = StringProperty()
    class2run = ObjectProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text = '3'

    def on_command(self):
        if self.text == 'Run':
            self.text = '3'
        Clock.schedule_interval(self.on_countdown, 1.0)

    def on_countdown(self, dt):
        self.text = str(int(self.text) - 1)
        if int(self.text) == 0:
            self.text = 'Run'
            self.class2run.set_sender()
            return False

class MySlider(BoxLayout):
    num_text = StringProperty()
    role_text = StringProperty()
    min = NumericProperty()
    max = NumericProperty()
    step = NumericProperty()
    value = NumericProperty()
    class2bias = ObjectProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.value = getConfigParams('PARAMS','bias_foot_z')
        self.num_text = str(self.value)

    def on_slider(self):
        self.value = round(self.ids.slider1.value, 2)
        self.num_text = str(self.value)
        self.class2bias.update_config_param(self.value)

    def up_slider(self):
        setConfigParams('PARAMS','bias_foot_z',self.value)

class MainScreen(FloatLayout):
    pass

class GuiApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = 'PoseTracking'

    def build(self):
        return MainScreen()

if __name__ == '__main__':
    GuiApp().run()