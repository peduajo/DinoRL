from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from stable_baselines3.common.utils import set_random_seed
from typing import Callable
from PIL import Image
import gymnasium as gym
import numpy as np
from selenium.webdriver.common.action_chains import ActionChains
import torch.nn.functional as F
import torch.optim as optim

import pyautogui

from ptan import ptan

import warnings
warnings.filterwarnings('ignore')

import collections

from torch.utils import tensorboard

import torch 

from dqn_utils import DuelingDQN, calc_loss_prio, batch_generator

from gymnasium import spaces
from ray.tune.registry import register_env

import pandas as pd

import os 
import time 
from ultralytics import YOLO
import copy 

from types import SimpleNamespace

dqn_simple_params = SimpleNamespace(
    env_name='Dino',
    stop_reward=2000.0,
    run_name='dino',
    replay_size=100000,
    replay_initial=100,
    target_net_sync=1000,
    epsilon_frames=15000,
    epsilon_start=1.0,
    epsilon_final=0.025,
    learning_rate=3e-4,
    gamma=0.9,
    batch_size=64,
    seed=123,
    counts_reward_scale=0.25,
    n_step=4
)

class ExperienceBuffer:
    def __init__(self, capacity, device):
        self.buffer = collections.deque(maxlen=capacity)
        self.device = device

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        s_a, probs = \
            zip(*[self.buffer[idx] for idx in indices])
        
        s_a = torch.cat(s_a, dim=0).to(self.device).to(torch.float32)
        probs = torch.cat(probs, dim=0).to(self.device).to(torch.float32)

        return s_a, probs

class DinoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, device):
        super().__init__()

        self.screenshot_filename = f'captura_de_pantalla.png'
        chrome_options = Options()
        chrome_options.add_argument("--window-size=2546x1294")
        chrome_options.add_experimental_option("prefs", {"profile.default_content_setting_values.automatic_downloads": 1})

        service = Service(executable_path="/usr/bin/chromedriver")
        self.driver = webdriver.Chrome(service=service, options=chrome_options)

        try:
            self.driver.get("https://www.google.com/")
        except WebDriverException:
            pass 
            self.driver.maximize_window()
            time.sleep(2)

        self.body = self.driver.find_element(By.TAG_NAME, "body")
        self.actions_driver = ActionChains(self.driver)

        self.coordinates = (1000, 0, 1600, 200)
        self.coordinates_end = (260, 115, 300, 150)
        self.width = self.coordinates[2] - self.coordinates [0]
        self.height = self.coordinates[3] - self.coordinates [1]
        self.img_end = None 
        self.img_size = 224

        self.max_width_enemy = 0.10
        self.max_height_enemy = 0.3
        self.max_delta_object = 0.75
        self.dino_y_default = 0.79
        self.dino_height_default = 0.3

        self.vision_model = YOLO('runs/detect/train12/weights/best.engine')

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape = (15,), dtype=np.float16)

        self.df_objects = None
        self.df_agg = None

        self.buffer = ExperienceBuffer(int(1e6), device)
        self.df_data = None

    def reset(self, seed=0, options=None):
        time.sleep(2)
        #self.body.send_keys(Keys.SPACE)
        pyautogui.press('space')
        self.timestep = 0
        self.dino_is_in_floor = True 
        self.last_dino_y = 0.80 #suponiendo que empieza desde el suelo
        self.last_is_down_dino = False

        #cache del primer objeto
        self.last_first_object_x = None
        self.last_first_object_y = None
        self.last_first_object_w = None
        self.last_first_object_h = None

        self.delta_last_object = 0.0

        self.is_key_down = False

        obs = self.observation
        self.history_obs = [obs]
        self.history_actions = []

        self.n_ups = 0
        self.n_pass = 0
        self.n_downs = 0

        return obs
    
    def predict_current_screen(self, image):
        data_prediction = self.vision_model(image, iou = 0.3, verbose = False)[0].boxes

        classes = data_prediction.cls.cpu().numpy().astype(np.int16).tolist()
        ys = data_prediction.xywhn[:,0].cpu().numpy()
        xs = data_prediction.xywhn[:,1].cpu().numpy()
        ws = data_prediction.xywhn[:,2].cpu().numpy()
        hs = data_prediction.xywhn[:,3].cpu().numpy()

        self.df_objects = pd.DataFrame(data={'cls': classes, 'x': ys, 'y': xs, 'w': ws, 'h': hs})

        self.df_agg = self.df_objects.groupby('cls').agg(n_items=('cls', 'count')).reset_index()
    
    @property
    def observation(self):
        self.driver.save_screenshot(self.screenshot_filename)

        #shutil.copy(self.screenshot_filename, f"debug/{self.timestep}.png")

        image = Image.open(self.screenshot_filename).convert('L')
        image = image.crop(self.coordinates)

        self.predict_current_screen(image)

        #set dino Y 
        dino_data = self.df_objects.query("cls == 0")

        n_dino_data = dino_data.shape[0]
        if n_dino_data == 0:
            print('ERROR: No dino found!')

        dino_y = dino_data.y.values[0] if n_dino_data > 0 else self.dino_y_default
        is_down_dino = 1 if self.is_key_down else 0
        last_is_down_dino = self.last_is_down_dino
        self.last_is_down_dino = is_down_dino

        last_y_dino = copy.deepcopy(self.last_dino_y)
        self.last_dino_y = dino_y
        
        self.dino_is_in_floor = True if dino_y < 0.82 and dino_y > 0.78 else False
        
        df_enemies = self.df_objects.query("cls == 1")

        dist_x_first = 1.0
        height_img_first = 0.8
        height_bbox_first = 0.17 / self.max_height_enemy
        width_bbox_first = 0.05 / self.max_width_enemy
        x2_first = 1.0

        dist_x_second = 1.0
        height_img_second = 0.8
        height_bbox_second = 0.17 / self.max_height_enemy
        width_bbox_second = 0.05 / self.max_width_enemy
        x2_second = 1.0

        new_delta = False 

        n_enemies = df_enemies.shape[0]
        if n_enemies > 0:
            df_enemies = df_enemies.sort_values(by="x", ascending=True)

            dist_x_first = df_enemies.x.values[0] 
            height_img_first = df_enemies.y.values[0]
            height_bbox_first = df_enemies.h.values[0] / self.max_height_enemy
            width_bbox_first = df_enemies.w.values[0] / self.max_width_enemy

            x2_first = df_enemies.y.values[0] + (df_enemies.h.values[0] * 0.5)
            x2_first /= 0.92

            #si antes no habia objeto no se puede obtener el delta
            if self.last_first_object_x is not None:
                #primero hacemos un check de que el objeto anterior sea el mismo
                check1 = (dist_x_first < self.last_first_object_x)
                check2 = (height_img_first < (self.last_first_object_y + 0.025) and height_img_first > (self.last_first_object_y - 0.025))
                check3 = (height_bbox_first < (self.last_first_object_h + 0.025) and height_bbox_first > (self.last_first_object_h - 0.025))
                check4 = (width_bbox_first < (self.last_first_object_w + 0.025) and width_bbox_first > (self.last_first_object_w - 0.025))

                if check1 and check2 and check3 and check4:
                    delta_first_object = self.last_first_object_x - dist_x_first
                    new_delta = True 

            #save info current first object
            self.last_first_object_x = dist_x_first
            self.last_first_object_y = height_img_first
            self.last_first_object_h = height_bbox_first
            self.last_first_object_w = width_bbox_first

            if n_enemies > 1:
                dist_x_second = df_enemies.x.values[1] 
                height_img_second = df_enemies.y.values[1]
                height_bbox_second = df_enemies.h.values[1] / self.max_height_enemy
                width_bbox_second = df_enemies.w.values[1] / self.max_width_enemy

                x2_second = df_enemies.y.values[1] + (df_enemies.h.values[1] * 0.5)
                x2_second /= 0.92

        else:
            self.last_first_object_x = None 

        timestep_rel = self.timestep / dqn_simple_params.stop_reward
        first_object_is_flying = 1 if x2_first < 0.9 else 0
        second_object_is_flying = 1 if x2_second < 0.9 else 0


        #delta_first_object /= self.max_delta_object
        n_end_objects = self.df_objects.query("cls == 2").shape[0]
        if new_delta and n_end_objects == 0:
            self.delta_last_object = delta_first_object
        else:
            delta_first_object = self.delta_last_object      

        return np.array([timestep_rel, delta_first_object, dino_y, last_y_dino, is_down_dino, \
                         dist_x_first, height_img_first, height_bbox_first, width_bbox_first, first_object_is_flying,\
                         dist_x_second, height_img_second, height_bbox_second, width_bbox_second, second_object_is_flying])
    

    def get_state(self):
        done = False
        obs = self.observation 

        #check end game 
        n_end_objects = self.df_objects.query("cls == 2").shape[0]
        if n_end_objects > 0:
            done = True

        return obs, done 

    
    def step(self, action):
        self.timestep += 1 
        reward = 1

        if action == 0:
            #self.body.send_keys(Keys.DOWN)
            if not self.is_key_down:
                pyautogui.keyDown('down', _pause=False)
                #self.actions_driver.key_down(Keys.DOWN, self.body)
                #self.actions_driver.perform()
                self.is_key_down = True

            self.n_downs += 1
            #self.n_pass += 1 

        elif action == 1:
            if self.is_key_down:
                pyautogui.keyUp('down', _pause= False)
                self.is_key_down = False
                #self.actions_driver.key_up(Keys.DOWN, self.body)
                #self.actions_driver.perform()

            #self.body.send_keys(Keys.SPACE)
            pyautogui.press('space')
            self.n_ups += 1
        else:
            if self.is_key_down:
                pyautogui.keyUp('down', _pause= False)
                self.is_key_down = False
                #self.actions_driver.key_up(Keys.DOWN, self.body)
                #self.actions_driver.perform()

            self.n_pass += 1 

        if self.is_key_down:
            reward += 0.2

        obs, done = self.get_state()        
        
        #la idea es que solo se pueden tomar decisiones cuando dino está en el suelo
        #while not self.dino_is_in_floor and not done:
        #    obs, done = self.get_state()

        self.history_actions.append(action)

        if done:
            print('LOSE!')
            print(f"N ups: {self.n_ups} N pass: {self.n_pass} N downs: {self.n_downs}")
            reward = -10
            print(obs)
            if self.is_key_down:
                pyautogui.keyUp('down', _pause= False)
        else:
            self.history_obs.append(obs)

        return obs, reward, done, False

def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env = DinoEnv()
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init    


class PseudoCountRewardWrapper(gym.Wrapper):
    def __init__(self, env, hash_function = lambda o: o,
                 reward_scale: float = 1.0):
        super(PseudoCountRewardWrapper, self).__init__(env)
        self.hash_function = hash_function
        self.reward_scale = reward_scale
        self.counts = collections.Counter()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        extra_reward = self._count_observation(obs)
        print(extra_reward)
        return obs, reward + self.reward_scale * extra_reward, \
               done, info

    def _count_observation(self, obs) -> float:
        """
        Increments observation counter and returns pseudo-count reward
        :param obs: observation
        :return: extra reward
        """
        h = self.hash_function(obs)
        self.counts[h] += 1
        return np.sqrt(1/self.counts[h])


def process_batch_prio(batch_data, iteration):
    batch, batch_indices, batch_weights = batch_data
    optimizer.zero_grad()
    loss_v, new_prios = calc_loss_prio(batch,
                                       batch_weights,
                                       net,
                                       tgt_net.target_model,
                                       dqn_simple_params.gamma**dqn_simple_params.n_step,
                                       device=device,
                                       double=True)
    loss_v.backward()
    optimizer.step()
    buffer.update_priorities(batch_indices, new_prios)
        
    if iteration % dqn_simple_params.target_net_sync == 0:
        tgt_net.sync()
    
    return {
        "loss":loss_v.item(),
    }

def counts_hash(obs):
    delta_first_object, dino_y, _, is_down_dino, dist_x_first, height_img_first, height_bbox_first, width_bbox_first, dist_x_second, height_img_second, height_bbox_second, width_bbox_second = obs.tolist()
    delta_first_object_round = round(delta_first_object, 2)

    dino_y_round = round(dino_y, 2)

    dist_x_first_round = round(dist_x_first, 2)
    height_img_first_round = round(height_img_first, 2)
    height_bbox_first_round = round(height_bbox_first, 2)
    width_bbox_first_round = round(width_bbox_first, 2)

    dist_x_second_round = round(dist_x_second, 2)
    height_img_second_round = round(height_img_second, 2)
    height_bbox_second_round = round(height_bbox_second, 2)
    width_bbox_second_round = round(width_bbox_second, 2)

    return tuple([delta_first_object_round, dino_y_round, is_down_dino, dist_x_first_round, height_img_first_round, height_bbox_first_round, width_bbox_first_round, dist_x_second_round, height_img_second_round, height_bbox_second_round, width_bbox_second_round])


if __name__ == '__main__':
    """
    device = torch.device("cuda")
    env = DinoEnv(device)
    obs = env.reset()

    done = False 

    while not done:
        action = env.action_space.sample()
        print(action)
        obs, reward, done, _= env.step(action)
        print(obs)
    """
    device = torch.device("cuda")
    env = DinoEnv(device)
    #env = PseudoCountRewardWrapper(env, reward_scale=dqn_simple_params.counts_reward_scale, hash_function=counts_hash)
    writer = tensorboard.SummaryWriter(comment="-model_dino")

    net = DuelingDQN(env.observation_space.shape[0], 3).to(device)
    tgt_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.ArgmaxActionSelector()
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env,
                                                           agent,
                                                           gamma=dqn_simple_params.gamma,
                                                           steps_count=dqn_simple_params.n_step)
    
    buffer = ptan.experience.PrioReplayBufferNaive(exp_source, dqn_simple_params.replay_size)

    optimizer = optim.AdamW(net.parameters(), lr=dqn_simple_params.learning_rate)

    total_rewards = []
    frame_idx = 0
    ts_frame = 0 #punto de partida del frame para un episodio
    ts = time.time()
    best_mean_reward = -99999
    writer = tensorboard.SummaryWriter(comment=f"-{dqn_simple_params.run_name}")

    for batch in batch_generator(buffer, dqn_simple_params.replay_initial, dqn_simple_params.batch_size):
        frame_idx += 1
        for total_reward_episode, steps in exp_source.pop_rewards_steps():
            total_rewards.append(total_reward_episode)
            speed = (frame_idx - ts_frame)/(time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])

            print(f"Total frames: {frame_idx}, Total games: {len(total_rewards)}, Mean reward: {mean_reward:.2f}, speed f/s: {speed:.2f}")
            
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("mean_reward", mean_reward, frame_idx)
            writer.add_scalar("reward", total_reward_episode, frame_idx)
        
            #se guarda el modelo cuando se supera al anterior segun la media de rewards
            if int(best_mean_reward) < int(mean_reward):
                torch.save(net.state_dict(), f"models/dqn_best_ptan_noise_{int(mean_reward)}.pt")
                print(f"Best reward uppdated {int(best_mean_reward)} -> {int(mean_reward)}")

                best_mean_reward = int(mean_reward)

            if mean_reward > dqn_simple_params.stop_reward:
                print(f"Solved in {frame_idx} frames!")
                break
        
        dict_loss = process_batch_prio(batch, frame_idx)
        if frame_idx % 500 == 0:
            writer.add_scalar("loss", dict_loss['loss'], frame_idx)
            for layer_idx, sigma_l2 in enumerate(net.noisy_layers_sigma_snr()):
                writer.add_scalar(f"noise ratio layer {layer_idx}", sigma_l2, frame_idx)
