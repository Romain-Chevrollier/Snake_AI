import torch
import random
import numpy as np
import cv2
import pygame
import os
import glob
import shutil
import argparse
from moviepy.editor import ImageSequenceClip
from collections import deque
from game import SnakeGameAI, Direction, Point, font
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def set_model(self, new_model):
        self.model = new_model

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)[0]) or 
            (dir_l and game.is_collision(point_l)[0]) or 
            (dir_u and game.is_collision(point_u)[0]) or 
            (dir_d and game.is_collision(point_d)[0]),

            # Danger right
            (dir_u and game.is_collision(point_r)[0]) or 
            (dir_d and game.is_collision(point_l)[0]) or 
            (dir_l and game.is_collision(point_u)[0]) or 
            (dir_r and game.is_collision(point_d)[0]),

            # Danger left
            (dir_d and game.is_collision(point_r)[0]) or 
            (dir_u and game.is_collision(point_l)[0]) or 
            (dir_r and game.is_collision(point_u)[0]) or 
            (dir_l and game.is_collision(point_d)[0]),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self, n_games):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones, n_games)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done, n_games):
        self.trainer.train_step(state, action, reward, next_state, done, n_games)

    def get_action_train(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
    def get_action_test(self, state, model):
        self.model.load_state_dict(model)
        final_move = [0,0,0]
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1

        return final_move

def main():
    args = parse_arguments()

    if args.train:
        train()
    else:
        if args.record:
            test(record())
        else:
            test()
        

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train or simulate the agent.')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--test', action='store_true', help='Test the agent')
    parser.add_argument('--record', action='store_true', help='Record the game')
    return parser.parse_args()

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    recording = False
    i=0
    files = glob.glob('output_image/*')
    for f in files:
        os.remove(f)
    while True:
        # get old state
        state_old = agent.get_state(game)
        
        # get move
        final_move = agent.get_action_train(state_old)
        
        # perform move and get new state
        reward, done, score = game.play_step(final_move, agent.n_games + 1)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done, agent.n_games)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if recording == False:
            # Initialize video recording
            recording = True

        if recording:
            if isinstance(game.display, pygame.Surface):
                frame = cv2.cvtColor(pygame.surfarray.array3d(game.display), cv2.COLOR_RGB2BGR)
                cv2.imwrite("output_image/frame_" + str(i).zfill(4) + ".png", frame)
                i+=1


        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory(agent.n_games)
            recording = False
            i=0
            if score > record:
                record = score
                agent.model.save()
                clip = ImageSequenceClip("output_image", fps=10)
                clip.write_videofile("output_video_training/video_" + "score_" + str(score) + ".mp4")
             
            files = glob.glob('output_image/*')
            for f in files:
                os.remove(f)

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)

def record():
    return True

def test(recording=False):
    model = torch.load("model/model.pth", map_location=torch.device("cuda:0"))
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    i=0
    files = glob.glob('output_image/*')
    for f in files:
        os.remove(f)

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action_test(state_old, model)
        _, done, score = game.play_step(final_move, agent.n_games + 1)

        if recording:
            if isinstance(game.display, pygame.Surface):
                frame = cv2.cvtColor(pygame.surfarray.array3d(game.display), cv2.COLOR_RGB2BGR)
                cv2.imwrite("output_image/frame_" + str(i).zfill(4) + ".png", frame)
                i+=1

        if done:
            agent.n_games += 1
            if recording:
                clip = ImageSequenceClip("output_image", fps=30)
                clip.write_videofile("output_video_test/video_" + "score_" + str(score) + ".mp4")

            if score > record:
                record = score
  
            files = glob.glob('output_image/*')
            for f in files:
                os.remove(f)

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            break

if __name__ == '__main__':
    main()