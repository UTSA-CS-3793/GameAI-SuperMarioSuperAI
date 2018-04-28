import numpy as np
import gym
import gym_rle

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, Dropout, Permute
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy ,EpsGreedyQPolicy#BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

from PIL import Image

FILE_NAME='duel_dqn_SuperMarioWorld-v0_weights.h5f'
ENV_NAME = 'SuperMarioWorld-v0'
INPUT_SHAPE = (224, 256)#Size of image given by observation
WINDOW_LENGTH = 4

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n #number of possible action actor can take

#Custom Processor to make this actually work
class MarioProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        processed_observation = processed_observation.T
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)



# Next, we build a very simple model regardless of the dueling architecture
# if you enable dueling network in DQN , DQN will build a dueling network base on your model automatically
# Also, you can build a dueling network by yourself and turn off the dueling network in DQN.
input_shape = (WINDOW_LENGTH,)+INPUT_SHAPE

model = Sequential()
model.add(Permute((2,3,1), input_shape=input_shape))
model.add(Convolution2D(32, 8, 8, subsample=(4, 4)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())


# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=10000, window_length=WINDOW_LENGTH)
processor = MarioProcessor()
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=1000)
#policy = BoltzmannQPolicy(tau=1.)
# enable the dueling network
# you can specify the dueling_type to one of {'avg','max','naive'}
#dqn = DQNAgent(model=model, nb_actions=nb_actions, processor = processor, memory=memory, nb_steps_warmup=10,
#               enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory, enable_dueling_network=True, dueling_type='avg',
               processor=processor, nb_steps_warmup=10, gamma=.99, target_model_update=1000,train_interval=4, delta_clip=1.)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])

#This is okay for now, but put an if statement here after this(Unless the command below is commented out)
#dqn.load_weights('duel_dqn_SuperMarioWorld-v0_weights.h5f')

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=70000, visualize=False, verbose=1, log_interval=1000)

# After training is done, we save the final weights.
#dqn.save_weights('duel_dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=False)


# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=500)


#old model
'''
model.add(Conv2D(32, 3, activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))


#If it stops at 80k again then use this:
dqn.load_weights(FILE_NAME)
dqn.fit()
dqn.save_weights(FILE_NAME, overwrite=False)
dqn.test()
'''
