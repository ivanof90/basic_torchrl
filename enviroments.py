from torchrl.envs.libs.gym import GymEnv, GymWrapper
import gymnasium as gym
import matplotlib.pyplot as plt


gym_env = gym.make("Pendulum-v1")
env = GymWrapper(gym_env)
env = GymEnv("Pendulum-v1")

tensordict = env.reset()
print(tensordict)
tensordict = env.rand_step(tensordict)
print(tensordict)

env = GymEnv("Pendulum-v1", frame_skip=3, from_pixels=True, pixels_only=False)
tensordict = env.reset()
print(tensordict) # ahora vienen los pixeles
img = tensordict["pixels"].numpy()
plt.imshow(img)
plt.show()

