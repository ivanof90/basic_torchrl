from torchrl.envs.utils import step_mdp
import torch
import gymnasium as gym
from torchrl.envs.libs.gym import GymEnv, GymWrapper
from torchrl.modules import SafeModule
from tensordict import TensorDict



env = GymEnv("Pendulum-v1")

action_spec = env.action_spec
actor_module = torch.nn.Linear(3, 1)
actor = SafeModule(
    actor_module, spec=action_spec, in_keys=["observation"], out_keys=["action"]
)
torch.manual_seed(0)
env.set_seed(0)

max_steps = 1
tensordict = env.reset()
tensordicts = TensorDict({}, [max_steps])
for i in range(max_steps):
    actor(tensordict)
    tensordicts[i] = env.step(tensordict)
    if tensordict["done"].any():
        break
    print(tensordict)
    tensordict = step_mdp(tensordict)  # roughly equivalent to obs = next_obs
    print(tensordict)

tensordicts_prealloc = tensordicts.clone()
#print("total steps:", i)
#print(tensordicts)