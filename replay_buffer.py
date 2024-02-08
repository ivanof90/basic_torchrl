import torch
from torchrl.data import PrioritizedReplayBuffer, ReplayBuffer
from tensordict import TensorDict




rb = ReplayBuffer(collate_fn=lambda x: x) # collate function: in order to pre-process before to be added to the buffer
rb.add(1)
print(rb.sample(1))

rb.extend([2, 3]) # add more elements to the buffer, and suffle de list
print(rb.sample(1))
print(rb.sample(1))
print(rb.sample(batch_size=3)) # return a especified number of elements from the buffer



#The PrioritizedReplayBuffer class in torchrl is a specialized replay buffer that implements prioritized experience replay (PER).
#It provides a mechanism for storing and sampling transitions from a reinforcement learning environment,
# with the prioritization based on the estimated importance of each transition.
#The PrioritizedReplayBuffer class inherits from the ReplayBuffer class,
# which provides the basic functionality for storing and sampling transitions.

rb = PrioritizedReplayBuffer(alpha=0.7, beta=1.1, collate_fn=lambda x: x)
rb.add(1)
rb.sample(1)
rb.update_priority(1, 0.5)

#Replay buffer with tensordics

collate_fn = torch.stack
rb = ReplayBuffer(collate_fn=collate_fn)
rb.add(TensorDict({"a": torch.randn(3)}, batch_size=[]))
print(len(rb))

rb.extend(TensorDict({"a": torch.randn(2, 3)}, batch_size=[2]))
print(len(rb))
print(rb.sample(10))
print(rb.sample(2).contiguous()) # values in contiguous memory address





