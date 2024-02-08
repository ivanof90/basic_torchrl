import torch

from torchrl.modules import ConvNet, MLP
from torchrl.modules.models.utils import SquashDims
from tensordict import TensorDict


net = MLP(num_cells=[32, 64], out_features=4, activation_class=torch.nn.ELU)
print(net)
print(net(torch.randn(10, 3)).shape)

cnn = ConvNet(
    num_cells=[32, 64],
    kernel_sizes=[8, 4],
    strides=[2, 1],
    aggregator_class=SquashDims,
)
print(cnn)
print(cnn(torch.randn(10, 3, 32, 32)).shape)

#TensorDictModules

from tensordict.nn import TensorDictModule

tensordict = TensorDict({"key 1": torch.randn(10, 3)}, batch_size=[10])
module = torch.nn.Linear(3, 4)
td_module = TensorDictModule(module, in_keys=["key 1"], out_keys=["key 2"])
td_module(tensordict)
print(tensordict)

from tensordict.nn import TensorDictSequential

backbone_module = torch.nn.Linear(5, 3)
backbone = TensorDictModule(
    backbone_module, in_keys=["observation"], out_keys=["hidden"]
)
actor_module = torch.nn.Linear(3, 4)
actor = TensorDictModule(actor_module, in_keys=["hidden"], out_keys=["action"])
value_module = MLP(out_features=1, num_cells=[4, 5])
value = TensorDictModule(value_module, in_keys=["hidden", "action"], out_keys=["value"])

sequence = TensorDictSequential(backbone, actor, value)
print(sequence)

tensordict = TensorDict(
    {"observation": torch.randn(3, 5)},
    [3],
)
sequence(tensordict)
print(tensordict)