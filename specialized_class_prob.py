
import torch
from tensordict import TensorDict
from torchrl.data import BoundedTensorSpec
from torchrl.modules import SafeModule
from tensordict.nn import TensorDictModule
from torchrl.modules import Actor
from torchrl.modules import NormalParamWrapper, TanhNormal
from tensordict.nn import (
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
)


base_module = torch.nn.Linear(5, 3)
actor = Actor(base_module, in_keys=["obs"])
tensordict = TensorDict({"obs": torch.randn(5)}, batch_size=[])
print(actor(tensordict))# action is the default value



td = TensorDict({"input": torch.randn(3, 5)}, [3])
net = NormalParamWrapper(torch.nn.Linear(5, 4))  # splits the output in loc and

td = TensorDict({"input": torch.randn(3, 5)}, [3])
net = NormalParamWrapper(torch.nn.Linear(5, 4))  # splits the output in loc and scale
module = TensorDictModule(net, in_keys=["input"], out_keys=["loc", "scale"])
td_module = ProbabilisticTensorDictSequential(
    module,
    ProbabilisticTensorDictModule(
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        return_log_prob=False,
    ),
)
td = td_module(td)
print(td)
