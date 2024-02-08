
import torch
from tensordict import TensorDict
from torchrl.data import BoundedTensorSpec
from torchrl.modules import SafeModule
from tensordict.nn import TensorDictModule


torch.manual_seed(0)
spec = BoundedTensorSpec(-torch.ones(3), torch.ones(3))
base_module = torch.nn.Linear(5, 3)
module = SafeModule(
    module=base_module, spec=spec, in_keys=["obs"], out_keys=["action"], safe=True
)
tensordict = TensorDict({"obs": torch.randn(5)}, batch_size=[])
print(module(tensordict)["action"])

tensordict = TensorDict({"obs": torch.randn(5) * 100}, batch_size=[])
print(module(tensordict)["action"])