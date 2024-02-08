import torch
from tensordict import TensorDict

#Tensordict
batch_size = 5
tensordict = TensorDict(
    source={
        "key 1": torch.zeros(batch_size, 3),
        "key 2": torch.zeros(batch_size, 5, 6, dtype=torch.bool),
    },
    batch_size=[batch_size],
)
print(tensordict)
print("tensordict at position ",4)
print(tensordict[4])
print("tensordict´s key: key 1. All the records with size batch_size")
print(tensordict["key 1"])
print("data at position ",4, " with key: key1", tensordict[4]["key 1"])


td = TensorDict({'a': torch.zeros(3,4,5)}, batch_size=[3, 4])
print(td)
print(td.shape)

# returns a TensorDict of batch size [3, 4, 1]
td_unsqueeze = td.unsqueeze(-1)
print("td_unsqueeze, add an aditional dimension to the size", td_unsqueeze.shape)
td_view = td.view(-1)
print("td_view, flatenea el tamaño del tensor, 3X4 = 12: ", td_view.shape)
# returns a tensor of batch size [12, 4]
a_view = td_view.get("a") # se size 12,5, porque se flateneo el batch_size (dos primeras dimensiones)
#luego pues el queda la ultima dimension, 5
print("a_view: ", a_view.shape)

print("to device: ", td.to("cpu"))




