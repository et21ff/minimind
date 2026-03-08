import torch

import torch.nn as nn

dropout_layer = nn.Dropout(p=0.5)

# Dropout expects floating-point tensors.
t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

print(dropout_layer(t))
t = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
linear_layer = nn.Linear(in_features=3, out_features=5, bias=True)

print(linear_layer(t))

t = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])

t_view1 = t.view(3, 4)
print(t_view1)
t_view2 = t.view(4, 3)
print(t_view2)

x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(torch.triu(x, diagonal=0))

print(torch.triu(x, diagonal=1))

x = torch.arange(1, 7)
y = torch.reshape(x, (2, 3))
print(y)
z = torch.reshape(x, (3, -1))
print(z)
