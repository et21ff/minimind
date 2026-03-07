import torch

# x = torch.tensor([1,2,3,4,5])
# y = torch.tensor([10,20,30,40,50])

# condition = x>3

# result = torch.where(x>3,x,y)
# # tensor([10, 20, 30,  4,  5])
# print(result)

# t1 = torch.arange(0,10,2) #(start,end,step_len)
# print(t1) # tensor([0, 2, 4, 6, 8])
# t2 = torch.arange(5,0,-1)
# print(t2) # tensor([5, 4, 3, 2, 1])

# v1 = torch.tensor([1,2,3])
# v2 = torch.tensor([4,5,6])
# # result = torch.outer(v1,v2)
# print(torch.outer(v1,v2))
# # tensor([[ 4,  5,  6],
# #         [ 8, 10, 12],
# #         [12, 15, 18]])

# t1 = torch.ones(2,2,3)
# t2 = torch.ones(2,2,3)
# print(t1.shape)
# print(torch.cat((t1,t2),dim=0).shape)
# print(torch.cat((t1,t2),dim=1).shape)
# print(torch.cat((t1,t2),dim=-1).shape)

# t1 = torch.tensor([1,2,3])
# print(t1.shape)
# print(t1.unsqueeze(0).shape)