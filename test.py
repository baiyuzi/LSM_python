import LSM_LIB
import math
import random
import matplotlib.pyplot as plt

step_total = 1

data=LSM_LIB.data_gene(1000)
sample=data[0]
print(sample)
net_in=LSM_LIB.neu_list_init(0, 0, 2, 0, 0, 0)
net_res=LSM_LIB.neu_list_init(2, 1, 4, 2, 0, 1)
for i in range(len(net_res)):
    print(net_res[i].state)

for i in range(step_total):
    net_in=LSM_LIB.poss(net_in, sample)
    net_res=LSM_LIB.LIF(net_in, net_res, i)
    # for j in range(len(net_res)):
    #     print(net_res[j].w)

for i in range(len(net_in)):
    print(net_in[i].state)
for i in range(len(net_res)):
    print(net_res[i].state)
for i in range(len(net_res)):
    print(net_res[i].v_mem)
# x=[]
# y=[]
# for i in range(step_total):
#     x.append([i] * len(net_res))
#     for j in range(len(net_res)):
#         if net_res[j].state[i]==1:
#             y.append(j)
#         else:
#             y.append(0)
# plt.scatter(x, y)
# plt.show()


# net=LSM_LIB.neu_list_init(0, 0, 2, 0, 0, 0)
# net=net_in
# for i in range(len(net)):
#     # print(net[i].w)
#     # print(net[i].delta_w)
#     print(net[i].state) 


