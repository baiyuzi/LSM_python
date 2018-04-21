import math 
import random

inp_size = 2 
cur_neu_size = 100
read_size = 2
inp_conn = 8
res_conn = 100
v_rest = -65
c_th = 5
delta_c = 3
delta_w = 0.01
sigma = 1
v_th = 30
v_leak = -5
tao_c = 64
I_pos = 10
I_neg = -10
p_pos = 0.5
p_neg = 0.5

class neuron:
    def __init__(self, w, delta_w, state, v_mem, c, I_inj):
        self.w=w
        self.delta_w=delta_w
        self.state=state
        self.v_mem=v_mem
        self.c=c
        self.I_inj=I_inj

def w_gen(type, i, j): # weight generation rule given neurons' adress
    if type==0: # fixed value generation
        weight=0.5
    elif type==1: # generation according to distance
        if i==j:
            weight=0
        else:
            weight=(1/2)/abs(i-j)
    elif type==2: # generation in a range
        weight=random.uniform(0.25, 0.75)
    return weight

def update_prob(p):
    if random.random() < p:
        delta_w_temp=delta_w
    else:
        delta_w_temp=0
    return delta_w_temp

def neu_list_init(pre_neu_size, pre_cur_conn, cur_neu_size, cur_cur_conn, pre_cur_type, cur_cur_type):
    neu_list=[]
    for i in range(cur_neu_size): # create an "empty" list with number of cur_neu_size neurons
        w=[]
        delta_w=[]
        state=[]
        v_mem=v_rest
        c=c_th
        I_inj=0
        neu_list.append(neuron(w, delta_w, state, v_mem, c, I_inj))
    if cur_cur_conn==0 and pre_cur_conn==0: # input layer generation
        for i in range(cur_neu_size):
            neu_list[i].w.append(0)
            neu_list[i].state.append(0)
        return neu_list
    if cur_cur_conn!=0:
        self_w_row=cur_neu_size
        for i in range(self_w_row): # initialize reservoir to reservoir weight
            list_self_conn=range(cur_neu_size)
            list_self_conn=[]
            for k in range(cur_neu_size):
                if k!=i:
                    list_self_conn.append(k)
            list_temp=random.sample(list_self_conn, cur_cur_conn)
            for j in range(self_w_row):
                if j in list_temp:
                    neu_list[j].w.append(w_gen(cur_cur_type, j, i))
                else:
                    neu_list[j].w.append(0)
                neu_list[j].delta_w.append(0)
    else:
        self_w_row=0
    for i in range(self_w_row, self_w_row + pre_neu_size): # initialize input to reservoir weight
        list_temp=random.sample(range(cur_neu_size), pre_cur_conn)
        for j in range(cur_neu_size):
            if j in list_temp:
                neu_list[j].w.append(w_gen(pre_cur_type, j, i))
            else:
                neu_list[j].w.append(0)
            neu_list[j].delta_w.append(0)
    for i in range(cur_neu_size):
        neu_list[i].state.append(0)
    return neu_list

def poss(list, data):
    for i in range(len(list)):
        p_poss=data[i]
        list[i].state.append(int(random.random() < p_poss))
    return list

def LIF(pre_neu_list, cur_neu_list, step):
    for i in range(len(cur_neu_list)): # calculating v_mem updating
        if len(cur_neu_list[i].w)==len(cur_neu_list) + len(pre_neu_list):
            lif_self_row=len(cur_neu_list)
            for j in range(lif_self_row):
                cur_neu_list[i].v_mem=cur_neu_list[i].v_mem + \
                cur_neu_list[j].state[step-1] * cur_neu_list[i].w[j]
        elif len(cur_neu_list[i].w)==len(pre_neu_list):
            lif_self_row=0
        for j in range(lif_self_row, lif_self_row + len(pre_neu_list)):
            cur_neu_list[i].v_mem=cur_neu_list[i].v_mem + \
                pre_neu_list[j-lif_self_row].state[step] * cur_neu_list[i].w[j]
        cur_neu_list[i].v_mem=cur_neu_list[i].v_mem - v_leak + cur_neu_list[i].I_inj
        if(cur_neu_list[i].v_mem>=v_th):
            cur_neu_list[i].state.append(1)
            cur_neu_list[i].v_mem=v_rest
        else: cur_neu_list[i].state.append(0)
        cur_neu_list[i].c=cur_neu_list[i].c - \
        cur_neu_list[i].c/tao_c + cur_neu_list[i].state[step] # updating cal concentration
    return cur_neu_list

def teacher(list, label):
    for i in range(len(list)):
        if label[i]==1:
            if list[i].c > c_th and list[i].c < c_th + sigma:
                list[i].I_inj=I_pos
        elif label[i]==0:
            if list[i].c > c_th - sigma and list[i].c < c_th:
                list[i].I_inj=I_neg
        else: 
            list[i].I_inj=0
    return list

def delta_w_gen(pre_neu_list, cur_neu_list, step): # generate delta_w for each step
    for i in range(len(cur_neu_list)):
        if len(cur_neu_list[i].w)==len(cur_neu_list)+len(pre_neu_list):
            dwg_self_row=0
            for j in range(len(cur_neu_list)):
                if cur_neu_list[j].state[step-1]==1 and cur_neu_list[i].w[j]!=0:
                    if cur_neu_list[i].c > c_th and cur_neu_list[i].c < c_th + delta_c:
                        cur_neu_list[i].delta_w[j]=update_prob(p_pos)
                    elif cur_neu_list[i].c > c_th - delta_c and cur_neu_list[i].c < c_th:
                        cur_neu_list[i].delta_w[j]=-update_prob(p_neg)
        elif len(cur_neu_list[i].w)==len(pre_neu_list):
            dwg_self_row=0
        for j in range(dwg_self_row, dwg_self_row + len(pre_neu_list)):
            if pre_neu_list[j - dwg_self_row].state[step]==1 and cur_neu_list[i].w[j]!=0:
                if cur_neu_list[i].c > c_th and cur_neu_list[i].c < c_th + delta_c:
                    cur_neu_list[i].delta_w[j]=update_prob(p_pos)
                elif cur_neu_list[i].c > c_th - delta_c and cur_neu_list[i].c < c_th:
                    cur_neu_list[i].delta_w[j]=-update_prob(p_neg)
    return cur_neu_list

def w_update(list): # update the weight of the neuron net
    for i in range(len(list)):
        for j in range(len(list[i].w)):
            list[i].w[j]=list[i].w[j] + list[i].delta_w[j]
    return list

def train(pre_neu_list, cur_neu_list, step, label):
    cur_neu_list=delta_w_gen(pre_neu_list, cur_neu_list, step)
    cur_neu_list=w_update(cur_neu_list)
    cur_neu_list=teacher(cur_neu_list, label)
    cur_neu_list=LIF(pre_neu_list, cur_neu_list, step)
    return cur_neu_list

def data_gene(sample):
    list=[]
    for i in range(sample):
        x1=random.random()
        x2=random.random()
        label=int(x1 + x2 > 0.5)
        list.append([x1, x2, label])
    return list