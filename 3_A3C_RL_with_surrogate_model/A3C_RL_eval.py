import torch
from collections import OrderedDict
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import lifelines
import multiprocessing as mp
import os

# set seed
seed = 20240712
np.random.seed(seed)
torch.manual_seed(seed)  

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)
else:
    device = torch.device('cpu')

predict_dt = 7 # 7, 14, 21 or 28

save_folder = f'A3C_RL_train/predict_dt={predict_dt}' # RL model save folder


# the log-rank test is used to compare the optimal RL treatment 
# with 'CSF1R_I & IGF1R_I' combination treatment.
eval_save_folder = f'A3C_eval_vs_combination/predict_dt={predict_dt}' # evaluation save folder

# deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        self.depth = len(layers) - 1
        self.activation = torch.nn.Tanh() # activation tanh
        layer_list = []

        # add layers with activation
        for i in range(self.depth - 1):
            linear_layer = torch.nn.Linear(layers[i], layers[i + 1])
            # initialize weights with Xavier normal initialization
            torch.nn.init.xavier_normal_(linear_layer.weight, gain=np.sqrt(2))
            # initialize biases to zero
            torch.nn.init.constant_(linear_layer.bias, 0)
            # Append the layer and activation to the list
            layer_list.append(('layer_%d' % i, linear_layer))
            layer_list.append(('activation_%d' % i, self.activation))

        # add the final linear layer without activation
        final_linear_layer = torch.nn.Linear(layers[-2], layers[-1])
        torch.nn.init.xavier_normal_(final_linear_layer.weight, gain=np.sqrt(2))
        torch.nn.init.constant_(final_linear_layer.bias, 0)
        layer_list.append(('layer_%d' % (self.depth - 1), final_linear_layer))

        # Convert list to OrderedDict and create the sequential model
        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        return self.layers(x)

# PINN based surrogate model used for reinfocement learning
class RL_dNN():
    def __init__(self, FP_layers, lb, ub, dc, dt_cyto): 
        # ode parameters
        self.q_c = 0.8
        self.eta_c = 2e-3 
        self.mu_c = 2e-3 
        self.q_i = 0.8
        self.eta_i = 2e-3 
        self.mu_i = 2e-3         
        # fokker-planck parameters
        #load parameters
        additional_params = torch.load('surrogate_model/additional_params.pth')  
        initial_log_p_cyto = additional_params['log_p_cyto']
        self.log_p_cyto  = torch.nn.Parameter(initial_log_p_cyto).to(device)        
        # deep neural networks
        self.dnn_TC = DNN(FP_layers).to(device)
        para_TC = torch.load('surrogate_model/dnn_TC_model1.pth')
        self.dnn_TC.load_state_dict(para_TC, strict=False)        
        
        # other parameters
        self.lb = torch.tensor(lb, dtype=torch.float32, requires_grad=True).float().to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32, requires_grad=True).float().to(device)
        self.c_max = torch.tensor(1/dc, dtype=torch.float32, requires_grad=True).float().to(device)
        self.dt_cyto = dt_cyto
        self.CSF1RI_cum_max = 200 / self.dt_cyto
  
    # DNN for Fokker-Planck equation of tumor cell
    def net_u_CC(self, C, t, dose_c, dose_cum, dose_I): 
        t = 2.0*(t- self.lb[1])/(self.ub[1] - self.lb[1]) - 1.0 
        C = 2.0*(C- self.lb[0])/(self.ub[0] - self.lb[0]) - 1.0
        dose_c = 2.0*(dose_c - 0) - 1.0
        dose_cum = 2.0*(dose_cum - 0)/self.CSF1RI_cum_max - 1.0
        dose_I = 2.0*(dose_I - 0) - 1.0
        HTC = torch.cat([t, C, dose_c, dose_cum, dose_I], dim=1)
        p = self.dnn_TC(HTC)
        return p
    
    # ODE solution of CSF1R_I
    def f_CSF1RI(self, c_CSF1RI_0, dose_c, interval):
        itera_cyto = int(interval/self.dt_cyto)
        c = torch.empty(itera_cyto)
        a = torch.exp(self.log_p_cyto[10])*self.q_c
        b = torch.exp(self.log_p_cyto[11])*self.eta_c + torch.exp(self.log_p_cyto[12])*self.mu_c #d_M1 + d_M2 + d_M0 = 1 
        c[0] = (c_CSF1RI_0 - a*dose_c/(a+b))*torch.exp(-(a+b)*self.dt_cyto) + a*dose_c/(a+b) 
        for k in range(itera_cyto-1):
            c[k+1] = (c[k] - a*dose_c/(a+b))*torch.exp(-(a+b)*self.dt_cyto) + a*dose_c/(a+b)        
        return c

    # ODE solution of IGF1R_I
    def f_IGF1RI(self, c_IGF1RI_0, dose_I, C_T, interval):
        itera_cyto = int(interval/self.dt_cyto)
        c = torch.empty(itera_cyto)
        a = torch.exp(self.log_p_cyto[13])*self.q_i
        b = torch.exp(self.log_p_cyto[14])*self.eta_i + torch.exp(self.log_p_cyto[15])*self.mu_i * C_T
        c[0] = (c_IGF1RI_0 - a*dose_I/(a+b))*torch.exp(-(a+b)*self.dt_cyto) + a*dose_I/(a+b)
        for k in range(itera_cyto-1):
            c[k+1] = (c[k] - a*dose_I/(a+b))*torch.exp(-(a+b)*self.dt_cyto) + a*dose_I/(a+b)        
        return c
    
    # predict Fokker-Planck equation of TC for RL
    def predict_DRL_FP(self, c, t, dose_c, CSF1RI_cum, dose_I, c_CSF1RI_0, c_IGF1RI_0, c_T_0, interval):
        self.dnn_TC.eval()  

        c = torch.tensor(c, dtype=torch.float32, requires_grad=True).float().to(device)
        t = torch.tensor(t, dtype=torch.float32, requires_grad=True).float().to(device)
        t_k = t * torch.ones_like(c)

        c_CSF1RI_0 = torch.tensor(c_CSF1RI_0, dtype=torch.float32, requires_grad=True).float().to(device)
        c_IGF1RI_0 = torch.tensor(c_IGF1RI_0, dtype=torch.float32, requires_grad=True).float().to(device)
        c_T_0 = torch.tensor(c_T_0, dtype=torch.float32, requires_grad=True).float().to(device)

        dose_c = torch.tensor(dose_c, dtype=torch.float32, requires_grad=True).float().to(device)
        CSF1RI_cum = torch.tensor(CSF1RI_cum, dtype=torch.float32, requires_grad=True).float().to(device)
        dose_I = torch.tensor(dose_I, dtype=torch.float32, requires_grad=True).float().to(device)
        
        c_CSF1RI = self.f_CSF1RI(c_CSF1RI_0, dose_c, interval)
        c_IGF1RI = self.f_IGF1RI(c_IGF1RI_0, dose_I, c_T_0, interval)
        CSF1RI_cum = CSF1RI_cum + torch.sum(c_CSF1RI)

        c_CSF1RI_k = c_CSF1RI[-1] * torch.ones_like(c)
        CSF1RI_cum_k = CSF1RI_cum * torch.ones_like(c)
        c_IGF1RI_k = c_IGF1RI[-1] * torch.ones_like(c)

        uTC_pred = self.net_u_CC(c, t_k, c_CSF1RI_k, CSF1RI_cum_k, c_IGF1RI_k)

        uTC_pred = uTC_pred.detach().cpu().numpy()
        CSF1RI_cum = CSF1RI_cum.detach().cpu().numpy()
        c_CSF1RI_k = c_CSF1RI_k[-1, 0].detach().cpu().numpy()
        c_IGF1RI_k = c_IGF1RI_k[-1, 0].detach().cpu().numpy()

        uTC_pred[uTC_pred < 0] = 0

        return uTC_pred, CSF1RI_cum, c_CSF1RI_k, c_IGF1RI_k
    
# initialize parameters of policy net and value net
def normalized_columns_initializer(weights, std):
    with torch.no_grad():
        out = torch.randn_like(weights)
        out *= std / torch.sqrt(out.pow(2).sum(dim=0, keepdim=True))
        return out

# long short-term memory (LSTM) net
class LSTMNet(torch.nn.Module):
    def __init__(self, input_size, lstm_hidden_dim, num_layers):
        super(LSTMNet, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, lstm_hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        lstm_out, _ = self.lstm(x)
        hidden_size = lstm_out.shape[2]
        lstm_out_flattened = lstm_out.view(batch_size * seq_len, hidden_size)
        return lstm_out_flattened

# policy net, actor               
class PolicyNet(torch.nn.Module):
    def __init__(self, lstm_hidden_dim, action_size):
        super(PolicyNet, self).__init__()
        # fully connected layers
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(lstm_hidden_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, action_size)
        )
        for m in self.fc_layers:
            if isinstance(m, torch.nn.Linear):
                m.weight.data = normalized_columns_initializer(m.weight.data, std=0.01)

    def forward(self, x):
        fc_out = self.fc_layers(x)
        policy_dist = torch.nn.functional.softmax(fc_out, dim=1)
        return policy_dist

# value net, critic 
class ValueNet(torch.nn.Module):
    def __init__(self, lstm_hidden_dim):
        super(ValueNet, self).__init__()
        # fully connected layers
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(lstm_hidden_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1)
        )
        for m in self.fc_layers:
            if isinstance(m, torch.nn.Linear):
                m.weight.data = normalized_columns_initializer(m.weight.data, std=1.0)

    def forward(self, x):
        value_estimate = self.fc_layers(x)
        return value_estimate.squeeze(1)

# RL environmen       
class Env:
    def __init__(self, model, initial):  
        self.state = [] # state of RL environment
        self.action_c_space = np.array([0.0, 1.0]) # action space of CSF1R_I dose 
        self.action_I_space = np.array([0.0, 1.0]) # action space of IGF1R_I dose   
        self.surrogat_model = model 
        self.initial = initial
        self.predict_CSF1RI_list = []
        self.predict_CSF1RI_list.append(self.initial[0])
        self.predict_IGF1RI_list = []
        self.predict_IGF1RI_list.append(self.initial[1])
        self.predict_TC_list = []
        self.predict_TC_list.append(self.initial[2])
        self.predict_CSF1RI_cum = 0
        self.CSF1RI_dose_cum = 0
        self.last_state_survival = 0
        self.time = 0

    # reset RL envrionment
    def reset(self):  
        self.state = []
        self.predict_CSF1RI_list = []
        self.predict_CSF1RI_list.append(self.initial[0])
        self.predict_IGF1RI_list = []
        self.predict_IGF1RI_list.append(self.initial[1])
        self.predict_TC_list = []
        self.predict_TC_list.append(self.initial[2])
        self.predict_CSF1RI_cum = 0
        self.CSF1RI_dose_cum = 0
        self.last_state_survival = 0
        self.time = 0

    # simulate each step in RL envrionment and calculate reward
    def step(self, action_c, action_I, epi, predict_dt, cc, lamda):  
        last_state = self.state.copy()
        done = False
        self.time += predict_dt
        dose_c = self.action_c_space[action_c]
        dose_I = self.action_I_space[action_I]
        self.CSF1RI_dose_cum += dose_c * predict_dt
        c_CSF1RI_0 = self.predict_CSF1RI_list[-1]
        c_IGF1RI_0 = self.predict_IGF1RI_list[-1]
        c_T_0 = self.predict_TC_list[-1]

        # surrogate model 
        rl_FP_TC, c_CSF1RI_cum, c_CSF1RI, c_IGF1RI = self.surrogat_model.predict_DRL_FP(cc, self.time, dose_c, self.predict_CSF1RI_cum, dose_I, 
                                                   c_CSF1RI_0, c_IGF1RI_0, c_T_0, predict_dt)

        # normalization
        rl_FP_TC_norm = rl_FP_TC/(np.sum(rl_FP_TC) + 1e-7)

        self.predict_CSF1RI_cum = c_CSF1RI_cum
        self.predict_CSF1RI_list.append(c_CSF1RI)
        self.predict_IGF1RI_list.append(c_IGF1RI)
        self.predict_TC_list.append(np.sum(rl_FP_TC_norm*cc))

        prob_sum = np.sum(rl_FP_TC)
        survival_prob = np.sum(rl_FP_TC[:-3, :]) / (prob_sum + 1e-7)    
        death_prob = 1 - survival_prob
        cure_prob = np.sum(rl_FP_TC[:1, :]) / (prob_sum + 1e-7)

        # punish no treatment in high death probability case
        dose_punish = (self.action_c_space.shape[0] - 1) + 1e-7

        # set thresholds
        death_threshold = 0.2
        cure_threshold = 0.99

        # calculate the change in survival probability
        state_survival = (survival_prob - (1 - death_threshold)) / death_threshold
        if epi == 0:
            survival_prob_delta = 0
        else:
            survival_prob_delta = state_survival - self.last_state_survival
        
        # update state of RL environment
        self.state.append([state_survival, survival_prob_delta])
        self.last_state_survival = state_survival
        now_state = self.state.copy()

        # reward function
        reward = 0.1
        if death_prob >= death_threshold:
            reward = -0.1
            done = True
        elif cure_prob >= cure_threshold:
            reward = reward + 1.0
            done = True
        if not done:
            if death_prob >= death_threshold * 0.5:
                reward = reward - 0.1 * (dose_punish - action_c)
            elif action_c == 0:
                reward = reward + 0.05

            if survival_prob <= 0.9:
                reward = reward - 0.1 * (dose_punish - action_I)
            elif action_I == 0:
                reward = reward + 0.05

            if survival_prob > 0.9:
                reward = reward + lamda * (self.time - self.CSF1RI_dose_cum)    

        return last_state, reward, done, now_state

# load A3C RL net
class A3C_RL_net:
    def __init__(self, lamda, state_size, lstm_hidden_dim, num_layers, action_c_size, action_I_size):
        str_lamda = str(lamda).replace('.', '_')

        self.lstm = LSTMNet(state_size, lstm_hidden_dim, num_layers).to(device)
        para_lstm = torch.load(f'{save_folder}/global_LSTM_{str_lamda}.pth')      
        self.lstm.load_state_dict(para_lstm, strict=False)

        self.actor_c = PolicyNet(lstm_hidden_dim, action_c_size).to(device)
        para_actor_c = torch.load(f'{save_folder}/global_actor_c_{str_lamda}.pth')      
        self.actor_c.load_state_dict(para_actor_c, strict=False)

        self.actor_I = PolicyNet(lstm_hidden_dim, action_I_size).to(device)
        para_actor_I = torch.load(f'{save_folder}/global_actor_I_{str_lamda}.pth')      
        self.actor_I.load_state_dict(para_actor_I, strict=False)

        self.critic = ValueNet(lstm_hidden_dim).to(device)
        para_critic = torch.load(f'{save_folder}/global_critic_{str_lamda}.pth')      
        self.critic.load_state_dict(para_critic, strict=False)

# evaluation
def A3C_RL_eval_worker(env, iteration, cc, state_size, lstm_hidden_dim, num_layers, predict_time, predict_dt, pre_therapy_time, lamda):
    str_lamda = str(lamda).replace('.', '_')
    action_c_size = env.action_c_space.shape[0]
    action_I_size = env.action_I_space.shape[0]
    
    # initialize A3C RL net
    rl_model = A3C_RL_net(lamda, state_size, lstm_hidden_dim, num_layers, action_c_size, action_I_size)
    
    # record action lists
    actions_list = []
    
    # multiple evaluations
    for k in range(iteration):
        env.reset()
        actions_taken = []
        actions_prob = []
        done = False
        epi = 0 

        while not done:
            
            if epi == 0: # preprocess treatment
                action_c = action_c_size - 1
                action_I = 0 
                _, _, done, now_state = env.step(action_c, action_I, epi, pre_therapy_time, cc, lamda)
                actions_taken.append([action_c, action_I]) 
                epi += 1
            else:
                rl_model.actor_c.eval()
                rl_model.actor_I.eval()
                state_tensor = torch.tensor(now_state, dtype=torch.float32, requires_grad=True).float().to(device).unsqueeze(0)
                state_tensor_lstm = rl_model.lstm(state_tensor)

                policy_dist_c = rl_model.actor_c(state_tensor_lstm[-1:,:])
                action_c = torch.argmax(policy_dist_c, dim=-1).item() # use argmax in evaluation

                policy_dist_I = rl_model.actor_I(state_tensor_lstm[-1:,:])
                action_I = torch.argmax(policy_dist_I, dim=-1).item() # use argmax in evaluation

                action_prob_c = policy_dist_c.detach().cpu().numpy().flatten()
                action_prob_I = policy_dist_I.detach().cpu().numpy().flatten()
                
                _, _, done, now_state= env.step(action_c, action_I, epi, predict_dt, cc, lamda)

                actions_taken.append([action_c, action_I])  # record actions
                actions_prob.append([action_prob_c, action_prob_I])
                if epi >= ((predict_time - pre_therapy_time)/predict_dt):
                    done = True    
                epi = epi + 1
     
        actions_list.append(actions_taken) # record action list

        # print
        '''
        print("actions_taken", actions_taken)
        for i, (prob_c, prob_I) in enumerate(actions_prob):
            print(f"Step {i+1}: Actor_c: [{prob_c[0]:.2f}, {prob_c[1]:.2f}], Actor_I: [{prob_I[0]:.2f}, {prob_I[1]:.2f}]")
        print(f" --------------- ")
        '''

    # save global A3C net
    torch.save(rl_model.critic.state_dict(), f'{eval_save_folder}/{str_lamda}/global_critic_{str_lamda}.pth')
    print("global_critic saved")
    torch.save(rl_model.actor_c.state_dict(), f'{eval_save_folder}/{str_lamda}/global_actor_c_{str_lamda}.pth')
    print("global_actor_c saved")
    torch.save(rl_model.actor_I.state_dict(), f'{eval_save_folder}/{str_lamda}/global_actor_I_{str_lamda}.pth')
    print("global_actor_I saved")
    torch.save(rl_model.lstm.state_dict(), f'{eval_save_folder}/{str_lamda}/global_LSTM_{str_lamda}.pth')
    print("global_LSTM saved")

    return actions_list

# convert survival rate sequence to time & event
def convert_to_time_event(survival_rate):
    time = []
    event = []
    survival_num = 100
    for i in range(len(survival_rate) - 1):
        cut_num = survival_num - int(np.floor((survival_rate[i]) * 100))
        if cut_num < 0:
            cut_num = 0
        survival_num = survival_num - cut_num
        for j in range(cut_num):
            time.append(i + 1)
            event.append(1)
    for i in range(survival_num):
            time.append(len(survival_rate))
            event.append(0)
    return time, event

def main():
    lamda = 0.005 # control coefficient in reward funtion
    str_lamda = str(lamda).replace('.', '_')

    FP_layers = [5, 40, 120, 250, 500, 1000, 1000, 600, 300, 150, 1]

    eval_iter = 1 # evaluation iteration
    
    state_size = 2
    lstm_hidden_dim = state_size * 2
    num_layers = 1

    predict_time = 28*7
    pre_therapy_time = 28

    dt = predict_dt
    dt_cyto = 1/48
    t_max = predict_time
    t = np.arange(pre_therapy_time, t_max + dt, dt)

    dc = 0.05
    c_min, c_max = 0, 1
    c = np.arange(c_min + dc, c_max + dc, dc)
    
    t = np.array(t)[:, None]
    c = np.array(c)[:, None]

    initial_TC = 0.58
    initial_CSF1RI = 0
    initial_IGF1RI = 0

    C, T = np.meshgrid(c,t) 
    C_star = np.hstack((C.flatten()[:,None], T.flatten()[:,None]))

    lb = [0.05, 5]
    ub = [1.0, 200]  
    
    surrogate_model = RL_dNN(FP_layers, lb, ub, dc, dt_cyto)
    env = Env(surrogate_model, [initial_CSF1RI, initial_IGF1RI, initial_TC])    
    
    # create folder for saving evaluation
    eval_folder = f'{eval_save_folder}/{str_lamda}'
    if not os.path.exists(eval_folder):
        os.makedirs(eval_folder)

    # optimal RL treatment list
    actions_list = A3C_RL_eval_worker(env, eval_iter, c, state_size, lstm_hidden_dim, num_layers, predict_time, predict_dt, pre_therapy_time, lamda)

    # combination treatment actions
    combination_actions = []
    combination_actions.append([1, 1])
    for i in range(39):
        combination_actions.append([1, 1])

    predict_CSF1RI_cum = 0
    predict_CC_list = []
    predict_dose_c_list = []
    predict_dose_I_list = []
    predict_CSF1RI_list = []
    predict_CSF1RI_list.append(initial_CSF1RI)
    predict_IGF1RI_list = []
    predict_IGF1RI_list.append(initial_IGF1RI)
    predict_TC_list = []
    predict_TC_list.append(initial_TC)

    combination_length_time = len(combination_actions)
    for k in range(combination_length_time):

        predict_dose_c = env.action_c_space[combination_actions[k][0]]   
        predict_dose_I = env.action_I_space[combination_actions[k][1]]
        
        c_CSF1RI_0 = predict_CSF1RI_list[-1]
        c_IGF1RI_0 = predict_IGF1RI_list[-1]
        c_T_0 = predict_TC_list[-1]

        if k == 0: # preprocess treatment
            predict_tt = pre_therapy_time
            rl_FP_TC, c_CSF1RI_cum, c_CSF1RI, c_IGF1RI = surrogate_model.predict_DRL_FP(c, predict_tt, predict_dose_c, predict_CSF1RI_cum, predict_dose_I,
                                                                c_CSF1RI_0, c_IGF1RI_0, c_T_0, 5)
        else:
            predict_tt = pre_therapy_time + k * 5
            rl_FP_TC, c_CSF1RI_cum, c_CSF1RI, c_IGF1RI = surrogate_model.predict_DRL_FP(c, predict_tt, predict_dose_c, predict_CSF1RI_cum, predict_dose_I,
                                                                c_CSF1RI_0, c_IGF1RI_0, c_T_0, 5)
        predict_CSF1RI_cum = c_CSF1RI_cum
        rl_FP_TC_norm = rl_FP_TC/(np.sum(rl_FP_TC) + 1e-7)

        predict_CC_list.append(rl_FP_TC_norm * 1/dc)
        predict_CSF1RI_list.append(c_CSF1RI)
        predict_IGF1RI_list.append(c_IGF1RI)
        predict_TC_list.append(np.sum(rl_FP_TC_norm*c))

        # more denser sequences of drug dose
        # better in plotting
        if k == 0:
            for j in range(48 * 5):
                predict_dose_c_list.append(predict_dose_c)
                predict_dose_I_list.append(predict_dose_I)
        else:
            for j in range(48 * 5):
                predict_dose_c_list.append(predict_dose_c)
                predict_dose_I_list.append(predict_dose_I)            
    
    t_combine = np.arange(5, t_max + 5, 5)
    t_combine = np.array(t_combine)[:, None]
    C_combine, T_combine = np.meshgrid(c, t_combine) 
    C_combine_star = np.hstack((C_combine.flatten()[:,None], T_combine.flatten()[:,None]))
    predict_CClist_flat = np.concatenate(predict_CC_list).ravel()
    # surrogate model results with combination treatment
    U_pred_combination = griddata(C_combine_star, predict_CClist_flat, (C_combine, T_combine), method='cubic')
    U_pred_combination[U_pred_combination < 0] = 0
    
    # evaluation of optimal RL treatment
    max_survival_prob = 0
    min_invivo_CSF1RI = np.inf

    for i in range(0, eval_iter):
        predict_actions = actions_list[i]
        length_time = len(predict_actions)
        # prerocess treatment
        while length_time < int((predict_time - pre_therapy_time) / predict_dt)+1 :
            predict_actions.append([0, 0])
            length_time = len(predict_actions)

        predict_CSF1RI_cum = 0
        predict_CC_list = []
        predict_dose_c_list = []
        predict_dose_I_list = []
        predict_CSF1RI_list = []
        predict_CSF1RI_list.append(initial_CSF1RI)
        predict_IGF1RI_list = []
        predict_IGF1RI_list.append(initial_IGF1RI)
        predict_TC_list = []
        predict_TC_list.append(initial_TC)

        for k in range(length_time):

            predict_dose_c = env.action_c_space[predict_actions[k][0]]   
            predict_dose_I = env.action_I_space[predict_actions[k][1]]

            c_CSF1RI_0 = predict_CSF1RI_list[-1]
            c_IGF1RI_0 = predict_IGF1RI_list[-1]
            c_T_0 = predict_TC_list[-1]

            if k == 0: # preprocess treatment
                predict_tt = pre_therapy_time
                rl_FP_TC, c_CSF1RI_cum, c_CSF1RI, c_IGF1RI = surrogate_model.predict_DRL_FP(c, predict_tt, predict_dose_c, predict_CSF1RI_cum, predict_dose_I,
                                                                    c_CSF1RI_0, c_IGF1RI_0, c_T_0, pre_therapy_time)
            else:
                predict_tt = pre_therapy_time + k * predict_dt
                rl_FP_TC, c_CSF1RI_cum, c_CSF1RI, c_IGF1RI = surrogate_model.predict_DRL_FP(c, predict_tt, predict_dose_c, predict_CSF1RI_cum, predict_dose_I,
                                                                    c_CSF1RI_0, c_IGF1RI_0, c_T_0, predict_dt)
            
            predict_CSF1RI_cum = c_CSF1RI_cum
            rl_FP_TC_norm = rl_FP_TC/(np.sum(rl_FP_TC) + 1e-7)
            predict_CC_list.append(rl_FP_TC_norm * 1/dc)
            predict_CSF1RI_list.append(c_CSF1RI)
            predict_IGF1RI_list.append(c_IGF1RI)
            predict_TC_list.append(np.sum(rl_FP_TC_norm*c))

            # 
            if k == 0:
                for j in range(48 * pre_therapy_time):
                    predict_dose_c_list.append(predict_dose_c)
                    predict_dose_I_list.append(predict_dose_I)
            else:
                for j in range(48 * predict_dt):
                    predict_dose_c_list.append(predict_dose_c)
                    predict_dose_I_list.append(predict_dose_I)    

        predict_CClist_flat = np.concatenate(predict_CC_list).ravel()
        dose_c = np.array(predict_dose_c_list)
        dose_I = np.array(predict_dose_I_list)
        predict_CSF1RI = np.array(predict_CSF1RI_list)
        predict_IGF1RI = np.array(predict_IGF1RI_list)
        predict_TC = np.array(predict_TC_list)
        
        U_pred_RL = griddata(C_star, predict_CClist_flat, (C, T), method='cubic')
        U_pred_RL[U_pred_RL < 0] = 0

        # plot and save surrogate model result with optimal RL treatment
        '''
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)
        h = ax.imshow(U_pred_RL.T, interpolation='nearest', cmap='rainbow', 
                    extent=[t.min(), t.max(), c.min(), c.max()], 
                    origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(h, cax=cax)
        cbar.ax.tick_params(labelsize=15)  
        ax.set_xlabel('$t$', size=20)
        ax.set_ylabel('$C$', size=20)
        ax.set_title('$p(t,C)$', fontsize = 20) # font size doubled
        ax.tick_params(labelsize=15)
        
        save_path = f'{eval_folder}/optimalRL_TC_{i}_{str_lamda}.pdf'
        plt.savefig(save_path, format='pdf')
        '''
        
        t_dose = np.arange(1/48, 1/48 + predict_time, 1/48)
        t_invivo = np.arange(pre_therapy_time, predict_dt + predict_time, predict_dt)
        t_invivo = np.insert(t_invivo, 0, 0)
        # plot drug dose and invivo drug concentration
        '''
        fig = plt.figure(figsize=(5, 12))
        ax = fig.add_subplot(111)
        gs1 = gridspec.GridSpec(2, 1)
        gs1.update(top=0.9, bottom=0.1, left=0.05, right=0.95, wspace=0.3, hspace=0.3)
    
        ax = plt.subplot(gs1[0, 0])
        ax.plot(t_dose, dose_c, 'b-', linewidth = 2, label = 'CSF1RI dose')  
        ax.plot(t_invivo, predict_CSF1RI, 'k-', linewidth = 4, label = 'invivo CSF1RI')          
        ax.set_xlabel('$t$')
        ax.set_ylabel('$c$')    
        ax.set_title('$CSF1RI$', fontsize = 15)
        ax.axis('square')
        ax.set_xlim([0.0,t_max])
        ax.set_ylim([0.0,1])
        aspect_ratio = 1 * (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.set_aspect(aspect_ratio)
        ax.set_xticks(t_invivo)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                        ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(15)
        ax.tick_params(axis='x', labelsize=6)
        ax.legend()

        ax = plt.subplot(gs1[1, 0])
        ax.plot(t_dose, dose_I, 'g-', linewidth = 2, label = 'IGF1RI dose')  
        ax.plot(t_invivo, predict_IGF1RI, 'k-', linewidth = 4, label = 'invivo IGF1RI')     
        ax.plot(t_invivo, predict_TC, 'r-', linewidth = 4, label = 'TC expectation')  
        ax.set_xlabel('$t$')
        ax.set_ylabel('$c$')    
        ax.set_title('$IGF1RI & TC$', fontsize = 15)
        ax.axis('square')
        ax.set_xlim([0.0,t_max])
        ax.set_ylim([0.0,1])
        aspect_ratio = 1 * (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.set_aspect(aspect_ratio)
        ax.set_xticks(t_invivo)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                        ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(15)
        ax.tick_params(axis='x', labelsize=6)
        ax.legend()

        save_path = f'{eval_folder}/dose_{i}_{str_lamda}.pdf'
        plt.savefig(save_path, format='pdf')
        print(f'dose_{i} done') 
        '''

        # plot KM curves
        # log-rank test on optimal RL treatment compared with combination treatment
        optimalRL_survival_rate = (1 - np.sum(U_pred_RL.T[-3:, :], axis = 0)/np.sum(U_pred_RL.T, axis = 0))
        Combination_survival_rate = (1 - np.sum(U_pred_combination.T[-3:, :], axis = 0)/np.sum(U_pred_combination.T, axis = 0))

        Combination_survival_rate[0] = 1
        optimalRL_survival_rate[0] = 1
        for j in range(1, len(Combination_survival_rate)):
            if Combination_survival_rate[j] > Combination_survival_rate[j-1]:
                Combination_survival_rate[j] = Combination_survival_rate[j-1]
        for j in range(1, len(optimalRL_survival_rate)):
            if optimalRL_survival_rate[j] > optimalRL_survival_rate[j-1]:
                optimalRL_survival_rate[j] = optimalRL_survival_rate[j-1]

        optimalRL_time, optimalRL_event = convert_to_time_event(optimalRL_survival_rate)
        combination_time, combination_event = convert_to_time_event(Combination_survival_rate)

        optimalRL_time = [(pre_therapy_time + (t-1) * predict_dt) for t in optimalRL_time]
        combination_time = [(5 + (t-1) * 5) for t in combination_time]

        data_optimalRL = pd.DataFrame({'time': optimalRL_time, 'event': optimalRL_event, 'group': ['optimalRL'] * len(optimalRL_time)})
        data_combination = pd.DataFrame({'time': combination_time, 'event': combination_event, 'group': ['combination'] * len(combination_time)})
        df = pd.concat([data_optimalRL, data_combination])

        # groups
        group_RL = df[df['group'] == 'optimalRL']
        group_combination = df[df['group'] == 'combination']

        # KM curves
        kmf_RL = lifelines.KaplanMeierFitter()
        kmf_RL.fit(group_RL['time'], event_observed=group_RL['event'], label='optimalRL')

        kmf_combination = lifelines.KaplanMeierFitter()
        kmf_combination.fit(group_combination['time'], event_observed=group_combination['event'], label='combination')

        # log-rank test
        results_RL_combination = lifelines.statistics.logrank_test(group_RL['time'], group_combination['time'], event_observed_A=group_RL['event'], event_observed_B=group_combination['event'])

        # final survival rate
        final_survival_rate_RL = kmf_RL.survival_function_.iloc[-1, 0]

        # plot
        fig = plt.figure(figsize=(14, 12))
        kmf_RL.plot_survival_function(ci_show=False)
        kmf_combination.plot_survival_function(ci_show=False)
        
        # add notes
        plt.annotate(f'optimalRL vs combination p-value: {results_RL_combination.p_value:.4g}', xy=(0.95, 0.15), xycoords='axes fraction', fontsize=12, color='red', horizontalalignment='right')
        plt.annotate(f'final survival rate (optimalRL): {final_survival_rate_RL:.4%}', xy=(0.95, 0.10), xycoords='axes fraction', fontsize=12, color='blue', horizontalalignment='right')

        plt.xlabel('time (days)')
        plt.ylabel('survival probability')
        plt.xticks(t_invivo)
        plt.legend()
        plt.xlim(0, t_max)
        plt.ylim(0, 1)
        print(f'KM_{i} done') 

        # select best treatment and save figures
        if final_survival_rate_RL >= max_survival_prob:
            save_path = f'{eval_folder}/KM_max_{final_survival_rate_RL:.1%}.pdf'
            plt.savefig(save_path, format='pdf')
            print(f'KM_max done')  
            U_show = U_pred_RL.T
            # higher final survival rate
            if final_survival_rate_RL > max_survival_prob: 
                # plot and save optimal RL surrogate model result
                c_edges = np.insert(c, 0, 0) 
                T_show, C_show = np.meshgrid(t_invivo, c_edges)
                fig, ax = plt.subplots(figsize=(9, 5))
                h = ax.pcolormesh(T_show, C_show, U_show, shading='auto', cmap='rainbow')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.10)
                cbar = fig.colorbar(h, cax=cax)
                cbar.ax.tick_params(labelsize=15)

                ax.set_xlabel('$t$', size=20)
                ax.set_ylabel('$C$', size=20)
                ax.set_title('$p(t,C)$', fontsize=20)
                ax.tick_params(labelsize=15)
                ax.set_xticks(t_invivo)
                ax.tick_params(axis='x', labelsize=6)

                save_path = f'{eval_folder}/optimalRL_TC_max.pdf'
                plt.savefig(save_path, format='pdf')
                print(f'optimalRL_TC_max done')

                # plot and save drug dose
                fig = plt.figure(figsize=(5, 12))
                ax = fig.add_subplot(111)
                gs1 = gridspec.GridSpec(2, 1)
                gs1.update(top=0.9, bottom=0.1, left=0.05, right=0.95, wspace=0.3, hspace=0.3)
                
                ax = plt.subplot(gs1[0, 0])
                ax.plot(t_dose, dose_c, 'b-', linewidth = 2, label = 'CSF1RI dose')  
                ax.plot(t_invivo, predict_CSF1RI, 'k-', linewidth = 4, label = 'invivo CSF1RI')       
                ax.set_xlabel('$t$')
                ax.set_ylabel('$c$')    
                ax.set_title('$CSF1RI$', fontsize = 15)
                ax.axis('square')
                ax.set_xlim([0.0,t_max])
                ax.set_ylim([0.0,1])
                aspect_ratio = 1 * (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
                ax.set_aspect(aspect_ratio)
                ax.set_xticks(t_invivo)
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(15)
                ax.tick_params(axis='x', labelsize=6)
                ax.legend()

                ax = plt.subplot(gs1[1, 0])
                ax.plot(t_dose, dose_I, 'g-', linewidth = 2, label = 'IGF1RI dose')  
                ax.plot(t_invivo, predict_IGF1RI, 'k-', linewidth = 4, label = 'invivo IGF1RI')     
                ax.plot(t_invivo, predict_TC, 'r-', linewidth = 4, label = 'TC expectation')  
                ax.set_xlabel('$t$')
                ax.set_ylabel('$c$')    
                ax.set_title('$IGF1RI & TC$', fontsize = 15)
                ax.axis('square')
                ax.set_xlim([0.0,t_max])
                ax.set_ylim([0.0,1])
                aspect_ratio = 1 * (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
                ax.set_aspect(aspect_ratio)
                ax.set_xticks(t_invivo)
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(15)
                ax.tick_params(axis='x', labelsize=6)
                ax.legend()

                save_path = f'{eval_folder}/dose_max.pdf'
                plt.savefig(save_path, format='pdf')
                print(f'dose_max done')  

            # lower CSF1R_I total usage
            elif min_invivo_CSF1RI < predict_CSF1RI_cum:
                # plot and save optimal RL surrogate model result
                fig, ax = plt.subplots(figsize=(9, 5))
                h = ax.pcolormesh(T_show, C_show, U_show, shading='auto', cmap='rainbow')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.10)
                cbar = fig.colorbar(h, cax=cax)
                cbar.ax.tick_params(labelsize=15)
                ax.set_xlabel('$t$', size=20)
                ax.set_ylabel('$C$', size=20)
                ax.set_title('$p(t,C)$', fontsize=20)
                ax.tick_params(labelsize=15)
                ax.set_xticks(t_invivo)
                ax.tick_params(axis='x', labelsize=6)
                save_path = f'{eval_folder}/optimalRL_TC_max.pdf'
                plt.savefig(save_path, format='pdf')
                print(f'optimalRL_TC_max done') 

                # plot and save drug dose
                fig = plt.figure(figsize=(5, 12))
                ax = fig.add_subplot(111)
                gs1 = gridspec.GridSpec(2, 1)
                gs1.update(top=0.9, bottom=0.1, left=0.05, right=0.95, wspace=0.3, hspace=0.3)
                
                ax = plt.subplot(gs1[0, 0])
                ax.plot(t_dose, dose_c, 'b-', linewidth = 2, label = 'CSF1RI dose')  
                ax.plot(t_invivo, predict_CSF1RI, 'k-', linewidth = 4, label = 'invivo CSF1RI')     
                ax.set_xlabel('$t$')
                ax.set_ylabel('$c$')    
                ax.set_title('$CSF1RI$', fontsize = 15)
                ax.axis('square')
                ax.set_xlim([0.0,t_max])
                ax.set_ylim([0.0,1])
                aspect_ratio = 1 * (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
                ax.set_aspect(aspect_ratio)
                ax.set_xticks(t_invivo)
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(15)
                ax.tick_params(axis='x', labelsize=6)
                ax.legend()

                ax = plt.subplot(gs1[1, 0])
                ax.plot(t_dose, dose_I, 'g-', linewidth = 2, label = 'IGF1RI dose')  
                ax.plot(t_invivo, predict_IGF1RI, 'k-', linewidth = 4, label = 'invivo IGF1RI')     
                ax.plot(t_invivo, predict_TC, 'r-', linewidth = 4, label = 'TC expectation')  
                ax.set_xlabel('$t$')
                ax.set_ylabel('$c$')    
                ax.set_title('$IGF1RI & TC$', fontsize = 15)
                ax.axis('square')
                ax.set_xlim([0.0,t_max])
                ax.set_ylim([0.0,1])
                aspect_ratio = 1 * (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
                ax.set_aspect(aspect_ratio)
                ax.set_xticks(t_invivo)
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(15)
                ax.tick_params(axis='x', labelsize=6)
                ax.legend()

                save_path = f'{eval_folder}/dose_max.pdf'
                plt.savefig(save_path, format='pdf')
                print(f'dose_max done')

            min_invivo_CSF1RI = predict_CSF1RI_cum
            max_survival_prob = final_survival_rate_RL

if __name__ == "__main__":
    main()
