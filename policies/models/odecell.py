import torch
import torch.nn as nn

class ODERNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, solver_type="fixed_euler"):
        super(ODERNNCell, self).__init__()
        self.solver_type = solver_type
        self.rnn = nn.GRUCell(input_size, hidden_size)
        # f_node for partial derivatives
        self.f_node = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        options = {
            "fixed_euler": self.euler,
            "fixed_heun": self.heun,
            "fixed_rk4": self.rk4,
        }
        if not solver_type in options.keys():
            raise ValueError("Unknown solver type '{:}'".format(solver_type))
        self.node = options[self.solver_type]

    def forward(self, x, hx, ts):
        # [batchsize, hidden_dim]
        '''
         x, hx, ts: torch.Size([32, 64]) torch.Size([32, 128]) torch.Size([32, 1])
         h_prime: [32, 128]
         new_h: [32, 128]
        '''
        h_prime = self.rnn(x, hx)
        new_h = self.solve_fixed(h_prime, ts)

        return new_h

    def solve_fixed(self, x, ts):
        ts = ts.view(-1, 1)
        for i in range(3):  #每次计算三次数值解的迭代
            x = self.node(x, ts * (1.0 / 3))
        return x

    def euler(self, y, delta_t):
        dy = self.f_node(y)
        return y + delta_t * dy

    def heun(self, y, delta_t):
        k1 = self.f_node(y)
        k2 = self.f_node(y + delta_t * k1)
        return y + delta_t * 0.5 * (k1 + k2)

    def rk4(self, y, delta_t):
        k1 = self.f_node(y)
        k2 = self.f_node(y + k1 * delta_t * 0.5)
        k3 = self.f_node(y + k2 * delta_t * 0.5)
        k4 = self.f_node(y + k3 * delta_t)

        return y + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

class ODERNN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_size,
        return_sequences=True,
        solver_type="fixed_euler",
    ):
        super(ODERNN, self).__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        self.out_feature = hidden_size


        self.rnn_cell = ODERNNCell(in_features, hidden_size, solver_type=solver_type)
        self.fc = nn.Linear(self.hidden_size, self.out_feature)


    def forward(self, x, init_hidden=None):
        # input size (Length, Batchsize, Input size)

        device = x.device
        batch_size = x.size(1)
        seq_len = x.size(0)
        timespans = torch.ones((batch_size, 1), device=device) * 0.1
        # timespans = torch.randint(10, 50, (batch_size, 1), device=device) * 0.01

    
        if init_hidden is not None:
            hidden_state = init_hidden.reshape(-1, self.hidden_size) 

        else:
            hidden_state = torch.zeros((batch_size, self.hidden_size), device=device) 
            
        outputs = []

        last_output = torch.zeros((batch_size, self.out_feature), device=device)
        for t in range(seq_len):
            inputs = x[t]
            ts = timespans
            hidden_state = self.rnn_cell.forward(inputs, hidden_state, ts)
            current_output = self.fc(hidden_state)  

            outputs.append(current_output)

            last_output = current_output

            
        if self.return_sequences:
            outputs = torch.stack(outputs, dim=0)  # return entire sequence    #
        else:
            outputs = last_output  # only last item
        return outputs, hidden_state