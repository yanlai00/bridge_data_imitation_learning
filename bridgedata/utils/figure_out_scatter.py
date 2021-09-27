import torch


nb_digits = 10
# Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
# active_dim = active_dim.type(torch.LongTensor).view(-1, 1)
batch_size = 8
T = 14
active_dim = torch.LongTensor(batch_size, T, 1).random_() % nb_digits
# active_dim = active_dim.type(torch.LongTensor)[:, 0]
# One hot encoding buffer that you create out of the loop and just keep reusing
# T = active_dim.shape[1]
batch_size = active_dim.shape[0]
y_onehot = torch.FloatTensor(batch_size, T, nb_digits)

# In your for loop
y_onehot.zero_()
import pdb; pdb.set_trace()
y_onehot.scatter_(2, active_dim, 1)

print(active_dim)
print(y_onehot)
import pdb; pdb.set_trace()
