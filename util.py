import numpy as np
import torch
import matplotlib.pyplot as plt

def sample_noise(batch_size, num_category = 10, num_conti=2, noise_dim=62):
    idx = np.random.randint(num_category, size=batch_size)
    category_code = np.zeros((batch_size, num_category))
    category_code[range(batch_size),idx] = 1.0
    conti_code = np.random.uniform(-1.0,1.0,(batch_size,num_conti))

    random_noise = np.random.uniform(-1.0,1.0,(batch_size,noise_dim))

    z = torch.cat((torch.Tensor(random_noise),torch.Tensor(category_code),torch.Tensor(conti_code)),dim=1)

    return z, idx

def get_test_noise(num_category = 10, num_conti=2, noise_dim=62):
    fixed_noise = np.random.uniform(-1.0,1.0,(noise_dim))
    # z1 : fix c2
    # z2 : fix c1
    z1 = []
    z2 = []
    for cat in range(num_category):
        category_code = np.zeros((num_category))
        category_code[cat] = 1.0
        for c in np.arange(-2,2.1,0.5):
            z1.append(np.concatenate([fixed_noise,category_code,np.array([c]),np.array([0])]))
            z2.append(np.concatenate([fixed_noise,category_code,np.array([0]),np.array([c])]))
    z1 = torch.Tensor(np.array(z1))
    z2 = torch.Tensor(np.array(z2))

    return z1,z2

def save_fig(z,G,fig_name,num_category = 10,num_conti=9):
    fake_x = G(z).data.cpu().numpy().reshape(num_category*num_conti,28,28)
    fig, axs = plt.subplots(num_category,num_conti,figsize=(20,20))
    for i in range(num_category):
        for j in range(num_conti):
            axs[i,j].get_xaxis().set_visible(False)
            axs[i,j].get_yaxis().set_visible(False)
            axs[i,j].imshow(fake_x[i*num_conti+j], cmap='gray')
    plt.savefig(fig_name,bbox_inches='tight')
    plt.close()