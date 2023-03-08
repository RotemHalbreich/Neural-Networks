import numpy as np
import matplotlib.pyplot as plt


def find_max_min(my_data):

    maxX = my_data[:,0].max()
    maxY = my_data[:,1].max()

    minX = my_data[:,0].min()
    minY = my_data[:,1].min()

    return maxX, maxY, minX, minY


def Gaussian(neurons_index,center=None,sigma=0.5):

    if center == None:
        center = np.array(list(neurons_index))//2
    d = 0
    for i in range(len(neurons_index)):
        d += ((center[i]-neurons_index[i])/float(len(neurons_index[i])))**2

    dist_sqrt = np.sqrt(d)/np.sqrt(len(neurons_index))
    h_x = np.exp(-dist_sqrt**2/sigma**2)

    return h_x


def draw_report(self, num_of_iter, epochs, opt):

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(opt, fontsize=14, fontweight='bold')
    axes = fig.add_subplot()

    x, y = self.samples[:, 0], self.samples[:, 1]
    plt.scatter(x, y, alpha=0.5, color='orange', marker='.', s=80)

    x, y = self.som_weight[..., 0], self.som_weight[..., 1]

    if len(self.som_weight.shape) > 2:

        for i in range(self.som_weight.shape[0]):
            plt.plot(x[i, :], y[i, :], 'b', alpha=0.8)
        for i in range(self.som_weight.shape[1]):
            plt.plot(x[:, i], y[:, i], 'b', alpha=0.8)
    else:
        plt.plot(x, y, 'b', alpha=0.8)

    plt.scatter(x, y,  s=50, facecolor='w', edgecolor='purple', zorder=10)
    axes.set_title("Samples size: " + str(len(self.samples)) + ", "
                  + "Epoch: " + str(epochs) + ", "
                  + "Iter number: " + str(num_of_iter) + ", "
                  + "Neurons Number: " + str(self.som_weight.shape[0]))
    save_file = opt + "_{}.png"
    plt.savefig(save_file.format(str(num_of_iter)))
    plt.show()


def init_som_weight(maxX, maxY, minX, minY, shape):
    if len(shape) > 2:
        som_weight = np.array([[[np.random.uniform(minX, maxX), np.random.uniform(minY, maxY)] for i in range(shape[0])] for j in range(shape[1])])
    else:
        som_weight = np.array([[np.random.uniform(minX, maxX), np.random.uniform(minY, maxY)] for i in range(shape[0])])
    return som_weight


class SOM:

    def __init__(self, *args):
        ''' Initialize SOM '''

        last_element_index = len(args) - 1
        tup = args[:last_element_index]
        self.som_shape = np.zeros(tup)
        self.samples = args[last_element_index]

        maxX, maxY, minX, minY = find_max_min(args[last_element_index])
        self.som_weight = init_som_weight(maxX, maxY, minX, minY, self.som_shape.shape)


def learn(self, epochs=10000, sigma=(10, 0.001), lrate=(1.0, 0.001), report_iter=1000, opt=None):

    sigma_i, sigma_f = sigma
    lrate_i, lrate_f = lrate

    for i in range(epochs):
        # Adjust learning rate and neighborhood
        t = i/float(epochs)

        lrate = lrate_i*(lrate_f/float(lrate_i))**t
        sigma = sigma_i*(sigma_f/float(sigma_i))**t

        # Get random sample
        index = np.random.randint(0,self.samples.shape[0])
        data = self.samples[index]

        # Get index of nearest node (minimum distance)

        min_dist = ((data-self.som_weight)**2).sum(axis=-1)
        winner = np.unravel_index(np.argmin(min_dist), min_dist.shape)

        if len(min_dist.shape) > 1:
            rows = np.arange(min_dist.shape[0]).reshape(-1, 1) + np.zeros((1, min_dist.shape[0]))
            cols = np.arange(float(0), float(min_dist.shape[0])) + np.zeros((min_dist.shape[0], 1))
            neurons_index = [rows, cols]
        else:
            line = np.arange(float(min_dist.shape[0]))
            neurons_index = [line]

        # Generate a Gaussian centered on winner
        G = Gaussian(neurons_index, winner, sigma)
        G = np.nan_to_num(G)

        # Move nodes towards sample according to Gaussian
        delta = data-self.som_weight

        if i % report_iter == 0 or i+1 == epochs:
            if i+1 == epochs:
                i = epochs
            draw_report(self, i, epochs, opt)

        for x in range(self.som_weight.shape[-1]):
            self.som_weight[..., x] += lrate * G * delta[..., x]
