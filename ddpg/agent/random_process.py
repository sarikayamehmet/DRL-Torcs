
import torch

class OrnsteinUhlenbeckProcess(object):

    def __init__(self, theta, sigma, dim, mu=0):
        self.theta = theta
        self.sigma = sigma
        self.mu = mu
        self.state = mu
        self.dim = dim

    def noise(self):
        v = self.theta*(self.mu - self.state) + self.sigma*torch.randn(self.dim)
        self.state += v
        return self.state

    def reset(self):
        self.state = self.mu

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    process = OrnsteinUhlenbeckProcess(0.15, 0.2, 1)
    signal = []
    for i in range(1000):
        signal.append(process.noise().item())
    plt.plot(signal)
    plt.show()