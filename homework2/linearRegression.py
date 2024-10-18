import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    learning_rate = 0.01

    x = torch.randn(20, 1)
    y = 3 * x + (5 + torch.randn(20, 1))

    w = torch.randn(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    for i in range(20000):
        y_pre = w * x + b
        loss = (0.5 * (y - y_pre) ** 2).mean()

        loss.backward()

        with torch.no_grad():
            w.sub_(learning_rate * w.grad)
            b.sub_(learning_rate * b.grad)

        w.grad.zero_()
        b.grad.zero_()

        if loss.item() < 0.5:
            x_min = float(torch.min(x)) * 0.8
            x_max = float(torch.max(x)) * 1.2
            y_min = float(torch.min(y)) * 0.8
            y_max = float(torch.max(y)) * 1.2
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), y_pre.data.numpy())
            plt.text(x_min + (x_max - x_min) * 0.1, y_max - (y_max - y_min) * 0.3, f"loss: {loss.item()}")
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.show()
            break

