import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt



class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        y = self.sigmoid(x)
        return y

if __name__ == '__main__':
    sample_num = 1000
    mean_val = 2.5
    bias = 1
    n_data = torch.ones(sample_num, 2)
    x0 = torch.normal(mean_val * n_data, 1) + bias # 数据服从正态分布，正值
    y0 = torch.zeros(sample_num) # 正值标签为0
    x1 = torch.normal(-mean_val * n_data, 1) + bias # 数据服从正态分布，负值
    y1 = torch.ones(sample_num) # 负值标签为1

    train_x = torch.cat((x0, x1), dim=0)
    train_y = torch.cat((y0, y1), dim=0)

    my_model = LogisticRegression()
    learning_rate = 0.01
    optimizer = optim.Adam(my_model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()

    for i in range(10000):
        optimizer.zero_grad()
        output = my_model(train_x)
        loss = loss_fn(output.squeeze(), train_y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mask = output.ge(0.5).float().squeeze()
            correct = (mask == train_y).sum()
            acc = correct / train_y.size(0)

        if acc > 0.99:
            plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], c='blue')
            plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1], c='red')

            w0, w1 = my_model.linear.weight[0]
            w0, w1 = float(w0.item()), float(w1.item())
            plot_b = float(my_model.linear.bias[0].item())
            plot_x = torch.arange(-6, 6, 0.1)
            """ 分界线上的点满足sigmoid(w0x0+w1x1+b)=0.5，可知w0x0+w1x1+b=0，
            而图像坐标系是x0Ox1，纵轴为x1，则分界线的纵坐标x1=-(w0x0+b)/w1 """
            plot_y = (-w0 * plot_x - plot_b) / w1

            plt.xlim(-5, 7)
            plt.ylim(-5, 7)
            plt.plot(plot_x, plot_y, color='green')
            plt.title(f"iterations: {i}, loss: {loss.item():.4f}, accuracy: {acc:.4%}\n"
                      f"w0: {w0:.4f}, w1: {w1:.4f}, b: {plot_b:.4f}")
            plt.show()
            break