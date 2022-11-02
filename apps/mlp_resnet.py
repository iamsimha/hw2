import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(nn.Residual(nn.Sequential(
                          nn.Linear(in_features=dim, out_features=hidden_dim),
                          norm(hidden_dim),
                          nn.ReLU(),
                          nn.Dropout(drop_prob),
                          nn.Linear(in_features=hidden_dim, out_features=dim),
                          norm(dim))), nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    blocks = [nn.Linear(in_features=dim, out_features=hidden_dim), nn.ReLU()]
    for i in range(num_blocks):
        blocks.append(ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob))
    blocks.append(nn.Linear(in_features=hidden_dim, out_features=num_classes))
    return nn.Sequential(*blocks)
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    if opt:
        model.train()
    else:
        model.eval()
    np.random.seed(4)
    loss_fn = ndl.nn.SoftmaxLoss()
    ### BEGIN YOUR SOLUTION
    losses, all_preds = [], []
    for idx, batch in enumerate(dataloader):
        batch_x, batch_y = batch
        preds = model(batch_x.reshape((batch_x.shape[0], -1)))
        all_preds.append((np.argmax(preds.numpy(), axis=-1) == batch_y.numpy()))
        loss = loss_fn(preds, batch_y)
        losses.append(loss.detach().numpy())
        if opt:
            opt.reset_grad()
            loss.backward()
            opt.step()
    return 1 - np.concatenate(all_preds).mean(), np.mean(np.array(losses))
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    mnist_train_dataset = ndl.data.MNISTDataset(os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
                                                os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    mnist_train_dataloader = ndl.data.DataLoader(dataset=mnist_train_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True)

    mnist_test_dataset = ndl.data.MNISTDataset("data/t10k-images-idx3-ubyte.gz",
                                               "data/t10k-labels-idx1-ubyte.gz")
    mnist_test_dataloader = ndl.data.DataLoader(dataset=mnist_test_dataset,
                                                batch_size=batch_size,
                                                shuffle=False)
    model = MLPResNet(784, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epcoh_num in range(epochs):
        train_error_rate, train_loss = epoch(mnist_train_dataloader, model, opt)

    for epcoh_num in range(epochs):
        test_error_rate, test_loss = epoch(mnist_test_dataloader, model)

    return train_error_rate, train_loss, test_error_rate, test_loss
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
