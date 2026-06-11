# Schedulers in PyTorch

Welcome to this exploration of learning rate schedulers in PyTorch! As you have learned in the lectures, managing learning rates is vital to optimizing model training, and PyTorch provides various tools to streamline this process. Learning rate schedulers are essential for dynamically adjusting learning rates according to the training phase, helping to facilitate faster convergence and improve generalization.

In this notebook, you will dive into the mechanics of different learning rate schedulers. Specifically, you will:

* Explore how schedulers like `StepLR`, `CosineAnnealingLR`, and `ReduceLROnPlateau` adjust the learning rate to maintain training efficiency.

* Use these schedulers within a typical training loop.

* Learn to visualize training dynamics, demonstrating the impact of schedulers on model performance compared to a constant learning rate.

By the end of this notebook, you will have practical experience in applying learning rate schedulers, enhancing your understanding of their role in model training.


```python
import sys
import time
import warnings

# Redirect stderr to a black hole to catch other potential messages
class BlackHole:
    def write(self, message):
        pass
    def flush(self):
        pass
sys.stderr = BlackHole()

# Ignore Python-level UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)
```


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import helper_utils

helper_utils.set_seed(42)
```


```python
# # Check device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using device: MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print(f"Using device: CPU")
```

    Using device: CUDA


## A deeper dive into the training process

In this section you will make a deeper analysis of the experiments we have done in the previous lab. 
You will use a simple model (`SimpleCNN`) and the CIFAR-10 dataset, but now we will inspect the training process more closely by plotting the **training loss** and **validation accuracy** curves.

This will help you understand how the model learns over time and how different learning rates affect the training dynamics.

In the following cell we find `SimpleCNN` model definition, and `evaluate_epoch` function that you can use to evaluate the model on the validation set *after each epoch*.
`train_and_evaluate` function is provided to train the model for a specified number of epochs and evaluate it during training.

`get_data_loaders`and `train_epoch` functions are provided from `helper_utils` module. 
They are used to load the CIFAR-10 dataset and train the model for one epoch, respectively.


```python
class SimpleCNN(nn.Module):
    """A simple Convolutional Neural Network (CNN) architecture.

    This class defines a two-layer CNN with max pooling, dropout, and
    fully connected layers, suitable for basic image classification tasks.
    """
    def __init__(self):
        """Initializes the layers of the neural network."""
        # Initialize the parent nn.Module class
        super(SimpleCNN, self).__init__()
        # First convolutional layer (3 input channels, 16 output channels, 3x3 kernel)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # Second convolutional layer (16 input channels, 32 output channels, 3x3 kernel)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Max pooling layer with a 2x2 window and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        # First fully connected (linear) layer
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        # Second fully connected (linear) layer, serving as the output layer
        self.fc2 = nn.Linear(64, 10)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        """Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, 3, height, width).

        Returns:
            torch.Tensor: The output logits from the network.
        """
        # Apply first convolution, ReLU activation, and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply second convolution, ReLU activation, and max pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the feature maps for the fully connected layers
        x = x.view(-1, 32 * 8 * 8)
        # Apply the first fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        # Apply dropout for regularization
        x = self.dropout(x)
        # Apply the final output layer
        x = self.fc2(x)
        return x


def train_and_evaluate(learning_rate, device, n_epochs=25, batch_size=128, p_bar=None):
    """Orchestrates the training and evaluation of a model for a given configuration.

    This function handles the end-to-end workflow: setting a random seed,
    initializing the model, optimizer, loss function, and dataloaders, and then
    running the main training loop.

    Args:
        learning_rate (float): The learning rate for the optimizer.
        device: The device (e.g., 'cuda' or 'cpu') for training and evaluation.
        n_epochs (int, optional): The number of training epochs. Defaults to 25.
        batch_size (int, optional): The batch size for dataloaders. Defaults to 128.
        p_bar (optional): An existing progress bar handler. Defaults to None.

    Returns:
        dict: A dictionary containing the training and validation history
              (loss and accuracy).
    """
    # Set the random seed for reproducibility
    helper_utils.set_seed(42)

    # Initialize the model and move it to the specified device
    model = SimpleCNN().to(device)

    # Define the optimizer with the specified learning rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Prepare the training and validation dataloaders
    train_loader, val_loader = helper_utils.get_dataset_dataloaders(
        batch_size=batch_size
    )

    # Call the main training loop to train the model and get the history
    history = helper_utils.train_model(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        loss_fcn=loss_fn,
        device=device,
        n_epochs=n_epochs,
        p_bar=p_bar
    )

    # Return the collected training history
    return history
```

Now the model is trained for several epochs for each of the specified learning rates.
The results are stored in `training_curves`, and the curves (train loss and validation accuracy) are plotted for each learning rate.


```python
# Different learning rates to be analyzed
learning_rates = [0.0002, 0.001, 0.005] # Small, medium, and large learning rates

training_curves = []
n_epochs = 25
batch_size = 128

p_bar = helper_utils.get_p_bar(n_epochs)

# Get the total number of learning rates to check against the index
num_learning_rates = len(learning_rates)

# Train and evaluate the model for each learning rate
for i, lr in enumerate(learning_rates):
    print(f"\nTraining with learning rate: {lr}\n")
    history = train_and_evaluate(learning_rate=lr, n_epochs=n_epochs, batch_size=batch_size, device=device, p_bar=p_bar)
    training_curves.append(history)
    # Only reset the progress bar if it's NOT the last iteration
    if i < num_learning_rates - 1:
        p_bar.reset()
```


    Current Epoch:   0%|          | 0/25 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/10 [00:00<?, ?it/s]


    
    Training with learning rate: 0.0002
    



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 5: Training loss: 1.7609, Training accuracy: 0.3680
    At epoch 5: Validation loss: 1.6639, Validation accuracy: 0.4135



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 10: Training loss: 1.5487, Training accuracy: 0.4532
    At epoch 10: Validation loss: 1.5010, Validation accuracy: 0.4720



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 15: Training loss: 1.4556, Training accuracy: 0.4739
    At epoch 15: Validation loss: 1.4352, Validation accuracy: 0.4910



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 20: Training loss: 1.3768, Training accuracy: 0.4999
    At epoch 20: Validation loss: 1.3790, Validation accuracy: 0.5100



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 25: Training loss: 1.3217, Training accuracy: 0.5186
    At epoch 25: Validation loss: 1.3449, Validation accuracy: 0.5195
    Training complete
    
    
    Training with learning rate: 0.001
    



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 5: Training loss: 1.4896, Training accuracy: 0.4620
    At epoch 5: Validation loss: 1.4181, Validation accuracy: 0.4935



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 10: Training loss: 1.2878, Training accuracy: 0.5399
    At epoch 10: Validation loss: 1.2975, Validation accuracy: 0.5415



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 15: Training loss: 1.1556, Training accuracy: 0.5766
    At epoch 15: Validation loss: 1.2240, Validation accuracy: 0.5635



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 20: Training loss: 1.0150, Training accuracy: 0.6270
    At epoch 20: Validation loss: 1.1879, Validation accuracy: 0.5805



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 25: Training loss: 0.8937, Training accuracy: 0.6754
    At epoch 25: Validation loss: 1.2198, Validation accuracy: 0.5850
    Training complete
    
    
    Training with learning rate: 0.005
    



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 5: Training loss: 1.4318, Training accuracy: 0.4724
    At epoch 5: Validation loss: 1.3777, Validation accuracy: 0.5065



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 10: Training loss: 1.1757, Training accuracy: 0.5703
    At epoch 10: Validation loss: 1.3072, Validation accuracy: 0.5390



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 15: Training loss: 1.0137, Training accuracy: 0.6281
    At epoch 15: Validation loss: 1.3769, Validation accuracy: 0.5470



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 20: Training loss: 0.9199, Training accuracy: 0.6504
    At epoch 20: Validation loss: 1.4094, Validation accuracy: 0.5410



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 25: Training loss: 0.7902, Training accuracy: 0.6925
    At epoch 25: Validation loss: 1.5786, Validation accuracy: 0.5425
    Training complete
    



```python
colors = ['blue', 'orange', 'red']
labels = ['Low LR', 'Medium LR', 'High LR']

helper_utils.plot_learning_curves(colors, labels, training_curves)

# Each color corresponds to a different learning rate: blue for low, orange for medium, and red for high.
```


    
![png](C2_M1_Lab_2_Schedulers_files/C2_M1_Lab_2_Schedulers_9_0.png)
    


### Learning Rate Analysis

Each color represents a different learning rate: blue for low, orange for medium, and red for high.

- **Low Learning Rate:**
The validation accuracy improves gradually and may require a significant amount of time (many epochs) to reach its peak.

- **Medium Learning Rate:**
The validation accuracy increases quickly, and by the end of the 25 epochs, it is outperforming the other two models.

- **High Learning Rate:**
Initially, validation accuracy is the highest during the first few epochs but eventually levels off, indicating instability where the model struggles to improve with such a high learning rate.

Among these options, the medium learning rate strikes the best balance between learning speed and generalization. Could you achieve a better model by combining these learning rates—starting with a high rate for the first few epochs, then transitioning to medium, and finally to low?

## Schedulers
In the previous section, you explored how the choice of learning rate impacts model performance. A high learning rate can lead to instability, while a low one might slow down training or yield suboptimal outcomes. Learning rate schedulers offer a solution by automatically adjusting the learning rate throughout training, initially setting it high for rapid learning and then reducing it to enhance convergence and generalization.

In this lab, you'll investigate three different schedulers. Let's start with the `StepLR` scheduler, which lowers the learning rate by a specified factor at regular intervals—specifically, after a set number of epochs. You'll repeat the training process from before, but this time you'll employ a scheduler that decreases the learning rate by a factor of 0.2 every 10 epochs.


```python
helper_utils.set_seed(42)

# Initialize the model, optimizer, loss function, and dataloaders
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005) # start with a high learning rate
```

Schedulers are built into the PyTorch `torch.optim.lr_scheduler` module and can be easily integrated into your training loop.
You typically initialize a scheduler with the optimizer and parameters that define how the learning rate changes over time.

The `StepLR` scheduler, for example:
  - Takes the optimizer as input.
  - Requires `step_size`, the number of epochs before reducing the learning rate.
  - Uses a `gamma` factor, the factor by which the learning rate is reduced.


```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2) # reduce the learning rate by 20% it's prior value
```

The training process is similar to the previous one, but the learning rate is adjusted by the scheduler after each epoch using the `scheduler.step()` method.
The current learning rate can be accessed using `scheduler.get_last_lr()`.

The model is trained for several epochs with the `StepLR` scheduler, and the training curves are plotted to compare the results with the previous experiments for the medium learning rate.


```python
loss_fn = nn.CrossEntropyLoss()

train_loader, val_loader = helper_utils.get_dataset_dataloaders(batch_size=batch_size)

history_LR = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "lr": [],
}


pbar = helper_utils.NestedProgressBar(
    total_epochs=n_epochs,
    total_batches=len(train_loader),
    epoch_message_freq=5,
    mode="train",
)

for epoch in range(n_epochs):
    pbar.update_epoch(epoch+1)

    # Train the model for one epoch
    train_loss, train_acc = helper_utils.train_epoch(model, train_loader, optimizer, loss_fn, device, pbar)

    # Evaluate the model on the validation set
    val_loss, val_acc = helper_utils.evaluate_epoch(model, val_loader, loss_fn, device)

    # Get the current learning rate BEFORE stepping the scheduler.
    # This captures the LR that was just used for the training epoch above.
    current_lr = scheduler.get_last_lr()[0]
    
    # Step the scheduler (updates the LR for the NEXT epoch)
    scheduler.step()
    
    pbar.maybe_log_epoch(epoch=epoch+1, message=f"At epoch {epoch+1}: Training loss: {train_loss:.4f}, Training accuracy: {train_acc:.4f}, LR: {current_lr:.6f}")

    pbar.maybe_log_epoch(epoch=epoch+1, message=f"At epoch {epoch+1}: Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")

    history_LR["train_loss"].append(train_loss)
    history_LR["train_acc"].append(train_acc)
    history_LR["val_loss"].append(val_loss)
    history_LR["val_acc"].append(val_acc)
    history_LR["lr"].append(current_lr)

pbar.close('Training complete with StepLR scheduler')
```


    Current Epoch:   0%|          | 0/25 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/63 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 5: Training loss: 1.4219, Training accuracy: 0.4788, LR: 0.005000
    At epoch 5: Validation loss: 1.3811, Validation accuracy: 0.4995



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 10: Training loss: 1.1874, Training accuracy: 0.5716, LR: 0.005000
    At epoch 10: Validation loss: 1.2772, Validation accuracy: 0.5620



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 15: Training loss: 0.9617, Training accuracy: 0.6460, LR: 0.001000
    At epoch 15: Validation loss: 1.2413, Validation accuracy: 0.5735



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 20: Training loss: 0.8994, Training accuracy: 0.6646, LR: 0.001000
    At epoch 20: Validation loss: 1.2540, Validation accuracy: 0.5770



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 25: Training loss: 0.8351, Training accuracy: 0.6881, LR: 0.000200
    At epoch 25: Validation loss: 1.2711, Validation accuracy: 0.5880
    Training complete with StepLR scheduler



```python
idx = 1
history_constant = training_curves[idx]

colors = ['orange', 'green']
labels = ['Medium LR', 'Step LR']
histories = [history_constant, history_LR]

helper_utils.plot_learning_curves(colors, labels, histories)
```


    
![png](C2_M1_Lab_2_Schedulers_files/C2_M1_Lab_2_Schedulers_17_0.png)
    


### Analysis of Medium LR and StepLR Scheduler

The plots show a comparison between training with a constant learning rate (orange) and using a `StepLR` scheduler (green). With StepLR, the training loss is significantly lower than with a fixed learning rate. You can also notice a few dips at the epoch 10 and 20 when the learning rate changes.  Although the validation accuracy starts off stronger, both models eventually converge to similar performance levels at the last epoch. Now, let's examine two more schedulers to compare their effectiveness.

### Other Schedulers

There are many other schedulers available in PyTorch, such as `CosineAnnealingLR`, and `ReduceLROnPlateau`, each with its own strategy for adjusting the learning rate.
You can explore these options in the [PyTorch documentation](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate).

Below `train_and_evaluate_with_scheduler` function extends the previous `train_and_evaluate` function to include a learning rate scheduler.


```python
def train_and_evaluate_with_scheduler(model, optimizer, scheduler, device, n_epochs=25, batch_size=128):
    """Trains and evaluates a model using a learning rate scheduler.

    Args:
        model: The neural network model to be trained.
        optimizer: The optimization algorithm.
        scheduler: The learning rate scheduler.
        device: The computing device ('cuda' or 'cpu') to run the training on.
        n_epochs: The total number of training epochs.
        batch_size: The number of samples per batch in the data loaders.

    Returns:
        A dictionary containing the training and validation history
        (loss, accuracy, and learning rate) for each epoch.
    """
    # Set the random seed for reproducibility
    helper_utils.set_seed(10)

    # Define the loss function
    loss_fn = nn.CrossEntropyLoss()
    # Prepare the training and validation data loaders
    train_loader, val_loader = helper_utils.get_dataset_dataloaders(
        batch_size=batch_size
    )

    # Initialize a dictionary to store training and validation history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        'lr': [],
    }

    # Initialize the progress bar for monitoring training
    pbar = helper_utils.NestedProgressBar(
        total_epochs=n_epochs,
        total_batches=len(train_loader),
        epoch_message_freq=5,
        mode="train",
    )

    # Loop through the specified number of epochs
    for epoch in range(n_epochs):

        # Update the progress bar for the current epoch
        pbar.update_epoch(epoch+1)

        # Train the model for one epoch
        train_loss, train_acc = helper_utils.train_epoch(model, train_loader, optimizer, loss_fn, device, pbar)
        # Evaluate the model on the validation set
        val_loss, val_acc = helper_utils.evaluate_epoch(model, val_loader, loss_fn, device)
        
        # Retrieve the current learning rate from the scheduler
        current_lr = scheduler.get_last_lr()[0]

        # Update the learning rate based on the scheduler type
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            # For schedulers that monitor a metric, pass the metric to the step function
            scheduler.step(val_acc)
        else:
            # For other schedulers, call the step function without arguments
            scheduler.step()
        
        # Log the training metrics for the current epoch, including the learning rate
        pbar.maybe_log_epoch(epoch=epoch+1, message=f"At epoch {epoch+1}: Training loss: {train_loss:.4f}, Training accuracy: {train_acc:.4f}, LR: {current_lr:.6f}")

        # Log the validation metrics for the current epoch, including the learning rate
        pbar.maybe_log_epoch(epoch=epoch+1, message=f"At epoch {epoch+1}: Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")

        # Append the metrics for the current epoch to the history dictionary
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history['lr'].append(current_lr)

    # Close the progress bar upon completion of training
    pbar.close('Training complete!')
    # Return the collected training and validation history
    return history
```

<br>

In the following cells, you will run experiments using both `CosineAnnealingLR` and `ReduceLROnPlateau` schedulers. The training curves for each scheduler will be generated and compared to visualize their effects on model performance.

- **CosineAnnealingLR:**  
    This scheduler adjusts the learning rate following a cosine curve, gradually reducing it from the initial value to a minimum over a specified number of epochs.  
    Parameters used in this notebook are:  
    - `optimizer`: The optimizer to schedule.  
    - `T_max`: Number of epochs for one cycle of cosine annealing.
    - `eta_min`: Minimum learning rate.
>
- **ReduceLROnPlateau:**  
    This scheduler monitors a metric (such as validation loss) and reduces the learning rate by a factor if no improvement is seen for a set number of epochs.  
    Parameters used in this notebook:  
    - `optimizer`: The optimizer to schedule.  
    - `mode`: Whether to look for a decrease (`min`) or increase (`max`) in the monitored metric.  
    - `factor`: Factor by which the learning rate will be reduced.  
    - `patience`: Number of epochs with no improvement before reducing the learning rate.

For more details and additional parameters, see the [PyTorch documentation](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate).


```python
# CosineAnnealingLR
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005)

scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min = 0.0002)

history_cosine = train_and_evaluate_with_scheduler(
    model, optimizer, scheduler_cosine, device, n_epochs=n_epochs, batch_size=batch_size
)
```


    Current Epoch:   0%|          | 0/25 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/63 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 5: Training loss: 1.2477, Training accuracy: 0.5474, LR: 0.004703
    At epoch 5: Validation loss: 1.2530, Validation accuracy: 0.5485



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 10: Training loss: 0.8982, Training accuracy: 0.6761, LR: 0.003622
    At epoch 10: Validation loss: 1.1856, Validation accuracy: 0.6100



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 15: Training loss: 0.6511, Training accuracy: 0.7504, LR: 0.002150
    At epoch 15: Validation loss: 1.3028, Validation accuracy: 0.6095



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 20: Training loss: 0.5060, Training accuracy: 0.8106, LR: 0.000850
    At epoch 20: Validation loss: 1.4114, Validation accuracy: 0.6030



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 25: Training loss: 0.4481, Training accuracy: 0.8357, LR: 0.000219
    At epoch 25: Validation loss: 1.4672, Validation accuracy: 0.6080
    Training complete!



```python
# ReduceLROnPlateau
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005)

scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3)

history_plateau = train_and_evaluate_with_scheduler(
    model, optimizer, scheduler_plateau, device, n_epochs=n_epochs, batch_size=batch_size
)
```


    Current Epoch:   0%|          | 0/25 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/63 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 5: Training loss: 1.2789, Training accuracy: 0.5411, LR: 0.005000
    At epoch 5: Validation loss: 1.2083, Validation accuracy: 0.5710



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 10: Training loss: 0.9322, Training accuracy: 0.6621, LR: 0.005000
    At epoch 10: Validation loss: 1.1993, Validation accuracy: 0.5850



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 15: Training loss: 0.6065, Training accuracy: 0.7751, LR: 0.001000
    At epoch 15: Validation loss: 1.2520, Validation accuracy: 0.6015



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 20: Training loss: 0.5190, Training accuracy: 0.8094, LR: 0.001000
    At epoch 20: Validation loss: 1.3407, Validation accuracy: 0.6055



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    At epoch 25: Training loss: 0.4636, Training accuracy: 0.8260, LR: 0.000200
    At epoch 25: Validation loss: 1.3853, Validation accuracy: 0.6120
    Training complete!



```python
labels = ['Medium LR', 'StepLR', 'CosineAnnealingLR', 'ReducedLRonPlateau']
colors = ['orange', 'green', 'blue', 'purple']

training_curves_new = [history_constant, history_LR, history_cosine, history_plateau]

helper_utils.plot_learning_curves(colors, labels, training_curves_new)
```


    
![png](C2_M1_Lab_2_Schedulers_files/C2_M1_Lab_2_Schedulers_23_0.png)
    


<br>

Upon analyzing the validation accuracy, you see that both the `ReduceLROnPlateau` and `CosineAnnealingLR` schedulers yield improved results when you observe the last epoch. Keep in mind that each scheduler has tunable hyperparameters as well. 

Below, you illustrate how the learning rate changes over the epochs for each scheduler. The `CosineAnnealingLR` scheduler reduces the learning rate according to a cosine decay pattern, while `ReduceLROnPlateau` adjusts it dynamically based on validation performance.


```python
helper_utils.plot_learning_rates_curves(training_curves_new, colors, labels)
```


    
![png](C2_M1_Lab_2_Schedulers_files/C2_M1_Lab_2_Schedulers_25_0.png)
    


<br>

In summary:

- `StepLR` is effective for reducing the learning rate at fixed intervals in a stepwise manner.

- `CosineAnnealingLR` offers a smooth decay without sudden changes, which is beneficial for fine-tuning the model as training concludes. It's particularly advantageous for long training sessions where gradual adjustments can enhance convergence.

- `ReduceLROnPlateau` is useful when validation performance plateaus, as it adjusts the learning rate based on validation metrics, potentially improving generalization. It is responsive to the model's performance, allowing for dynamic adjustments based on training dynamics.

# Conclusion

Congratulations on completing this in-depth exploration of PyTorch's learning rate schedulers! In this notebook, you examined how dynamically adjusting the learning rate can enhance training efficiency, speed up convergence, and improve generalization when compared to a constant learning rate. By utilizing schedulers and visualizing their effects on training and validation metrics, you observed how they contribute to improved model performance. Additionally, you gained valuable experience in tuning scheduler hyperparameters to suit various training scenarios.
