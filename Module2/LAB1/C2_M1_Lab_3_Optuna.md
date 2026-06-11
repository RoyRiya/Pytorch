# Hyperparameter Optimization with Optuna

Welcome to the Optuna-based hyperparameter optimization tutorial! In this interactive notebook, you will explore world of hyperparameter tuning for a Convolutional Neural Network (CNN) specifically aimed at image classification using the CIFAR-10 dataset. Hyperparameter optimization is pivotal in enhancing model performance, making your models more accurate and efficient.

Optuna, a robust and versatile library, plays a central role in automating and streamlining this process. It empowers you to navigate through complex hyperparameter spaces with ease. In this tutorial, you will engage with Optuna's core functionalities, and you'll also have the opportunity to construct a flexible CNN architecture. This adaptable design is essential for understanding how models can be fine-tuned effortlessly to suit various hyperparameter configurations.

Throughout this session, you will:
- Learn how to set up and execute an Optuna study, incorporating all essential elements required for effective hyperparameter optimization.
- Perform a thorough analysis of the results to evaluate how different hyperparameters influence model performance, gaining insights into their practical impact.

Additionally, this tutorial includes an optional section where you will compare two prevalent methods of hyperparameter optimization: Optuna's default sampling method (Tree-structured Parzen Estimator, or TPE) and the traditional Grid Search method. This comparison will not only highlight the strengths of Optuna but also provide a clearer perspective on how it can outperform conventional optimization techniques.


## Import Libraries


```python
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
import helper_utils
import torch.nn.functional as F
from pprint import pprint

helper_utils.set_seed(15)
```


```python
# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
```

    cuda


## Hyperparameter Optimization for CNNs on CIFAR-10

In this section, you explore the vital task of finding the optimal hyperparameters for a Convolutional Neural Network (CNN) tailored to the CIFAR-10 dataset. 
Utilizing Optuna, a sophisticated framework for hyperparameter optimization, your goal is to streamline and automate the process, ensuring efficiency and effectiveness. 
The selection of hyperparameters is notably intensive computationally and depends on various factors including the architecture of the model, the dataset characteristics, and the specific training processes involved. These elements, collectively and individually, have significant impacts on the performance outcomes of the model.

### Defining a Flexible CNN Architecture

The model architecture here is deliberately designed to be flexible, accommodating variability in its layers which is pivotal for adapting to different hyperparameter configurations suggested by Optuna during optimization trials.  
The architecture is defined in a modular manner, allowing for easy adjustments and experimentation with different layer configurations, activation functions, and other hyperparameters. 

`FlexibleCNN` is a class that encapsulates the architecture of the CNN model:

* **`__init__`**: The constructor initializes the model's feature extraction layers.
>    * It constructs a series of convolutional blocks based on the `n_layers` parameter. Each block is a sequence of `nn.Conv2d`, `nn.ReLU`, and `nn.MaxPool2d`.
>    * The `in_channels` for each block is set to the `out_channels` of the preceding block to ensure a seamless data flow.
>    * All blocks are combined into a single `nn.Sequential` module assigned to the `.features` attribute, which handles feature extraction.
>    * The classifier, `.classifier`, is initially set to `None` and will be constructed dynamically later.
 * **`_create_classifier`**: This helper method dynamically builds the classifier part of the network.
>    * It's called during the first forward pass once the input size for the linear layers is known.
 * **`forward`**: This method defines the forward pass of the model.
>    * The input `x` first passes through the `.features` layers.
>    * The output from the feature extractor is flattened to determine the input size for the classifier.
>    * If the `.classifier` has not been created yet, it calls `_create_classifier` to build it on the fly.
>    * Finally, the flattened data is passed through the `.classifier` to produce the final output.


```python
class FlexibleCNN(nn.Module):
    """
    A flexible Convolutional Neural Network with a dynamically created classifier.

    This CNN's architecture is defined by the provided hyperparameters,
    allowing for a variable number of convolutional layers. The classifier
    (fully connected layers) is constructed during the first forward pass
    to adapt to the output size of the convolutional feature extractor.
    """
    def __init__(self, n_layers, n_filters, kernel_sizes, dropout_rate, fc_size):
        """
        Initializes the feature extraction part of the CNN.

        Args:
            n_layers: The number of convolutional blocks to create.
            n_filters: A list of integers specifying the number of output
                       filters for each convolutional block.
            kernel_sizes: A list of integers specifying the kernel size for
                          each convolutional layer.
            dropout_rate: The dropout probability to be used in the classifier.
            fc_size: The number of neurons in the hidden fully connected layer.
        """
        super(FlexibleCNN, self).__init__()

        # Initialize an empty list to hold the convolutional blocks
        blocks = []
        # Set the initial number of input channels for RGB images
        in_channels = 3

        # Loop to construct each convolutional block
        for i in range(n_layers):

            # Get the parameters for the current convolutional layer
            out_channels = n_filters[i]
            kernel_size = kernel_sizes[i]
            # Calculate padding to maintain the input spatial dimensions ('same' padding)
            padding = (kernel_size - 1) // 2

            # Define a block as a sequence of Conv, ReLU, and MaxPool layers
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
            # Add the newly created block to the list
            blocks.append(block)

            # Update the number of input channels for the next block
            in_channels = out_channels

        # Combine all blocks into a single feature extractor module
        self.features = nn.Sequential(*blocks)

        # Store hyperparameters needed for building the classifier later
        self.dropout_rate = dropout_rate
        self.fc_size = fc_size

        # The classifier will be initialized dynamically in the forward pass
        self.classifier = None

    def _create_classifier(self, flattened_size, device):
        """
        Dynamically creates and initializes the classifier part of the network.

        This helper method is called during the first forward pass to build the
        fully connected layers based on the feature map size from the
        convolutional base.

        Args:
            flattened_size: The number of input features for the first linear
                            layer, determined from the flattened feature map.
            device: The device to which the new classifier layers should be moved.
        """
        # Define the classifier's architecture
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(flattened_size, self.fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fc_size, 10)  # Assumes 10 output classes (e.g., CIFAR-10)
        ).to(device)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x: The input tensor of shape (batch_size, channels, height, width).

        Returns:
            The output logits from the classifier.
        """
        # Get the device of the input tensor to ensure consistency
        device = x.device

        # Pass the input through the feature extraction layers
        x = self.features(x)

        # Flatten the feature map to prepare it for the fully connected layers
        flattened = torch.flatten(x, 1)
        flattened_size = flattened.size(1)

        # If the classifier has not been created yet, initialize it
        if self.classifier is None:
            self._create_classifier(flattened_size, device)

        # Pass the flattened features through the classifier to get the final output
        return self.classifier(flattened)
```

## Defining the Optuna Objective Function

The objective function is the core of the hyperparameter optimization process, being the function that Optuna will repeatedly call to evaluate different hyperparameter configurations.
This function encapsulates the entire training and evaluation process, including the definition of the CNN model architecture, the optimizer, the data loaders, the training loop, and the evaluation metrics.
Within this function, you define the search space for hyperparameters using `trial.suggest_*` methods, which allow Optuna to sample hyperparameters from a defined range or set of values. 
For a full list of available `suggest_*` methods, you can refer to the [Optuna documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html).

*The objective function is designed to return a single scalar value, which represents the performance of the model for the given hyperparameters*. 
In your case, the aim to maximize the accuracy of the model on the validation set, which is computed using the `evaluate_accuracy` function.

**Dynamic Layer Initialization**: A noteworthy addition to this objective function is the initialization step involving a dummy input. Because the `FlexibleCNN` creates its classifier layers dynamically during the first forward pass, these parameters do not exist immediately after the model is instantiated.
- **Why use a dummy input?** Passing data through the model forces it to calculate the flattened feature size and build the classifier layers. You must do this *before* defining the optimizer so that `model.parameters()` includes the classifier weights. Otherwise, the optimizer would only track the feature extractor, leaving the classifier untrained.
>
- **Why these dimensions?** The tensor `torch.randn(1, 3, 32, 32)` is used to mimic the structure of the CIFAR-10 dataset. It represents a single image (batch size of 1) with 3 color channels (RGB) and a resolution of `32x32` pixels.

Observe that some hyperparameters are defined as fixed values, such as the number of epochs, the batch size, and the learning rate.


```python
def objective(trial, device):
    """
    Defines the objective function for hyperparameter optimization using Optuna.

    For each trial, this function samples a set of hyperparameters,
    constructs a model, trains it for a fixed number of epochs, evaluates
    its performance on a validation set, and returns the accuracy. Optuna
    uses the returned accuracy to guide its search for the best
    hyperparameter combination.

    Args:
        trial: An Optuna `Trial` object, used to sample hyperparameters.
        device: The device ('cpu' or 'cuda') for model training and evaluation.

    Returns:
        The validation accuracy of the trained model as a float.
    """
    # Sample hyperparameters for the feature extractor using the Optuna trial
    n_layers = trial.suggest_int("n_layers", 1, 3)
    n_filters = [trial.suggest_int(f"n_filters_{i}", 16, 128) for i in range(n_layers)]
    kernel_sizes = [trial.suggest_categorical(f"kernel_size_{i}", [3, 5]) for i in range(n_layers)]

    # Sample hyperparameters for the classifier
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    fc_size = trial.suggest_int("fc_size", 64, 256)

    # Instantiate the model with the sampled hyperparameters
    model = FlexibleCNN(n_layers, n_filters, kernel_sizes, dropout_rate, fc_size).to(device)

    # Initialize the dynamic classifier layer by passing a dummy input through the model
    # This ensures all parameters are instantiated before the optimizer is defined
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    model(dummy_input)

    # Define fixed training parameters: learning rate, loss function, and optimizer
    learning_rate = 0.001
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define fixed data loading parameters and create data loaders
    batch_size = 128
    train_loader, val_loader = helper_utils.get_dataset_dataloaders(batch_size=batch_size)

    # Define the fixed number of epochs for training
    n_epochs = 10
    # Train the model using a helper function
    helper_utils.train_model(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_loader,
        n_epochs=n_epochs,
        loss_fcn=loss_fcn,
        device=device
    )

    # Evaluate the trained model's accuracy on the validation set
    accuracy = helper_utils.evaluate_accuracy(model, val_loader, device)
    
    # Return the final accuracy for this trial
    return accuracy
```

## Running the Optuna Study

Once that the objective function is defined, an Optuna study is created to manage the hyperparameter optimization process.
The study is responsible for running the objective function multiple times with different hyperparameter configurations, allowing Optuna to explore the search space and find the best hyperparameters.
In this case, your goal is to **maximize the accuracy** of the CNN model on the CIFAR-10 dataset, this is why we use `direction='maximize'` when creating the study.
The `optimize` method of the study is called to start the optimization process, which will run the objective function for a defined number of trials.

A lambda function is used to pass the device to the objective function, allowing the model to be trained on the specified device
*Note*: you can also pass other parameters to the objective function using the lambda function, if needed.

**NOTE:** the code below will take about 8 minutes to run.


```python
# Create a study object and optimize the objective function
study = optuna.create_study(direction='maximize') # The goal in this case is to maximize accuracy

# Start the optimization process (it takes about 8 minutes for 20 trials)
n_trials = 20
study.optimize(lambda trial: objective(trial, device), n_trials=n_trials) # use more trials in practice
```

    [I 2026-06-05 09:13:55,858] A new study created in memory with name: no-name-d2d4ccca-d450-4342-84c4-3c903a6eecc2



    Current Epoch:   0%|          | 0/10 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/63 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 1.2694
    Epoch 10 - Train Loss: 0.9294
    Training complete!
    



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    [I 2026-06-05 09:14:11,600] Trial 0 finished with value: 0.606 and parameters: {'n_layers': 2, 'n_filters_0': 36, 'n_filters_1': 128, 'kernel_size_0': 5, 'kernel_size_1': 5, 'dropout_rate': 0.4694474685337693, 'fc_size': 133}. Best is trial 0 with value: 0.606.


    Evaluation complete.



    Current Epoch:   0%|          | 0/10 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/63 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 1.3830
    Epoch 10 - Train Loss: 1.1054
    Training complete!
    



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    [I 2026-06-05 09:14:27,038] Trial 1 finished with value: 0.555 and parameters: {'n_layers': 3, 'n_filters_0': 65, 'n_filters_1': 39, 'n_filters_2': 92, 'kernel_size_0': 5, 'kernel_size_1': 3, 'kernel_size_2': 3, 'dropout_rate': 0.3376226216055599, 'fc_size': 123}. Best is trial 0 with value: 0.606.


    Evaluation complete.



    Current Epoch:   0%|          | 0/10 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/63 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 1.4643
    Epoch 10 - Train Loss: 1.2544
    Training complete!
    



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    [I 2026-06-05 09:14:43,668] Trial 2 finished with value: 0.5735 and parameters: {'n_layers': 2, 'n_filters_0': 111, 'n_filters_1': 44, 'kernel_size_0': 5, 'kernel_size_1': 3, 'dropout_rate': 0.4449728935581857, 'fc_size': 76}. Best is trial 0 with value: 0.606.


    Evaluation complete.



    Current Epoch:   0%|          | 0/10 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/63 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 1.2826
    Epoch 10 - Train Loss: 0.8616
    Training complete!
    



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    [I 2026-06-05 09:15:00,894] Trial 3 finished with value: 0.6055 and parameters: {'n_layers': 3, 'n_filters_0': 115, 'n_filters_1': 107, 'n_filters_2': 99, 'kernel_size_0': 5, 'kernel_size_1': 3, 'kernel_size_2': 5, 'dropout_rate': 0.2634881615600053, 'fc_size': 82}. Best is trial 0 with value: 0.606.


    Evaluation complete.



    Current Epoch:   0%|          | 0/10 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/63 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 1.4213
    Epoch 10 - Train Loss: 1.1146
    Training complete!
    



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    [I 2026-06-05 09:15:17,216] Trial 4 finished with value: 0.5775 and parameters: {'n_layers': 3, 'n_filters_0': 90, 'n_filters_1': 101, 'n_filters_2': 71, 'kernel_size_0': 3, 'kernel_size_1': 3, 'kernel_size_2': 5, 'dropout_rate': 0.4918827235080332, 'fc_size': 152}. Best is trial 0 with value: 0.606.


    Evaluation complete.



    Current Epoch:   0%|          | 0/10 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/63 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 1.0102
    Epoch 10 - Train Loss: 0.4648
    Training complete!
    



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    [I 2026-06-05 09:15:33,798] Trial 5 finished with value: 0.612 and parameters: {'n_layers': 2, 'n_filters_0': 98, 'n_filters_1': 119, 'kernel_size_0': 3, 'kernel_size_1': 3, 'dropout_rate': 0.11348414124439268, 'fc_size': 184}. Best is trial 5 with value: 0.612.


    Evaluation complete.



    Current Epoch:   0%|          | 0/10 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/63 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 1.2160
    Epoch 10 - Train Loss: 0.8991
    Training complete!
    



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    [I 2026-06-05 09:15:49,645] Trial 6 finished with value: 0.5605 and parameters: {'n_layers': 1, 'n_filters_0': 106, 'kernel_size_0': 5, 'dropout_rate': 0.12251549814521218, 'fc_size': 64}. Best is trial 5 with value: 0.612.


    Evaluation complete.



    Current Epoch:   0%|          | 0/10 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/63 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 1.2118
    Epoch 10 - Train Loss: 0.9015
    Training complete!
    



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    [I 2026-06-05 09:16:04,252] Trial 7 finished with value: 0.552 and parameters: {'n_layers': 1, 'n_filters_0': 68, 'kernel_size_0': 3, 'dropout_rate': 0.3982673354112163, 'fc_size': 158}. Best is trial 5 with value: 0.612.


    Evaluation complete.



    Current Epoch:   0%|          | 0/10 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/63 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 1.3396
    Epoch 10 - Train Loss: 1.0502
    Training complete!
    



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    [I 2026-06-05 09:16:19,862] Trial 8 finished with value: 0.597 and parameters: {'n_layers': 3, 'n_filters_0': 76, 'n_filters_1': 92, 'n_filters_2': 66, 'kernel_size_0': 3, 'kernel_size_1': 3, 'kernel_size_2': 3, 'dropout_rate': 0.26040085533307233, 'fc_size': 103}. Best is trial 5 with value: 0.612.


    Evaluation complete.



    Current Epoch:   0%|          | 0/10 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/63 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 1.4189
    Epoch 10 - Train Loss: 1.1446
    Training complete!
    



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    [I 2026-06-05 09:16:34,483] Trial 9 finished with value: 0.5575 and parameters: {'n_layers': 3, 'n_filters_0': 51, 'n_filters_1': 29, 'n_filters_2': 38, 'kernel_size_0': 5, 'kernel_size_1': 5, 'kernel_size_2': 3, 'dropout_rate': 0.22959394578946435, 'fc_size': 142}. Best is trial 5 with value: 0.612.


    Evaluation complete.



    Current Epoch:   0%|          | 0/10 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/63 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 1.1551
    Epoch 10 - Train Loss: 0.6805
    Training complete!
    



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    [I 2026-06-05 09:16:48,697] Trial 10 finished with value: 0.565 and parameters: {'n_layers': 2, 'n_filters_0': 24, 'n_filters_1': 70, 'kernel_size_0': 3, 'kernel_size_1': 5, 'dropout_rate': 0.12382087222369069, 'fc_size': 215}. Best is trial 5 with value: 0.612.


    Evaluation complete.



    Current Epoch:   0%|          | 0/10 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/63 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 1.1429
    Epoch 10 - Train Loss: 0.6681
    Training complete!
    



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    [I 2026-06-05 09:17:03,088] Trial 11 finished with value: 0.5995 and parameters: {'n_layers': 2, 'n_filters_0': 23, 'n_filters_1': 128, 'kernel_size_0': 3, 'kernel_size_1': 5, 'dropout_rate': 0.1925602108165501, 'fc_size': 203}. Best is trial 5 with value: 0.612.


    Evaluation complete.



    Current Epoch:   0%|          | 0/10 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/63 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 1.1880
    Epoch 10 - Train Loss: 0.7719
    Training complete!
    



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    [I 2026-06-05 09:17:17,855] Trial 12 finished with value: 0.5845 and parameters: {'n_layers': 2, 'n_filters_0': 45, 'n_filters_1': 128, 'kernel_size_0': 3, 'kernel_size_1': 5, 'dropout_rate': 0.3595220750909184, 'fc_size': 187}. Best is trial 5 with value: 0.612.


    Evaluation complete.



    Current Epoch:   0%|          | 0/10 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/63 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 1.0506
    Epoch 10 - Train Loss: 0.5452
    Training complete!
    



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    [I 2026-06-05 09:17:34,062] Trial 13 finished with value: 0.597 and parameters: {'n_layers': 2, 'n_filters_0': 86, 'n_filters_1': 69, 'kernel_size_0': 5, 'kernel_size_1': 5, 'dropout_rate': 0.1741655763978607, 'fc_size': 253}. Best is trial 5 with value: 0.612.


    Evaluation complete.



    Current Epoch:   0%|          | 0/10 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/63 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 1.3316
    Epoch 10 - Train Loss: 1.1031
    Training complete!
    



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    [I 2026-06-05 09:17:47,898] Trial 14 finished with value: 0.5565 and parameters: {'n_layers': 1, 'n_filters_0': 41, 'kernel_size_0': 5, 'dropout_rate': 0.49882092577028775, 'fc_size': 178}. Best is trial 5 with value: 0.612.


    Evaluation complete.



    Current Epoch:   0%|          | 0/10 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/63 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 1.2270
    Epoch 10 - Train Loss: 0.8865
    Training complete!
    



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    [I 2026-06-05 09:18:05,975] Trial 15 finished with value: 0.6015 and parameters: {'n_layers': 2, 'n_filters_0': 126, 'n_filters_1': 114, 'kernel_size_0': 3, 'kernel_size_1': 5, 'dropout_rate': 0.3162138225299837, 'fc_size': 125}. Best is trial 5 with value: 0.612.


    Evaluation complete.



    Current Epoch:   0%|          | 0/10 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/63 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 1.1895
    Epoch 10 - Train Loss: 0.8091
    Training complete!
    



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    [I 2026-06-05 09:18:22,440] Trial 16 finished with value: 0.6 and parameters: {'n_layers': 2, 'n_filters_0': 97, 'n_filters_1': 86, 'kernel_size_0': 5, 'kernel_size_1': 3, 'dropout_rate': 0.3996851964833198, 'fc_size': 229}. Best is trial 5 with value: 0.612.


    Evaluation complete.



    Current Epoch:   0%|          | 0/10 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/63 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 1.3661
    Epoch 10 - Train Loss: 1.1325
    Training complete!
    



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    [I 2026-06-05 09:18:36,256] Trial 17 finished with value: 0.536 and parameters: {'n_layers': 1, 'n_filters_0': 16, 'kernel_size_0': 3, 'dropout_rate': 0.4328157200137368, 'fc_size': 178}. Best is trial 5 with value: 0.612.


    Evaluation complete.



    Current Epoch:   0%|          | 0/10 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/63 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 1.0456
    Epoch 10 - Train Loss: 0.6250
    Training complete!
    



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    [I 2026-06-05 09:18:50,659] Trial 18 finished with value: 0.5615 and parameters: {'n_layers': 1, 'n_filters_0': 61, 'kernel_size_0': 5, 'dropout_rate': 0.17273922147969664, 'fc_size': 122}. Best is trial 5 with value: 0.612.


    Evaluation complete.



    Current Epoch:   0%|          | 0/10 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/63 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 1.1780
    Epoch 10 - Train Loss: 0.8049
    Training complete!
    



    Evaluating:   0%|          | 0/16 [00:00<?, ?it/s]


    [I 2026-06-05 09:19:05,098] Trial 19 finished with value: 0.6115 and parameters: {'n_layers': 2, 'n_filters_0': 36, 'n_filters_1': 114, 'kernel_size_0': 3, 'kernel_size_1': 3, 'dropout_rate': 0.27843099615126987, 'fc_size': 139}. Best is trial 5 with value: 0.612.


    Evaluation complete.


### Analyzing the Results
After the optimization process is complete, you can analyze the results to understand which hyperparameters yielded the best performance.
The `study` object contains a wealth of information about the trials, including the hyperparameters sampled, the corresponding performance metrics, and the best trial.

You can access the full DataFrame of trials using the `trials_dataframe()` method, which provides a comprehensive overview of all the trials conducted during the optimization process.
This DataFrame includes columns for the trial number, hyperparameters, and the objective value (in our case, the accuracy).

To access the best hyperparameters and the best trial, you can use the `best_trial` attributes of the study object.

**Note:** these results may change every time you re-run the training study.


```python
# Extract the dataframe with the results
df = study.trials_dataframe()

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>number</th>
      <th>value</th>
      <th>datetime_start</th>
      <th>datetime_complete</th>
      <th>duration</th>
      <th>params_dropout_rate</th>
      <th>params_fc_size</th>
      <th>params_kernel_size_0</th>
      <th>params_kernel_size_1</th>
      <th>params_kernel_size_2</th>
      <th>params_n_filters_0</th>
      <th>params_n_filters_1</th>
      <th>params_n_filters_2</th>
      <th>params_n_layers</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.6060</td>
      <td>2026-06-05 09:13:55.859356</td>
      <td>2026-06-05 09:14:11.600685</td>
      <td>0 days 00:00:15.741329</td>
      <td>0.469447</td>
      <td>133</td>
      <td>5</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>36</td>
      <td>128.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.5550</td>
      <td>2026-06-05 09:14:11.601601</td>
      <td>2026-06-05 09:14:27.038641</td>
      <td>0 days 00:00:15.437040</td>
      <td>0.337623</td>
      <td>123</td>
      <td>5</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>65</td>
      <td>39.0</td>
      <td>92.0</td>
      <td>3</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.5735</td>
      <td>2026-06-05 09:14:27.039611</td>
      <td>2026-06-05 09:14:43.668537</td>
      <td>0 days 00:00:16.628926</td>
      <td>0.444973</td>
      <td>76</td>
      <td>5</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>111</td>
      <td>44.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.6055</td>
      <td>2026-06-05 09:14:43.669465</td>
      <td>2026-06-05 09:15:00.894213</td>
      <td>0 days 00:00:17.224748</td>
      <td>0.263488</td>
      <td>82</td>
      <td>5</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>115</td>
      <td>107.0</td>
      <td>99.0</td>
      <td>3</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.5775</td>
      <td>2026-06-05 09:15:00.894988</td>
      <td>2026-06-05 09:15:17.216454</td>
      <td>0 days 00:00:16.321466</td>
      <td>0.491883</td>
      <td>152</td>
      <td>3</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>90</td>
      <td>101.0</td>
      <td>71.0</td>
      <td>3</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.6120</td>
      <td>2026-06-05 09:15:17.217148</td>
      <td>2026-06-05 09:15:33.798672</td>
      <td>0 days 00:00:16.581524</td>
      <td>0.113484</td>
      <td>184</td>
      <td>3</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>98</td>
      <td>119.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.5605</td>
      <td>2026-06-05 09:15:33.799539</td>
      <td>2026-06-05 09:15:49.645708</td>
      <td>0 days 00:00:15.846169</td>
      <td>0.122515</td>
      <td>64</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>106</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.5520</td>
      <td>2026-06-05 09:15:49.646800</td>
      <td>2026-06-05 09:16:04.252114</td>
      <td>0 days 00:00:14.605314</td>
      <td>0.398267</td>
      <td>158</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>68</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0.5970</td>
      <td>2026-06-05 09:16:04.253028</td>
      <td>2026-06-05 09:16:19.862414</td>
      <td>0 days 00:00:15.609386</td>
      <td>0.260401</td>
      <td>103</td>
      <td>3</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>76</td>
      <td>92.0</td>
      <td>66.0</td>
      <td>3</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0.5575</td>
      <td>2026-06-05 09:16:19.863293</td>
      <td>2026-06-05 09:16:34.482942</td>
      <td>0 days 00:00:14.619649</td>
      <td>0.229594</td>
      <td>142</td>
      <td>5</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>51</td>
      <td>29.0</td>
      <td>38.0</td>
      <td>3</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>0.5650</td>
      <td>2026-06-05 09:16:34.483793</td>
      <td>2026-06-05 09:16:48.697668</td>
      <td>0 days 00:00:14.213875</td>
      <td>0.123821</td>
      <td>215</td>
      <td>3</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>24</td>
      <td>70.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>0.5995</td>
      <td>2026-06-05 09:16:48.698340</td>
      <td>2026-06-05 09:17:03.088259</td>
      <td>0 days 00:00:14.389919</td>
      <td>0.192560</td>
      <td>203</td>
      <td>3</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>23</td>
      <td>128.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>0.5845</td>
      <td>2026-06-05 09:17:03.089211</td>
      <td>2026-06-05 09:17:17.855547</td>
      <td>0 days 00:00:14.766336</td>
      <td>0.359522</td>
      <td>187</td>
      <td>3</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>45</td>
      <td>128.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>0.5970</td>
      <td>2026-06-05 09:17:17.856222</td>
      <td>2026-06-05 09:17:34.061951</td>
      <td>0 days 00:00:16.205729</td>
      <td>0.174166</td>
      <td>253</td>
      <td>5</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>86</td>
      <td>69.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>0.5565</td>
      <td>2026-06-05 09:17:34.062527</td>
      <td>2026-06-05 09:17:47.898086</td>
      <td>0 days 00:00:13.835559</td>
      <td>0.498821</td>
      <td>178</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>0.6015</td>
      <td>2026-06-05 09:17:47.898896</td>
      <td>2026-06-05 09:18:05.975166</td>
      <td>0 days 00:00:18.076270</td>
      <td>0.316214</td>
      <td>125</td>
      <td>3</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>126</td>
      <td>114.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>0.6000</td>
      <td>2026-06-05 09:18:05.976057</td>
      <td>2026-06-05 09:18:22.439994</td>
      <td>0 days 00:00:16.463937</td>
      <td>0.399685</td>
      <td>229</td>
      <td>5</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>97</td>
      <td>86.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>0.5360</td>
      <td>2026-06-05 09:18:22.440893</td>
      <td>2026-06-05 09:18:36.256076</td>
      <td>0 days 00:00:13.815183</td>
      <td>0.432816</td>
      <td>178</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>16</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>0.5615</td>
      <td>2026-06-05 09:18:36.256721</td>
      <td>2026-06-05 09:18:50.659241</td>
      <td>0 days 00:00:14.402520</td>
      <td>0.172739</td>
      <td>122</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>61</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>0.6115</td>
      <td>2026-06-05 09:18:50.659893</td>
      <td>2026-06-05 09:19:05.098452</td>
      <td>0 days 00:00:14.438559</td>
      <td>0.278431</td>
      <td>139</td>
      <td>3</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>36</td>
      <td>114.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>COMPLETE</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Extract and print the best trial
best_trial = study.best_trial

print("Best trial:")
print(f"  Value (Accuracy): {best_trial.value:.4f}")

print("  Hyperparameters:")
pprint(best_trial.params)
```

    Best trial:
      Value (Accuracy): 0.6120
      Hyperparameters:
    {'dropout_rate': 0.11348414124439268,
     'fc_size': 184,
     'kernel_size_0': 3,
     'kernel_size_1': 3,
     'n_filters_0': 98,
     'n_filters_1': 119,
     'n_layers': 2}


## Visualizing the Results

Optuna provides several built-in visualization functions to help analyze the results of the hyperparameter optimization process.
These visualizations can provide valuable insights into the optimization process and the impact of different hyperparameters on the model's performance:
- `plot_optimization_history`: This plot shows the optimization history of the objective function, allowing you to see how the performance of the model improved over time. It provides a visual representation of the objective values (in this case, accuracy) across different trials.
 
- `plot_param_importances`: This plot shows the importance of each hyperparameter in the optimization process. It helps identify which hyperparameters had the most significant impact on the model's performance, allowing you to focus on the most influential hyperparameters in future experiments.

- `plot_parallel_coordinate`: This plot visualizes the relationship between different hyperparameters and the objective function. It allows you to see how different hyperparameter configurations affected the model's performance, providing insights into the interactions between hyperparameters and their impact on the objective value.


```python
# Plotting the optimization history
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.title('Optimization History')
plt.show()

# Importance of hyperparameters
optuna.visualization.matplotlib.plot_param_importances(study)
plt.show()

ax = optuna.visualization.matplotlib.plot_parallel_coordinate(
    study, params=['n_layers', 'n_filters_0', 'kernel_size_0', 'dropout_rate', 'fc_size']
)
fig = ax.figure
fig.set_size_inches(12, 6, forward=True)  # forward=True updates the canvas
fig.tight_layout()
```


    
![png](C2_M1_Lab_3_Optuna_files/C2_M1_Lab_3_Optuna_14_0.png)
    



    
![png](C2_M1_Lab_3_Optuna_files/C2_M1_Lab_3_Optuna_14_1.png)
    



    
![png](C2_M1_Lab_3_Optuna_files/C2_M1_Lab_3_Optuna_14_2.png)
    


## (Optional) Comparing Grid Search with Optuna's Default Search

In this section, you will compare the performance of Optuna's default search algorithm with a grid search approach.
Grid search is a traditional method for hyperparameter optimization that exhaustively searches through a predefined set of hyperparameter combinations.
While it can be effective, it is often computationally expensive and may not explore the search space as efficiently as Optuna's adaptive algorithms.

To perform the comparison, a simpler example is used, such as the one from previous labs. 
The `FlexibleSimpleCNN` model is used on a smaller dataset, consisting in a subset of the [Fruit and Vegetable Disease (Healthy vs Rotten)](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten).
This dataset contains images of healthy and rotten fruits and vegetables. 
You will only use a custom subset of the dataset, which contains 1200 images of apples, 1000 being healthy and 200 being rotten. 


```python
class FlexibleSimpleCNN(nn.Module):
    """
    A simple, flexible Convolutional Neural Network.

    This network consists of two convolutional layers, each followed by a
    max-pooling layer, and two fully connected layers. The number of filters
    in the convolutional layers and the size of the hidden linear layer are
    configurable, making the architecture adaptable to different requirements.
    """
    def __init__(self, conv1_out, conv2_out, fc_size, num_classes):
        """
        Initializes the layers of the CNN.

        Args:
            conv1_out: The number of output channels for the first
                       convolutional layer.
            conv2_out: The number of output channels for the second
                       convolutional layer.
            fc_size: The number of neurons in the hidden fully connected layer.
            num_classes: The number of output classes for the final layer.
        """
        super(FlexibleSimpleCNN, self).__init__()
        # Define the first convolutional layer
        self.conv1 = nn.Conv2d(3, conv1_out, kernel_size=3, padding=1)
        # Define the second convolutional layer
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=3, padding=1)
        # Define a max pooling layer to be used after each convolution
        self.pool = nn.MaxPool2d(2, 2)
        
        # Define the first fully connected (hidden) layer
        # Assumes input images are 32x32, resulting in an 8x8 feature map after two pooling layers
        self.fc1 = nn.Linear(conv2_out * 8 * 8, fc_size)
        # Define the final fully connected (output) layer
        self.fc2 = nn.Linear(fc_size, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x: The input tensor of shape (batch_size, channels, height, width).

        Returns:
            The output logits from the network.
        """
        # Apply the first convolutional block: convolution, ReLU activation, and pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply the second convolutional block
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the feature map to prepare for the fully connected layers
        x = x.view(x.size(0), -1)
        # Pass through the first fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        # Pass through the final output layer
        x = self.fc2(x)
        # Return the resulting logits
        return x
```

The goal of the model is to classify the images into two classes: healthy and rotten.
Note that the dataset is imbalanced, with a higher number of healthy images compared to rotten ones.
In such cases, accuracy is not the best metric to evaluate model performance, as it can be misleading due to the class imbalance.
Therefore, the **F1 score will be used as the evaluation metric**, as it is more suitable for imbalanced datasets. 
To ensure a fair comparison between grid search and Optuna's search, the same hyperparameter space will be used for both approaches.

You will also use the `trial.set_user_attr` method, which allows you to store additional information in each trial object, such as accuracy, recall, precision, and F1 score.
This is useful for tracking additional metrics that can be analyzed later.

**Why no dummy input is needed here**: Unlike the `FlexibleCNN` used earlier, the `FlexibleSimpleCNN` used in this section does not require a dummy input to initialize its parameters.

- **Static Initialization**: In `FlexibleSimpleCNN`, the classifier layers (`self.fc1` and `self.fc2`) are defined directly within the `__init__` method based on a fixed input, rather than being created dynamically during the first forward pass.
>
- **Optimizer Safety**: Because these layers exist immediately upon model instantiation, `model.parameters()` correctly includes the classifier weights right from the start. The optimizer will therefore track and train the classifier correctly without needing a "warm-up" pass.


```python
def objective_apples(trial, device):
    """
    Defines the Optuna objective function for a CNN on an apple dataset.

    For each trial, this function samples hyperparameters for a CNN
    architecture, trains the model on a custom apple dataset, and evaluates
    its performance. It logs accuracy, precision, and recall, while
    returning the F1-score as the primary metric for Optuna to optimize.

    Args:
        trial: An Optuna `Trial` object used to sample hyperparameters.
        device: The device ('cpu' or 'cuda') for model training and evaluation.

    Returns:
        The F1-score of the trained model on the validation set.
    """
    # Sample a set of hyperparameters for the model architecture
    conv1_out = trial.suggest_int("conv1_out", 8, 64, step=8)
    conv2_out = trial.suggest_int("conv2_out", 16, 128, step=16)
    fc_size = trial.suggest_int("fc_size", 32, 256, step=32)

    # Define fixed parameters for the data loaders
    img_size = 32
    batch_size = 128

    # Create the training and validation data loaders for the apple dataset
    train_loader, val_loader = helper_utils.get_apples_dataset_dataloaders(
        img_size=img_size,
        batch_size=batch_size
    )

    # Specify the number of output classes for the dataset
    num_classes = 2
    # Create an instance of the model with the sampled hyperparameters
    model = FlexibleSimpleCNN(
        conv1_out=conv1_out,
        conv2_out=conv2_out,
        fc_size=fc_size,
        num_classes=num_classes
    ).to(device)
    
    # Define fixed training components: learning rate, optimizer, and loss function
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fcn = nn.CrossEntropyLoss()

    # Set the fixed number of epochs for the training loop
    n_epochs = 5
    # Train the model using a helper function
    helper_utils.train_model(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_loader,
        n_epochs=n_epochs,
        loss_fcn=loss_fcn,
        device=device
    )

    # Evaluate the trained model on the validation set to get performance metrics
    accuracy, precision, recall, f1 = helper_utils.evaluate_metrics(
        model, val_loader, device, num_classes=2
    )

    # Log additional metrics to the Optuna trial for more detailed analysis
    trial.set_user_attr("accuracy", accuracy)
    trial.set_user_attr("precision", precision)
    trial.set_user_attr("recall", recall)

    # Return the F1-score as the objective value for Optuna to maximize
    return f1
```

With the objective function defined, you can now run both the default Optuna search algorithm and the grid search algorithm. Both methods will be executed for the same number of trials, and the results will be compared using the `plot_optimization_history` function.

First, in this case, you will specify `sampler = optuna.samplers.TPESampler(seed=42)` (the default sampler in Optuna), which is the Tree-structured Parzen Estimator (TPE) sampler. The TPE sampler is a sophisticated algorithm that adapts the search space based on the results of previous trials, allowing for more efficient exploration of the hyperparameter space. Then, you will create a study object with the desired sampler and set `direction='maximize'` to indicate that you want to maximize the objective function (in this case, the F1 score). Note that the trials dataframe now includes additional columns for accuracy, recall, and precision, which were set using the `trial.set_user_attr` method inside the objective function.

Next, run the grid search sampler (`sampler = optuna.samplers.GridSampler(param_grid)`) to explore the hyperparameter space. The grid search will iterate, until reaching the number of trials defined, over all possible combinations of hyperparameters in the grid, evaluating the objective function for each unique configuration without repeating any combination.


```python
seed = 42
helper_utils.set_seed(seed)

sampler = optuna.samplers.TPESampler(seed=seed)  # Use TPE sampler (the default sampler in Optuna)

# Create a study object and optimize the objective function
study_apples = optuna.create_study(direction='maximize', sampler=sampler)

n_trials = 10
study_apples.optimize(lambda trial: objective_apples(trial, device), n_trials=n_trials)  
```

    [I 2026-06-05 09:48:16,591] A new study created in memory with name: no-name-3df275d6-b573-4df0-b51b-e89a5d15d59c


    {'healthy': 0, 'rotten': 1}



    Current Epoch:   0%|          | 0/5 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 0.3548
    Training complete!
    


    [I 2026-06-05 09:49:09,644] Trial 0 finished with value: 0.6255446672439575 and parameters: {'conv1_out': 24, 'conv2_out': 128, 'fc_size': 192}. Best is trial 0 with value: 0.6255446672439575.


    {'healthy': 0, 'rotten': 1}



    Current Epoch:   0%|          | 0/5 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 0.4515
    Training complete!
    


    [I 2026-06-05 09:49:52,123] Trial 1 finished with value: 0.439461886882782 and parameters: {'conv1_out': 40, 'conv2_out': 32, 'fc_size': 64}. Best is trial 0 with value: 0.6255446672439575.


    {'healthy': 0, 'rotten': 1}



    Current Epoch:   0%|          | 0/5 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 0.3359
    Training complete!
    


    [I 2026-06-05 09:50:33,712] Trial 2 finished with value: 0.7176079750061035 and parameters: {'conv1_out': 8, 'conv2_out': 112, 'fc_size': 160}. Best is trial 2 with value: 0.7176079750061035.


    {'healthy': 0, 'rotten': 1}



    Current Epoch:   0%|          | 0/5 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 0.3462
    Training complete!
    


    [I 2026-06-05 09:51:14,552] Trial 3 finished with value: 0.6919227838516235 and parameters: {'conv1_out': 48, 'conv2_out': 16, 'fc_size': 256}. Best is trial 2 with value: 0.7176079750061035.


    {'healthy': 0, 'rotten': 1}



    Current Epoch:   0%|          | 0/5 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 0.3534
    Training complete!
    


    [I 2026-06-05 09:51:56,630] Trial 4 finished with value: 0.8412935733795166 and parameters: {'conv1_out': 56, 'conv2_out': 32, 'fc_size': 64}. Best is trial 4 with value: 0.8412935733795166.


    {'healthy': 0, 'rotten': 1}



    Current Epoch:   0%|          | 0/5 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 0.3716
    Training complete!
    


    [I 2026-06-05 09:52:38,803] Trial 5 finished with value: 0.5040310621261597 and parameters: {'conv1_out': 16, 'conv2_out': 48, 'fc_size': 160}. Best is trial 4 with value: 0.8412935733795166.


    {'healthy': 0, 'rotten': 1}



    Current Epoch:   0%|          | 0/5 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 0.2958
    Training complete!
    


    [I 2026-06-05 09:53:22,475] Trial 6 finished with value: 0.7342193126678467 and parameters: {'conv1_out': 32, 'conv2_out': 48, 'fc_size': 160}. Best is trial 4 with value: 0.8412935733795166.


    {'healthy': 0, 'rotten': 1}



    Current Epoch:   0%|          | 0/5 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 0.2837
    Training complete!
    


    [I 2026-06-05 09:54:04,182] Trial 7 finished with value: 0.858942985534668 and parameters: {'conv1_out': 16, 'conv2_out': 48, 'fc_size': 96}. Best is trial 7 with value: 0.858942985534668.


    {'healthy': 0, 'rotten': 1}



    Current Epoch:   0%|          | 0/5 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 0.3144
    Training complete!
    


    [I 2026-06-05 09:54:47,591] Trial 8 finished with value: 0.9125000238418579 and parameters: {'conv1_out': 32, 'conv2_out': 112, 'fc_size': 64}. Best is trial 8 with value: 0.9125000238418579.


    {'healthy': 0, 'rotten': 1}



    Current Epoch:   0%|          | 0/5 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 0.3693
    Training complete!
    


    [I 2026-06-05 09:55:29,973] Trial 9 finished with value: 0.6905393600463867 and parameters: {'conv1_out': 40, 'conv2_out': 80, 'fc_size': 32}. Best is trial 8 with value: 0.9125000238418579.



```python
df_apples_study = study_apples.trials_dataframe()

df_apples_study
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>number</th>
      <th>value</th>
      <th>datetime_start</th>
      <th>datetime_complete</th>
      <th>duration</th>
      <th>params_conv1_out</th>
      <th>params_conv2_out</th>
      <th>params_fc_size</th>
      <th>user_attrs_accuracy</th>
      <th>user_attrs_precision</th>
      <th>user_attrs_recall</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.625545</td>
      <td>2026-06-05 09:48:16.592329</td>
      <td>2026-06-05 09:49:09.644304</td>
      <td>0 days 00:00:53.051975</td>
      <td>24</td>
      <td>128</td>
      <td>192</td>
      <td>0.824</td>
      <td>0.839588</td>
      <td>0.606009</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.439462</td>
      <td>2026-06-05 09:49:09.645596</td>
      <td>2026-06-05 09:49:52.123235</td>
      <td>0 days 00:00:42.477639</td>
      <td>40</td>
      <td>32</td>
      <td>64</td>
      <td>0.784</td>
      <td>0.392000</td>
      <td>0.500000</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.717608</td>
      <td>2026-06-05 09:49:52.123972</td>
      <td>2026-06-05 09:50:33.712616</td>
      <td>0 days 00:00:41.588644</td>
      <td>8</td>
      <td>112</td>
      <td>160</td>
      <td>0.864</td>
      <td>0.860886</td>
      <td>0.676211</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.691923</td>
      <td>2026-06-05 09:50:33.713682</td>
      <td>2026-06-05 09:51:14.552730</td>
      <td>0 days 00:00:40.839048</td>
      <td>48</td>
      <td>16</td>
      <td>256</td>
      <td>0.868</td>
      <td>0.807815</td>
      <td>0.654647</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.841294</td>
      <td>2026-06-05 09:51:14.553465</td>
      <td>2026-06-05 09:51:56.630243</td>
      <td>0 days 00:00:42.076778</td>
      <td>56</td>
      <td>32</td>
      <td>64</td>
      <td>0.932</td>
      <td>0.925148</td>
      <td>0.792602</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.504031</td>
      <td>2026-06-05 09:51:56.630995</td>
      <td>2026-06-05 09:52:38.803303</td>
      <td>0 days 00:00:42.172308</td>
      <td>16</td>
      <td>48</td>
      <td>160</td>
      <td>0.812</td>
      <td>0.904858</td>
      <td>0.530000</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.734219</td>
      <td>2026-06-05 09:52:38.804228</td>
      <td>2026-06-05 09:53:22.475740</td>
      <td>0 days 00:00:43.671512</td>
      <td>32</td>
      <td>48</td>
      <td>160</td>
      <td>0.872</td>
      <td>0.930736</td>
      <td>0.686275</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.858943</td>
      <td>2026-06-05 09:53:22.476489</td>
      <td>2026-06-05 09:54:04.182842</td>
      <td>0 days 00:00:41.706353</td>
      <td>16</td>
      <td>48</td>
      <td>96</td>
      <td>0.916</td>
      <td>0.899405</td>
      <td>0.830574</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0.912500</td>
      <td>2026-06-05 09:54:04.183598</td>
      <td>2026-06-05 09:54:47.590963</td>
      <td>0 days 00:00:43.407365</td>
      <td>32</td>
      <td>112</td>
      <td>64</td>
      <td>0.944</td>
      <td>0.912500</td>
      <td>0.912500</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0.690539</td>
      <td>2026-06-05 09:54:47.591716</td>
      <td>2026-06-05 09:55:29.973004</td>
      <td>0 days 00:00:42.381288</td>
      <td>40</td>
      <td>80</td>
      <td>32</td>
      <td>0.860</td>
      <td>0.847701</td>
      <td>0.652185</td>
      <td>COMPLETE</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Run with Grid Search Sampler

# Define the hyperparameter grid
param_grid = {
    "conv1_out": list(range(8, 65, 8)),       # [8, 16, 24, 32, 40, 48, 56, 64]
    "conv2_out": list(range(16, 129, 16)),    # [16, 32, 48, 64, 80, 96, 112, 128]
    "fc_size":   list(range(32, 257, 32))     # [32, 64, 96, 128, 160, 192, 224, 256]
}

# Create a GridSampler with the defined grid
grid_sampler = optuna.samplers.GridSampler(param_grid, seed=seed)  # Use seed for reproducibility

# Create a study object with the GridSampler
study_grid = optuna.create_study(direction='maximize', sampler=grid_sampler)

study_grid.optimize(lambda trial: objective_apples(trial, device), n_trials=n_trials)
```

    [I 2026-06-05 09:55:29,989] A new study created in memory with name: no-name-56bc83cd-48e2-497f-a950-7d708e74a39a


    {'healthy': 0, 'rotten': 1}



    Current Epoch:   0%|          | 0/5 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 0.3343
    Training complete!
    


    [I 2026-06-05 09:56:11,534] Trial 0 finished with value: 0.4343891441822052 and parameters: {'conv1_out': 40, 'conv2_out': 112, 'fc_size': 32}. Best is trial 0 with value: 0.4343891441822052.


    {'healthy': 0, 'rotten': 1}



    Current Epoch:   0%|          | 0/5 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 0.3459
    Training complete!
    


    [I 2026-06-05 09:56:51,964] Trial 1 finished with value: 0.44933921098709106 and parameters: {'conv1_out': 64, 'conv2_out': 96, 'fc_size': 256}. Best is trial 1 with value: 0.44933921098709106.


    {'healthy': 0, 'rotten': 1}



    Current Epoch:   0%|          | 0/5 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 0.2648
    Training complete!
    


    [I 2026-06-05 09:57:33,726] Trial 2 finished with value: 0.7944719791412354 and parameters: {'conv1_out': 56, 'conv2_out': 112, 'fc_size': 256}. Best is trial 2 with value: 0.7944719791412354.


    {'healthy': 0, 'rotten': 1}



    Current Epoch:   0%|          | 0/5 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 0.2756
    Training complete!
    


    [I 2026-06-05 09:58:17,819] Trial 3 finished with value: 0.8275692462921143 and parameters: {'conv1_out': 24, 'conv2_out': 64, 'fc_size': 64}. Best is trial 3 with value: 0.8275692462921143.


    {'healthy': 0, 'rotten': 1}



    Current Epoch:   0%|          | 0/5 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 0.3037
    Training complete!
    


    [I 2026-06-05 09:59:01,921] Trial 4 finished with value: 0.8207193613052368 and parameters: {'conv1_out': 64, 'conv2_out': 112, 'fc_size': 64}. Best is trial 3 with value: 0.8275692462921143.


    {'healthy': 0, 'rotten': 1}



    Current Epoch:   0%|          | 0/5 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 0.4427
    Training complete!
    


    [I 2026-06-05 09:59:45,849] Trial 5 finished with value: 0.45652174949645996 and parameters: {'conv1_out': 24, 'conv2_out': 16, 'fc_size': 128}. Best is trial 3 with value: 0.8275692462921143.


    {'healthy': 0, 'rotten': 1}



    Current Epoch:   0%|          | 0/5 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 0.3234
    Training complete!
    


    [I 2026-06-05 10:00:29,992] Trial 6 finished with value: 0.7238179445266724 and parameters: {'conv1_out': 32, 'conv2_out': 32, 'fc_size': 160}. Best is trial 3 with value: 0.8275692462921143.


    {'healthy': 0, 'rotten': 1}



    Current Epoch:   0%|          | 0/5 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 0.2883
    Training complete!
    


    [I 2026-06-05 10:01:12,925] Trial 7 finished with value: 0.7032498121261597 and parameters: {'conv1_out': 64, 'conv2_out': 128, 'fc_size': 96}. Best is trial 3 with value: 0.8275692462921143.


    {'healthy': 0, 'rotten': 1}



    Current Epoch:   0%|          | 0/5 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 0.3741
    Training complete!
    


    [I 2026-06-05 10:01:57,192] Trial 8 finished with value: 0.5978121161460876 and parameters: {'conv1_out': 48, 'conv2_out': 16, 'fc_size': 192}. Best is trial 3 with value: 0.8275692462921143.


    {'healthy': 0, 'rotten': 1}



    Current Epoch:   0%|          | 0/5 [00:00<?, ?it/s]



    Current Batch:   0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 5 - Train Loss: 0.2793
    Training complete!
    


    [I 2026-06-05 10:02:39,372] Trial 9 finished with value: 0.7767857313156128 and parameters: {'conv1_out': 32, 'conv2_out': 112, 'fc_size': 256}. Best is trial 3 with value: 0.8275692462921143.



```python
# Plotting the optimization history
optuna.visualization.matplotlib.plot_optimization_history(study_apples)
plt.title('Optimization History')
plt.show()

# Plotting the optimization history
optuna.visualization.matplotlib.plot_optimization_history(study_grid)
plt.title('Optimization History')
plt.show()
```


    
![png](C2_M1_Lab_3_Optuna_files/C2_M1_Lab_3_Optuna_23_0.png)
    



    
![png](C2_M1_Lab_3_Optuna_files/C2_M1_Lab_3_Optuna_23_1.png)
    


Under the grid search approach, the objective values vary significantly because all possible combinations of hyperparameters in the grid are exhaustively evaluated.  
Since grid search does not take into account the performance of previous trials, it may waste resources evaluating many suboptimal configurations, especially in high-dimensional or redundant hyperparameter spaces.

In contrast, Optuna's search algorithm employs a more intelligent, adaptive sampling strategy (such as TPE).
It uses the results of previous trials to guide the search toward more promising regions of the hyperparameter space, allowing for a more efficient and focused exploration.
As a result, Optuna is often able to identify better-performing hyperparameter configurations with fewer trials compared to grid search.

The larger and more complex the search space becomes, the more pronounced the advantage of adaptive sampling methods like Optuna over exhaustive methods like grid search.

## Conclusion

Congratulations on completing the hyperparameter optimization tutorial with Optuna!

In this notebook, you explored the key components of an Optuna-based optimization workflow: defining the objective function, setting up the study, and analyzing the results. You also built a flexible CNN architecture designed to accommodate varying hyperparameter configurations, allowing for more dynamic experimentation.

Additionally, you compared Optuna’s default search algorithm (TPE) with a traditional grid search, demonstrating the efficiency and adaptability of Optuna in navigating complex hyperparameter spaces. Throughout the tutorial, you evaluated model performance using multiple metrics and learned how to interpret results through Optuna’s powerful visualization tools.

By the end of this tutorial, you’ve gained a practical and conceptual understanding of how to integrate Optuna into deep learning projects, equipping you with a valuable skill for optimizing model performance efficiently and effectively.
