import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from torchvision.transforms import functional as F


letter_ref = [
    "Dear Laurence",
    "Hope the PyTorch course is going well",
    "Do notforget to keep the labs interesting and engaging",
    "Maybe the students could decode my messy handwriting",
    "That might be a bit too challenging though",
    "I am impressed you are able to read this",
]


path_data = "./EMNIST_data"


def load_hidden_message_images(file_name="hidden_message_images.pkl"):
    """
    Loads hidden message images from a pickle file.

    Args:
        file_name (str): The name of the file to load the images from.

    Returns:
        message_imgs (list): A list containing the loaded message images.
    """
    # Open the specified file in read-binary mode
    with open(file_name, "rb") as f:
        # Import the pickle module
        import pickle

        # Load the data from the file
        message_imgs = pickle.load(f)
        
    # Return the loaded images
    return message_imgs


def decode_word_imgs(word_imgs, model, device):
    """
    Decodes a sequence of character images into a single word string using a 
    provided classification model.

    Args:
        word_imgs (list): A collection of image tensors representing 
            individual characters in a word.
        model (torch.nn.Module): The trained neural network model used to 
            predict the character from the image.
        device (torch.device): The computation device to which the tensors 
            should be moved before inference.

    Returns:
        decoded_word (str): The concatenated string of predicted characters 
            forming the complete word.
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Initialize an empty list to store the predicted characters
    decoded_chars = []
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        # Iterate through each character image in the provided sequence
        for char_img in word_imgs:
            # Add a batch dimension to the tensor and move to target device
            char_img = char_img.unsqueeze(0).to(
                device
            ) 
            
            # Forward pass to predict the character probabilities
            output = model(char_img)
            
            # Extract the predicted class index with the highest probability
            _, predicted = output.max(1)
            
            # Retrieve the numerical value of the prediction
            predicted_label = predicted.item()
            
            # Convert the predicted label to corresponding lowercase letter
            lowercase_char = chr(ord("a") + predicted_label)
            
            # Append the predicted character to the list
            decoded_chars.append(f"{lowercase_char}")
            
    # Join the list of individual characters to form the final decoded word
    decoded_word = "".join(decoded_chars)
    
    # Return the final string
    return decoded_word


def visualize_image(img, label=None, ax=None):
    """
    Visualizes an EMNIST image with its label. If an axis is provided, 
    plots on that axis; otherwise, creates a new figure.

    Args:
        img (np.ndarray): The image array to display.
        label (int): The numeric EMNIST label. If None, no title is shown.
        ax (matplotlib.axes.Axes): Axis to plot on. If None, creates a 
            new figure.
    """
    # Check if the image is a PyTorch tensor and convert to numpy array
    if isinstance(img, torch.Tensor):
        # Squeeze the tensor to remove single-dimensional entries
        img = img.numpy().squeeze()
    # Check if the image is a numpy array
    elif isinstance(img, np.ndarray):
        # Check if the array has three dimensions
        if img.ndim == 3:
            # Extract the first channel of the image
            img = img[:, :, 0]

    # Check if a label is provided for the title
    if label is not None:
        # Convert the label to uppercase and lowercase characters
        uppercase_char, lowercase_char = convert_emnist_label_to_char(label)
        
        # Create a formatted title string
        title = f"EMNIST Letter: {uppercase_char}/{lowercase_char}"
    else:
        # Set the title to None if no label is provided
        title = None

    # Check if an axis is provided for plotting
    if ax is None:
        # Create a new figure and axis with specific dimensions
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # Set a flag to display the colorbar
        show_colorbar = True
    else:
        # Disable the colorbar display
        show_colorbar = False

    # Display the image on the axis using a grayscale colormap
    im = ax.imshow(img, cmap="gray")
    
    # Set the x-axis tick marks
    ax.set_xticks(np.arange(0, 28, 1))
    
    # Set the y-axis tick marks
    ax.set_yticks(np.arange(0, 28, 1))
    
    # Set the size of the tick labels
    ax.tick_params(labelsize=6)
    
    # Enable the grid on the axis with specific color and transparency
    ax.grid(True, color="red", alpha=0.3)
    
    # Check if a title string was created
    if title:
        # Set the title of the axis
        ax.set_title(title)

    # Check if the colorbar flag is enabled
    if show_colorbar:
        # Add a colorbar to the plot
        plt.colorbar(im, ax=ax)
        
        # Display the plot
        plt.show()


def display_data_loader_contents(data_loader):
    """
    Displays the contents of the data loader including sizes and shapes.

    Args:
        data_loader (torch.utils.data.DataLoader): The data loader to 
            display.
    """
    # Begin exception handling block
    try:
        # Print the total number of images in the dataset
        print("Total number of images in dataset:", len(data_loader.dataset))
        
        # Print the total number of batches in the data loader
        print("Total number of batches:", len(data_loader))
        
        # Iterate through the batches in the data loader
        for batch_idx, (data, labels) in enumerate(data_loader):
            # Print the current batch number
            print(f"--- Batch {batch_idx + 1} ---")
            
            # Print the shape of the data tensor
            print(f"Data shape: {data.shape}")
            
            # Print the shape of the labels tensor
            print(f"Labels shape: {labels.shape}")
            
            # Exit the loop after displaying the first batch
            break
            
    # Catch iteration exceptions if the data loader is empty
    except StopIteration:
        # Print a message indicating the data loader is empty
        print("data loader is empty.")
        
    # Catch any other general exceptions
    except Exception as e:
        # Print the specific error message
        print(f"An error occurred: {e}")


def evaluate_per_class(model, test_loader, device):
    """
    Evaluates the model's accuracy for each individual class.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        test_loader (torch.utils.data.DataLoader): DataLoader for the 
            test dataset.
        device (torch.device): Device to run the model on.

    Returns:
        class_accuracies (dict): A dictionary containing accuracy for each 
            class letter.
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Initialize an empty list to store all target labels
    all_targets = []
    
    # Initialize an empty list to store all predicted labels
    all_predictions = []

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        # Iterate through inputs and targets in the test loader
        for inputs, targets in test_loader:
            # Move inputs and targets to the specified computation device
            inputs, targets = inputs.to(device), targets.to(device)

            # Shift target labels down by a single value
            targets = targets - 1

            # Generate model outputs from the inputs
            outputs = model(inputs)
            
            # Extract the highest probability predictions
            _, predicted = outputs.max(1)

            # Append the target labels to the storage list
            all_targets.extend(targets.cpu().numpy())
            
            # Append the predicted labels to the storage list
            all_predictions.extend(predicted.cpu().numpy())

    # Initialize an empty dictionary for class accuracies
    class_accuracies = {}

    # Iterate through the possible class indices
    for class_idx in range(26):
        # Extract targets matching the current class index
        class_targets = [
            t for t, p in zip(all_targets, all_predictions) if t == class_idx
        ]
        
        # Extract predictions matching the current class index
        class_predictions = [
            p for t, p in zip(all_targets, all_predictions) if t == class_idx
        ]

        # Check if there are any targets for the current class
        if len(class_targets) > 0:
            # Calculate and store the accuracy score for the class
            class_accuracies[chr(65 + class_idx)] = accuracy_score(
                class_targets, class_predictions
            )
        else:
            # Set a default accuracy of zero if the class is empty
            class_accuracies[chr(65 + class_idx)] = 0.0

    # Return the final dictionary of accuracies
    return class_accuracies


def save_student_model(model, filename="trained_student_model.pth"):
    """
    Saves the student's trained model and metadata to a file.

    Args:
        model (torch.nn.Module): The student's trained model.
        filename (str): The filename to save to.
    """
    # Create a dictionary containing the model state
    save_dict = {"model": model}
    
    # Save the dictionary to the specified file path
    torch.save(save_dict, filename)
    
    # Print a confirmation message indicating the save location
    print(f"Model saved to {filename}")


def convert_emnist_label_to_char(label):
    """
    Converts an EMNIST label to corresponding uppercase and lowercase letters.

    Args:
        label (int): The numeric EMNIST label.

    Returns:
        char_tuple (tuple): A tuple containing the uppercase and lowercase 
            characters.
    """
    # Check if the label is within the valid range
    if not (1 <= label <= 26):
        # Raise an error if the label is invalid
        raise ValueError("Label must be between 1 and 26 inclusive.")

    # Calculate the uppercase character based on the label
    uppercase_char = chr(64 + label)
    
    # Calculate the lowercase character based on the label
    lowercase_char = chr(96 + label)

    # Group characters into a tuple variable
    char_tuple = (uppercase_char, lowercase_char)

    # Return the character tuple
    return char_tuple