from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error


def load_and_partition_data(
    data_path: Path, seq_length: int = 100
) -> tuple[np.ndarray, int]:
    """Loads the given data and paritions it into sequences of equal length.

    Args:
        data_path: path to the dataset
        sequence_length: length of the generated sequences

    Returns:
        tuple[np.ndarray, int]: tuple of generated sequences and number of
            features in dataset
    """
    data = np.load(data_path)
    num_features = len(data.keys())

    # Check that each feature provides the same number of data points
    data_lens = [len(data[key]) for key in data.keys()]
    assert len(set(data_lens)) == 1

    num_sequences = data_lens[0] // seq_length
    sequences = np.empty((num_sequences, seq_length, num_features))

    for i in range(0, num_sequences):
        # [sequence_length, num_features]
        sample = np.asarray(
            [data[key][i * seq_length : (i + 1) * seq_length] for key in data.keys()]
        ).swapaxes(0, 1)
        sequences[i] = sample

    return sequences, num_features


def make_datasets(sequences: np.ndarray) -> tuple[TensorDataset, TensorDataset]:
    """Create train and test dataset.

    Args:
        sequences: sequences to use [num_sequences, sequence_length, num_features]

    Returns:
        tuple[TensorDataset, TensorDataset]: train and test dataset
    """
    # Split sequences into train and test split
    train, test = train_test_split(sequences, test_size=0.2)
    return TensorDataset(torch.Tensor(train)), TensorDataset(torch.Tensor(test))


def visualize(
    src: torch.Tensor,
    tgt: torch.Tensor,
    pred: torch.Tensor,
    pred_infer: torch.Tensor,
    idx=0,
) -> None:
    """Visualizes a given sample including predictions.

    Args:
        src: source sequence [bs, src_seq_len, num_features]
        tgt: target sequence [bs, tgt_seq_len, num_features]
        pred: prediction of the model [bs, tgt_seq_len, num_features]
        pred_infer: prediction obtained by running inference
            [bs, tgt_seq_len, num_features]
        idx: batch index to visualize
    """
    x = np.arange(src.shape[1] + tgt.shape[1])
    src_len = src.shape[1]

    plt.plot(x[:src_len], src[idx].cpu().detach(), "bo-", label="src")
    plt.plot(x[src_len:], tgt[idx].cpu().detach(), "go-", label="tgt")
    plt.plot(x[src_len:], pred[idx].cpu().detach(), "ro-", label="pred")
    plt.plot(x[src_len:], pred_infer[idx].cpu().detach(), "yo-", label="pred_infer")

    plt.legend()
    plt.show()
    plt.clf()


def split_sequence(
    sequence: np.ndarray, ratio: float = 0.8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Splits a sequence into 2 (3) parts, as is required by our transformer
    model.

    Assume our sequence length is L, we then split this into src of length N
    and tgt_y of length M, with N + M = L.
    src, the first part of the input sequence, is the input to the encoder, and we
    expect the decoder to predict tgt_y, the second part of the input sequence.
    In addition we generate tgt, which is tgt_y but "shifted left" by one - i.e. it
    starts with the last token of src, and ends with the second-last token in tgt_y.
    This sequence will be the input to the decoder.


    Args:
        sequence: batched input sequences to split [bs, seq_len, num_features]
        ratio: split ratio, N = ratio * L

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: src, tgt, tgt_y
    """
    src_end = int(sequence.shape[1] * ratio)
    # [bs, src_seq_len, num_features]
    src = sequence[:, :src_end]
    # [bs, tgt_seq_len, num_features]
    tgt = sequence[:, src_end - 1 : -1]
    # [bs, tgt_seq_len, num_features]
    tgt_y = sequence[:, src_end:]

    return src, tgt, tgt_y


def move_to_device(device: torch.Tensor, *tensors: torch.Tensor) -> list[torch.Tensor]:
    """Move all given tensors to the given device.

    Args:
        device: device to move tensors to
        tensors: tensors to move

    Returns:
        list[torch.Tensor]: moved tensors
    """
    moved_tensors = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            moved_tensors.append(tensor.to(device))
        else:
            moved_tensors.append(tensor)
    return moved_tensors


def visualize_multi_posx_posy(src, tgt, pred, pred_infer, idx=0):
    """
    Visualizes the source, target, model predictions, and inference predictions
    for the first two features (posx and posy).

    Args:
        src (torch.Tensor): Source sequence [batch_size, src_seq_len, num_features].
        tgt (torch.Tensor): Target sequence [batch_size, tgt_seq_len, num_features].
        pred (torch.Tensor): Prediction of the model [batch_size, tgt_seq_len, num_features].
        pred_infer (torch.Tensor): Prediction obtained by running inference [batch_size, tgt_seq_len, num_features].
        idx (int): Index of the batch to visualize.
    """
    # Extract posx and posy for visualization
    src_posx = src[idx, :, 0].cpu().detach().numpy()
    src_posy = src[idx, :, 1].cpu().detach().numpy()
    tgt_posx = tgt[idx, :, 0].cpu().detach().numpy()
    tgt_posy = tgt[idx, :, 1].cpu().detach().numpy()
    pred_posx = pred[idx, :, 0].cpu().detach().numpy()
    pred_posy = pred[idx, :, 1].cpu().detach().numpy()
    pred_infer_posx = pred_infer[idx, :, 0].cpu().detach().numpy()
    pred_infer_posy = pred_infer[idx, :, 1].cpu().detach().numpy()

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(src_posx, 'bo-', label='Source posx')
    axs[0].plot(np.arange(len(src_posx), len(src_posx) + len(tgt_posx)), tgt_posx, 'go-', label='Target posx')
    axs[0].plot(np.arange(len(src_posx), len(src_posx) + len(pred_posx)), pred_posx, 'ro-', label='Prediction posx')
    axs[0].plot(np.arange(len(src_posx), len(src_posx) + len(pred_infer_posx)), pred_infer_posx, 'yo-', label='Inference Prediction posx')
    axs[0].set_ylabel('posx')
    axs[0].legend()
    
    axs[1].plot(src_posy, 'bo-', label='Source posy')
    axs[1].plot(np.arange(len(src_posy), len(src_posy) + len(tgt_posy)), tgt_posy, 'go-', label='Target posy')
    axs[1].plot(np.arange(len(src_posy), len(src_posy) + len(pred_posy)), pred_posy, 'ro-', label='Prediction posy')
    axs[1].plot(np.arange(len(src_posy), len(src_posy) + len(pred_infer_posy)), pred_infer_posy, 'yo-', label='Inference Prediction posy')
    axs[1].set_ylabel('posy')
    axs[1].set_xlabel('Timestep')
    axs[1].legend()

    plt.tight_layout()
    plt.show()



def visualize_multi(src, tgt, pred, pred_infer, idx=0):
    """
    Visualizes the source, target, model predictions, and inference predictions
    for the first two features (posx and posy) on a 2D plane.

    Args:
        src (torch.Tensor): Source sequence [batch_size, src_seq_len, num_features].
        tgt (torch.Tensor): Target sequence [batch_size, tgt_seq_len, num_features].
        pred (torch.Tensor): Prediction of the model [batch_size, tgt_seq_len, num_features].
        pred_infer (torch.Tensor): Prediction obtained by running inference [batch_size, tgt_seq_len, num_features].
        idx (int): Index of the batch to visualize.
    """
    # Extract posx and posy for visualization
    src_posx = src[idx, :, 0].cpu().detach().numpy()
    src_posy = src[idx, :, 1].cpu().detach().numpy()
    tgt_posx = tgt[idx, :, 0].cpu().detach().numpy()
    tgt_posy = tgt[idx, :, 1].cpu().detach().numpy()
    pred_posx = pred[idx, :, 0].cpu().detach().numpy()
    pred_posy = pred[idx, :, 1].cpu().detach().numpy()
    pred_infer_posx = pred_infer[idx, :, 0].cpu().detach().numpy()
    pred_infer_posy = pred_infer[idx, :, 1].cpu().detach().numpy()

    # Plotting on a 2D plane
    plt.figure(figsize=(8, 6))
    plt.plot(src_posx, src_posy, 'bo-', label='Source')
    plt.plot(tgt_posx, tgt_posy, 'go-', label='Target')
    plt.plot(pred_posx, pred_posy, 'ro-', label='Prediction')
    plt.plot(pred_infer_posx, pred_infer_posy, 'yo-', label='Inference Prediction')
    plt.xlabel('posx')
    plt.ylabel('posy')
    plt.legend()
    plt.title('2D Position Trajectory')
    plt.axis('equal')  # Ensure equal aspect ratio for x and y axes to accurately represent distances
    plt.grid(True)
    plt.show()

def visualize_best_prediction(src, tgt, pred, idx=0):
    """
    Visualizes the source, target, model predictions, and inference predictions
    for the first two features (posx and posy) on a 2D plane.

    Args:
        src (torch.Tensor): Source sequence [batch_size, src_seq_len, num_features].
        tgt (torch.Tensor): Target sequence [batch_size, tgt_seq_len, num_features].
        pred (torch.Tensor): Prediction of the model [batch_size, tgt_seq_len, num_features].
        pred_infer (torch.Tensor): Prediction obtained by running inference [batch_size, tgt_seq_len, num_features].
        idx (int): Index of the batch to visualize.
    """
    # Extract posx and posy for visualization
    src_posx = src[idx, :, 0].cpu().detach().numpy()
    src_posy = src[idx, :, 1].cpu().detach().numpy()
    tgt_posx = tgt[idx, :, 0].cpu().detach().numpy()
    tgt_posy = tgt[idx, :, 1].cpu().detach().numpy()
    pred_posx = pred[idx, :, 0].cpu().detach().numpy()
    pred_posy = pred[idx, :, 1].cpu().detach().numpy()

    # Plotting on a 2D plane
    plt.figure(figsize=(8, 6))
    plt.plot(src_posx, src_posy, 'bo-', label='Source')
    plt.plot(tgt_posx, tgt_posy, 'go-', label='Target')
    plt.plot(pred_posx, pred_posy, 'ro-', label='Prediction')
    plt.xlabel('posx')
    plt.ylabel('posy')
    plt.legend()
    plt.title('2D Position Trajectory')
    plt.axis('equal')  # Ensure equal aspect ratio for x and y axes to accurately represent distances
    plt.grid(True)
    plt.show()

def mape(y_true, y_pred, epsilon=1e-8):
    """Mean Absolute Percentage Error with handling zeros in the actual values"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def calculate_metrics(y_true, y_pred):
    # Focus on posx and posy, the first two features
    y_true_flat = y_true[..., :2].reshape(-1, 2)  # Reshape to [batch_size * sequence_length, 2]
    y_pred_flat = y_pred[..., :2].reshape(-1, 2)  # Same reshaping for predictions
    
    rmse_val = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    mae_val = mean_absolute_error(y_true_flat, y_pred_flat)
    mape_val = mape(y_true_flat[:, 0], y_pred_flat[:, 0])  # Example for posx
    
    print("RMSE:", rmse_val)
    print("MAE:", mae_val)
    print("MAPE (posx):", mape_val)