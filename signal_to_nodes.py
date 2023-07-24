import torch
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def signal_to_nodes_WISDM(signal, window_size, overlap=0.5):
    step = round(window_size * (1 - overlap))  # calculate step size for 50% overlap
    signal.sort_values(by=['user', 'timestamp'], inplace=True)  # sort the signal by 'user' and 'timestamp'

    # Map activity labels to integer values
    signal['activity'], labels_index = pd.factorize(signal['activity'])

    # create windows for each activity type
    nodes = []
    nodes_labels = []
    for activity in signal['activity'].unique():
        activity_data = signal[signal['activity'] == activity]
        activity_nodes = [activity_data[i:i + window_size] for i in
                          range(0, len(activity_data) - window_size + 1, step)]

        if len(activity_nodes[-1]) < window_size:
            # Calculate how many rows we need to add
            rows_to_add = window_size - len(activity_nodes[-1])
            # Get the last row of the last node
            last_row = pd.DataFrame(activity_nodes[-1].iloc[-1]).transpose()
            # Append last row to last node until it's of window_size length
            for _ in range(rows_to_add):
                activity_nodes[-1] = activity_nodes[-1].append(last_row, ignore_index=True)

        activity_nodes_labels = [node['activity'].iloc[0] for node in activity_nodes]
        nodes.extend(activity_nodes)
        nodes_labels.extend(activity_nodes_labels)

    # convert to torch tensors
    nodes = torch.stack(
        [torch.tensor(node.drop(['user', 'activity', 'timestamp'], axis=1).values) for node in nodes]).float().to(
        device)
    nodes_labels = torch.tensor(nodes_labels, dtype=torch.long, device=device)

    return nodes, nodes_labels, labels_index


def signal_to_nodes_PAMAP2(signal, window_size, overlap=0.5):
    step = round(window_size * (1 - overlap))  # calculate step size for 50% overlap
    signal.sort_values(by=['id', 'timestamp'], inplace=True)

    # create windows for each activity type
    nodes = []
    nodes_labels = []
    for activity in signal['activity_id'].unique():
        activity_data = signal[signal['activity_id'] == activity]
        activity_nodes = [activity_data[i:i + window_size] for i in
                          range(0, len(activity_data) - window_size + 1, step)]

        if len(activity_nodes[-1]) < window_size:
            # Calculate how many rows we need to add
            rows_to_add = window_size - len(activity_nodes[-1])
            # Get the last row of the last node
            last_row = pd.DataFrame(activity_nodes[-1].iloc[-1]).transpose()
            # Append last row to last node until it's of window_size length
            for _ in range(rows_to_add):
                activity_nodes[-1] = activity_nodes[-1].append(last_row, ignore_index=True)

        activity_nodes_labels = [node['activity_id'].iloc[0] for node in activity_nodes]
        nodes.extend(activity_nodes)
        nodes_labels.extend(activity_nodes_labels)

    # convert to torch tensors
    nodes = torch.stack(
        [torch.tensor(node.drop(['id', 'activity_id', 'timestamp'], axis=1).values) for node in nodes]).float().to(
        device)
    nodes_labels = torch.tensor(nodes_labels, dtype=torch.long, device=device)

    return nodes, nodes_labels

def signal_to_nodes_HARTH(signal, window_size, overlap=0.5):
    step = round(window_size * (1 - overlap))  # calculate step size for 50% overlap
    signal.sort_values(by=['timestamp'], inplace=True)

    # create windows for each activity type
    nodes = []
    nodes_labels = []
    for activity in signal['label'].unique():
        activity_data = signal[signal['label'] == activity]
        activity_nodes = [activity_data[i:i + window_size] for i in
                          range(0, len(activity_data) - window_size + 1, step)]

        if len(activity_nodes[-1]) < window_size:
            # Calculate how many rows we need to add
            rows_to_add = window_size - len(activity_nodes[-1])
            # Get the last row of the last node
            last_row = pd.DataFrame(activity_nodes[-1].iloc[-1]).transpose()
            # Append last row to last node until it's of window_size length
            for _ in range(rows_to_add):
                activity_nodes[-1] = activity_nodes[-1].append(last_row, ignore_index=True)

        activity_nodes_labels = [node['label'].iloc[0] for node in activity_nodes]
        nodes.extend(activity_nodes)
        nodes_labels.extend(activity_nodes_labels)

    # convert to torch tensors
    nodes = torch.stack(
        [torch.tensor(node.drop(['label', 'timestamp'], axis=1).values) for node in nodes]).float().to(
        device)
    nodes_labels = torch.tensor(nodes_labels, dtype=torch.long, device=device)

    return nodes, nodes_labels
