import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load datasets
cluster_0 = pd.read_csv('cluster_0.csv')
cluster_1 = pd.read_csv('cluster_1.csv')
cluster_2 = pd.read_csv('cluster_2.csv')


# Function to preprocess data
def preprocess_data(df):
    label_encoders = {}
    for column in ['Rank', 'Supervisor ID', 'Coordinator ID']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    onehot_enc = OneHotEncoder()
    categorical_data = df[['Rank', 'Supervisor ID', 'Coordinator ID']]
    encoded_data = onehot_enc.fit_transform(categorical_data).toarray()

    return df, encoded_data


# Preprocess each cluster
df_0, encoded_data_0 = preprocess_data(cluster_0)
df_1, encoded_data_1 = preprocess_data(cluster_1)
df_2, encoded_data_2 = preprocess_data(cluster_2)


# Custom Dataset class for PyTorch
class CustomDataset(Dataset):
    def __init__(self, df, encoded_data):
        self.users = df['Staff Number'].values
        self.items = df['Coordinator ID'].values
        self.ratings = df['Completion_Rate'].values
        self.encoded_data = encoded_data

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        item = self.items[idx]
        rating = self.ratings[idx]
        features = self.encoded_data[idx]
        return user, item, rating, features


# Collaborative Filtering Model
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_size):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)

    def forward(self, user, item):
        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)
        return (user_embedding * item_embedding).sum(1)


# Content-Based Filtering Model
class ContentBasedNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(ContentBasedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Train models for each cluster
def train_collaborative_filtering(df, cluster_name):
    num_users = df['Staff Number'].nunique()
    num_items = df['Coordinator ID'].nunique()
    embedding_size = 50

    model_cf = MatrixFactorization(num_users, num_items, embedding_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_cf.parameters(), lr=0.01)

    dataset = CustomDataset(df, np.zeros((df.shape[0], 1)))  # No features for CF
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Training loop
    for epoch in range(10):
        model_cf.train()
        running_loss = 0.0
        for user, item, rating, _ in train_loader:
            # Add index checks here
            if user.max() >= num_users or item.max() >= num_items:
                print(
                    f"Index out of range: user max {user.max()}, num_users {num_users}, item max {item.max()}, num_items {num_items}")
                continue

            user = user.long()
            item = item.long()
            rating = rating.float()

            optimizer.zero_grad()
            output = model_cf(user, item)
            loss = criterion(output, rating)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Cluster {cluster_name} - Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

    return model_cf, test_loader


def train_content_based_filtering(df, encoded_data, cluster_name):
    input_size = encoded_data.shape[1]
    output_size = 1

    model_cb = ContentBasedNN(input_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_cb.parameters(), lr=0.01)

    dataset = CustomDataset(df, encoded_data)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Training loop
    for epoch in range(10):
        model_cb.train()
        running_loss = 0.0
        for _, _, rating, features in train_loader:
            features = features.float()
            rating = rating.float()

            optimizer.zero_grad()
            output = model_cb(features).squeeze()
            loss = criterion(output, rating)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Cluster {cluster_name} - Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

    return model_cb, test_loader


model_cf_0, test_loader_0 = train_collaborative_filtering(df_0, '0')
model_cf_1, test_loader_1 = train_collaborative_filtering(df_1, '1')
model_cf_2, test_loader_2 = train_collaborative_filtering(df_2, '2')

model_cb_0, test_loader_cb_0 = train_content_based_filtering(df_0, encoded_data_0, '0')
model_cb_1, test_loader_cb_1 = train_content_based_filtering(df_1, encoded_data_1, '1')
model_cb_2, test_loader_cb_2 = train_content_based_filtering(df_2, encoded_data_2, '2')


# Hybrid Recommendation Function
def hybrid_recommendation(user, item, features, model_cf, model_cb, weight_cf=0.5, weight_cb=0.5):
    model_cf.eval()
    model_cb.eval()

    user = torch.tensor(user).long().unsqueeze(0)
    item = torch.tensor(item).long().unsqueeze(0)
    features = torch.tensor(features).float().unsqueeze(0)

    with torch.no_grad():
        cf_pred = model_cf(user, item).item()
        cb_pred = model_cb(features).item()

    return (weight_cf * cf_pred) + (weight_cb * cb_pred)


# Evaluate Hybrid Model
def evaluate_hybrid_model(test_loader, model_cf, model_cb):
    model_cf.eval()
    model_cb.eval()
    mse = 0.0
    count = 0

    for user, item, rating, features in test_loader:
        for u, i, r, f in zip(user, item, rating, features):
            prediction = hybrid_recommendation(u.item(), i.item(), f, model_cf, model_cb)
            mse += (prediction - r.item()) ** 2
            count += 1

    mse /= count
    return mse


mse_0 = evaluate_hybrid_model(test_loader_0, model_cf_0, model_cb_0)
mse_1 = evaluate_hybrid_model(test_loader_1, model_cf_1, model_cb_1)
mse_2 = evaluate_hybrid_model(test_loader_2, model_cf_2, model_cb_2)

print(f'Cluster 0 - Hybrid Model MSE: {mse_0}')
print(f'Cluster 1 - Hybrid Model MSE: {mse_1}')
print(f'Cluster 2 - Hybrid Model MSE: {mse_2}')


# Recommend modules
def recommend_modules(df, user_id, threshold=60):
    user_df = df[df['Staff Number'] == user_id]
    low_score_modules = user_df[user_df['Assessment_Scores'] < threshold]['Coordinator ID'].tolist()

    recommendations = []
    for module in low_score_modules:
        recommendations.append(module)
        # Add more recommendations based on hybrid model

    return recommendations


# Example usage
user_id = 1001  # Example user ID
cluster_id = 0  # Example cluster ID
if cluster_id == 0:
    recommendations = recommend_modules(df_0, user_id)
elif cluster_id == 1:
    recommendations = recommend_modules(df_1, user_id)
else:
    recommendations = recommend_modules(df_2, user_id)

print(f'Recommendations for user {user_id} in cluster {cluster_id}: {recommendations}')