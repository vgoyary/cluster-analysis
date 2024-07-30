import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Data
# num_users = 100
# num_items = 500
# embedding_dim = 32

user_item_matrix = np.random.randint(2, size=(num_users, num_items))
user_profiles = np.random.rand(num_users, 10)
item_profiles = np.random.rand(num_items, 10)

# Convert data to PyTorch tensors
user_item_matrix = torch.tensor(user_item_matrix, dtype=torch.float32)
user_profiles = torch.tensor(user_profiles, dtype=torch.float32)
item_profiles = torch.tensor(item_profiles, dtype=torch.float32)


# Define Collaborative Filtering Model
class CFModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(CFModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        return (user_embeds * item_embeds).sum(1)


# Define Content-Based Filtering Model
class CBModel(nn.Module):
    def __init__(self, user_profile_dim, item_profile_dim, embedding_dim):
        super(CBModel, self).__init__()
        self.user_fc = nn.Linear(user_profile_dim, embedding_dim)
        self.item_fc = nn.Linear(item_profile_dim, embedding_dim)

    def forward(self, user_profiles, item_profiles):
        user_embeds = self.user_fc(user_profiles)
        item_embeds = self.item_fc(item_profiles)
        return (user_embeds * item_embeds).sum(1)


# Training function
def train_model(model, optimizer, criterion, user_ids, item_ids, labels, epochs=10):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(user_ids, item_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')


# Instantiate models
cf_model = CFModel(num_users, num_items, embedding_dim)
cb_model = CBModel(user_profiles.shape[1], item_profiles.shape[1], embedding_dim)

# Optimizers and loss function
cf_optimizer = optim.Adam(cf_model.parameters(), lr=0.001)
cb_optimizer = optim.Adam(cb_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Sample training data
user_ids = torch.randint(0, num_users, (1000,))
item_ids = torch.randint(0, num_items, (1000,))
labels = torch.tensor([user_item_matrix[u, i] for u, i in zip(user_ids, item_ids)], dtype=torch.float32)

# Train Collaborative Filtering Model
train_model(cf_model, cf_optimizer, criterion, user_ids, item_ids, labels)

# Train Content-Based Filtering Model
# For simplicity, we use the same user_ids and item_ids; adapt as needed for real data
user_profiles_batch = user_profiles[user_ids]
item_profiles_batch = item_profiles[item_ids]
train_model(cb_model, cb_optimizer, criterion, user_profiles_batch, item_profiles_batch, labels)


# Hybrid Recommendation Function
def hybrid_recommendation(cf_model, cb_model, user_id, item_ids, user_profiles, item_profiles, cf_weight=0.5,
                          cb_weight=0.5):
    cf_model.eval()
    cb_model.eval()

    user_ids = torch.tensor([user_id] * len(item_ids), dtype=torch.int64)
    item_ids = torch.tensor(item_ids, dtype=torch.int64)

    with torch.no_grad():
        cf_scores = cf_model(user_ids, item_ids)
        cb_scores = cb_model(user_profiles[user_id].unsqueeze(0).repeat(len(item_ids), 1), item_profiles[item_ids])

    combined_scores = cf_weight * cf_scores + cb_weight * cb_scores
    recommended_items = torch.argsort(combined_scores, descending=True)
    return recommended_items


# Example usage
user_id = 0
item_ids = np.arange(num_items)
recommendations = hybrid_recommendation(cf_model, cb_model, user_id, item_ids, user_profiles, item_profiles)
print("Top 10 Recommendations for user", user_id, ":", recommendations[:10].numpy())
