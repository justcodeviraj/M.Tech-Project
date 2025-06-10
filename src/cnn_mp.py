
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm
torch.manual_seed(42)

class Linear(nn.Module):
    def __init__(self, h1, h2, K, normalize=False):
        super().__init__()
        self.K = K
        self.normalize = normalize
        self.weight = torch.nn.Parameter(torch.empty(h2, h1), requires_grad=True)
        nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        w = self.weight.t()
        if self.normalize:
            x = self.K*x/(torch.norm(x, dim=1, keepdim=True) + 1e-8)
            w = self.K*w/(torch.norm(w, dim=0, keepdim=True) + 1e-8)
        x = torch.matmul(x,w)
        return x


class MPLinear(nn.Module):
    def __init__(self, h1, h2, gamma, K, normalize=False):
        super().__init__()
        self.gamma = gamma
        self.K = K
        self.normalize = normalize
        self.weight = torch.nn.Parameter(torch.empty(h2, h1), requires_grad=True)
        nn.init.xavier_normal_(self.weight)

    def MP(self, x):
        B, H, D = x.shape
        sorted_vals, _ = torch.sort(x, dim=1, descending=True)   # (B, H, D)
        cumsums = torch.cumsum(sorted_vals, dim=1)               # (B, H, D)

        idx = torch.arange(1, H+1, device=x.device, dtype=x.dtype).view(1, H, 1)
        thetas = (cumsums - self.gamma) / idx                          # (B, H, D)

        mask = sorted_vals - thetas > 0                           # (B, H, D) bool
        k = mask.sum(dim=1)                                       # (B, D), # of valid i's

        k_idx = (k - 1).clamp(min=0).unsqueeze(1)                  # (B,1,D)
        alpha = thetas.gather(dim=1, index=k_idx).squeeze(1)      # (B, D)
        alpha = torch.where(k > 0, alpha, torch.zeros_like(alpha))

        return alpha

    def forward(self, x):
        B, D = x.shape
        w = self.weight.t()
        if self.normalize:
            x = self.K*x/(torch.norm(x, dim=1, keepdim=True) + 1e-8)
            w = self.K*w/(torch.norm(w, dim=0, keepdim=True) + 1e-8)
        x = x.unsqueeze(-1)
        w = w.unsqueeze(0)
        p = torch.cat([w + x, - w - x], dim=1)
        q = torch.cat([w - x, - w + x], dim=1)
        alpha = self.MP(p)
        beta = self.MP(q)
        wtx = 0.5*self.gamma*D*(alpha-beta)
        return wtx

class Conv2d(nn.Module):
    def __init__(self, C_in, C_out, kernel_size,  padding=0, stride=1, K=1, normalize=False):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.K = K
        self.normalize = normalize
        self.weight = torch.nn.Parameter(torch.empty(C_out, C_in, kernel_size, kernel_size), requires_grad=True)
        nn.init.xavier_normal_(self.weight)


    def forward(self, x):
        batch_size, C_in, H, W = x.shape
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        w = self.weight.view(self.weight.size(0), -1).t() # [C_in*k*k, C_out]
        patches = F.unfold(x, (self.kernel_size,self.kernel_size), padding=self.padding, stride=self.stride)             # [batch_size, C_in*k*k, L]

        if self.normalize:
            patches = self.K*patches/(torch.norm(patches, dim=-1, keepdim=True) + 1e-8)
            w = self.K*w/(torch.norm(w, dim=0, keepdim=True) + 1e-8)

        patches = patches.permute(0,2,1)                                       # [batch_size, L, C_in*k*k]
        conv = torch.matmul(patches, w)

        return conv.permute(0,2,1).reshape(batch_size, self.C_out, H_out, W_out)

class MPConv2d(nn.Module):
    def __init__(self, C_in, C_out, kernel_size,  padding=0, stride=1, gamma=1, K=1, normalize=False):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.gamma = gamma
        self.K = K
        self.normalize = normalize
        self.weight = torch.nn.Parameter(torch.empty(C_out, C_in, kernel_size, kernel_size), requires_grad=True)
        nn.init.xavier_normal_(self.weight)

    def MP(self, x, gamma, D):
        B, L, H, O = x.shape
        x_reshaped = x.permute(0, 1, 3, 2).reshape(-1, H)
        sorted_vals, _ = torch.sort(x_reshaped, dim=1, descending=True)

        cum_sum = torch.cumsum(sorted_vals, dim=1)
        k_values = torch.zeros(B * L * O, dtype=torch.long, device=x.device) - 1

        for i in range(H):
            alpha_candidates = (cum_sum[:, i] - gamma) / (i + 1)
            valid_indices = sorted_vals[:, i] - alpha_candidates >= 0
            k_values[valid_indices] = i + 1

        result = torch.zeros(B * L * O, device=x.device)
        valid_k_mask = k_values > 0

        if valid_k_mask.any():
            valid_indices = torch.arange(B * L * O, device=x.device)[valid_k_mask]
            k_indices = k_values[valid_k_mask] - 1
            cum_sum_at_k = cum_sum[valid_indices, k_indices]
            alphas = (cum_sum_at_k - gamma) / k_values[valid_k_mask]
            result[valid_k_mask] = gamma * D * alphas

        return result.reshape(B, L, O)


    def forward(self, x):
        batch_size, C_in, H, W = x.shape
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        patches = F.unfold(x, (self.kernel_size, self.kernel_size), padding=self.padding, stride=self.stride)                              # [batch_size, C_in*k*k, L]
        w = self.weight.view(self.weight.size(0), -1).t()           # [C_in*k*k, C_out]

        if self.normalize:
            patches = self.K*patches/(torch.norm(patches, dim=-1, keepdim=True) + 1e-8)
            w = self.K*w/(torch.norm(w, dim=0, keepdim=True) + 1e-8)

        patches = patches.permute(0,2,1).unsqueeze(-1)              # [batch_size, L, C_in*k*k, 1]
        w = w.unsqueeze(0).unsqueeze(0)                             # [1, 1, C_in*k*k, C_out]

        p = torch.cat([patches+w, -patches-w],dim=2)
        q = torch.cat([patches-w, -patches+w],dim=2)
        D = p.size(2)
        alpha = self.MP(p, self.gamma, D)                     # [batch_size, L, C_out]
        beta = self.MP(q, self.gamma, D)
        conv = 0.5*(alpha-beta)
        return conv.permute(0,2,1).reshape(batch_size, self.C_out, H_out, W_out)



in_channels = 1
img_size = 28
hidden_dims = [6, 16, 120, 84, 36]
output_dim = 10
kernel_sizes = [5, 5, 5, 5, 5, 3]
batch_size = 32
K = 5
gamma_values = [5]
# gamma_values = [1, 2]
epochs = 1
normalize = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LeNet(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K
        self.conv1 = Conv2d(1, 6, kernel_size=5, padding=2, K=K, normalize=normalize)
        self.conv2 = Conv2d(6, 16, kernel_size=5, padding=2, K=K, normalize=normalize)
        self.conv3 = Conv2d(16, 120, kernel_size=5, padding=2, K=K, normalize=normalize)
        self.conv4 = Conv2d(120, 84, kernel_size=3, padding=0, K=K, normalize=normalize)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = Linear(84, 10, K=K, normalize=normalize)

        self.intermediate_outputs = []
        self.hooks = []

    def get_layers(self):
        return [self.conv1, self.conv2, self.conv3, self.conv4, self.fc]

    def _hook_function(self, i):
        def hook(module, input, output):
            if i < len(self.intermediate_outputs):
                self.intermediate_outputs[i] = output.detach()
            else:
                self.intermediate_outputs.append(output.detach())
        return hook

    def register_hooks(self):
        self.intermediate_outputs = []
        self.hooks = []

        for i, layer in enumerate(self.get_layers()):
            hook = layer.register_forward_hook(self._hook_function(i))
            self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class MP_LeNet(nn.Module):
    def __init__(self, gamma, K, scale=0.01):
        super().__init__()

        self.gamma = gamma
        self.K = K
        self.scale = scale
        self.conv1 = MPConv2d(1, 6, kernel_size=5, padding=2, gamma=gamma, K=K, normalize=normalize)
        self.conv2 = MPConv2d(6, 16, kernel_size=5, padding=2, gamma=gamma, K=K, normalize=normalize)
        self.conv3 = MPConv2d(16, 120, kernel_size=5, padding=2, gamma=gamma, K=K, normalize=normalize)
        self.conv4 = MPConv2d(120, 84, kernel_size=3, padding=0, gamma=gamma, K=K, normalize=normalize)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = MPLinear(84, 10, gamma=gamma, K=K, normalize=normalize)

        self.intermediate_outputs = []
        self.hooks = []

    def get_layers(self):
        return [self.conv1, self.conv2, self.conv3, self.conv4, self.fc]

    def _hook_function(self, i):
        def hook(module, input, output):
            if i < len(self.intermediate_outputs):
                self.intermediate_outputs[i] = output.detach()
            else:
                self.intermediate_outputs.append(output.detach())
        return hook

    def register_hooks(self):
        self.intermediate_outputs = []
        self.hooks = []

        for i, layer in enumerate(self.get_layers()):
            hook = layer.register_forward_hook(self._hook_function(i))
            self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def forward(self, x):
        x = self.pool(F.relu(self.scale*self.conv1(x)))
        x = self.pool(F.relu(self.scale*self.conv2(x)))
        x = self.pool(F.relu(self.scale*self.conv3(x)))
        x = F.relu(self.scale*self.conv4(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Load MNIST dataset
def load_mnist():
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(28, padding=4),  # MNIST image size is 28x28
        torchvision.transforms.RandomRotation(10),         # Small rotation for augmentation
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),  # Standard MNIST mean and std
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

def load_cifar10(batch_size=128):
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),  # CIFAR-10 image size is 32x32
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # CIFAR-10 mean & std
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

def train_linear_network(model, train_loader, epochs, K):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    train_accs = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # epoch statistics
        epoch_acc = 100 * correct / total
        train_losses.append(running_loss / len(train_loader))
        train_accs.append(epoch_acc)

        print(f'Epoch [{epoch+1}/{epochs}], Training Accuracy: {epoch_acc:.2f}%')

    return train_losses, train_accs

# Function to evaluate network performance
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Function to find optimal gamma for MP network using fixed K
def find_optimal_gamma(linear_net, test_loader, K):

    results = []
    best_accuracy = 0
    best_gamma = 0

    linear_accuracy = evaluate(linear_net, test_loader)
    print(f"Linear Network Accuracy: {linear_accuracy:.2f}%")
    print(f"Using fixed K value: {K}")

    for gamma in gamma_values:
        mp_net = MP_LeNet(gamma=gamma, K=K).to(device)
        mp_net.load_state_dict(linear_net.state_dict())

        mp_accuracy = evaluate(mp_net, test_loader)
        print(f"MP Network with gamma={gamma}, K={K}: Accuracy = {mp_accuracy:.2f}%")
        results.append((gamma, mp_accuracy))

        if mp_accuracy > best_accuracy:
            best_accuracy = mp_accuracy
            best_gamma = gamma

    print(f"\nBest MP Network: gamma={best_gamma}, K={K}, Accuracy={best_accuracy:.2f}%")
    print(f"Difference from Linear Network: {best_accuracy - linear_accuracy:.2f}%")

    plot_gamma_results(results, gamma_values, linear_accuracy)

    return best_gamma, best_accuracy

def plot_gamma_results(results, gamma_values, linear_accuracy):

    accuracies = [acc for _, acc in results]

    plt.figure(figsize=(10, 6))
    plt.plot(gamma_values, accuracies, 'o-', color='blue', linewidth=2, markersize=8)
    plt.axhline(y=linear_accuracy, color='r', linestyle='--',
                label=f'Linear Network: {linear_accuracy:.2f}%')

    plt.title(f'MP Network Accuracy vs. Gamma for K={K}')
    plt.xlabel('Gamma')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Annotate points
    for gamma, acc in results:
        plt.annotate(f'{acc:.2f}%',
                    (gamma, acc),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center')

    plt.tight_layout()
    plt.savefig('gamma_results.png')



train_loader, test_loader = load_mnist()

linear_net = LeNet(K=K).to(device)
print(f"- Number of parameters: {sum(p.numel() for p in linear_net.parameters())}")

print("Training Linear Network...")
print(f"Using K = {K} for weight normalization")

train_losses, train_accs = train_linear_network(linear_net, train_loader, epochs, K=K)

test_accuracy = evaluate(linear_net, test_loader)
print(f"\nLinear Network Test Accuracy: {test_accuracy:.2f}%")


print("\nFinding optimal gamma for MP Network...")
best_gamma, best_accuracy = find_optimal_gamma(linear_net, test_loader, K)


mp_net = MP_LeNet(gamma=best_gamma, K=K).to(device)
mp_net.load_state_dict(linear_net.state_dict())

print(f"\nFinal Results:")
print(f"Linear Network (K={K}): {test_accuracy:.2f}%")
print(f"MP Network (gamma={best_gamma}, K={K}): {best_accuracy:.2f}%")
print(f"Difference: {best_accuracy - test_accuracy:.2f}%")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ks_2samp
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression

def analyze_hidden_representations(model1, model2, data_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Analyze hidden representations between two LeNet models.

    Args:
        model1: First LeNet model
        model2: Second LeNet model
        data_loader: DataLoader containing input data
        device: Device to run models on

    Returns:
        Dictionary of analysis results
    """
    model1.to(device)
    model2.to(device)
    model1.eval()
    model2.eval()

    # Register hooks to capture intermediate outputs
    model1.register_hooks()
    model2.register_hooks()

    results = {
        'layer_correlation': [],
        'layer_ks_test': [],
        'layer_mutual_info': [],
        'layer_names': []
    }

    # Get a batch of data
    for batch_idx, (data, _) in enumerate(data_loader):
        data = data.to(device)

        # Forward pass through both models
        with torch.no_grad():
            _ = model1(data)
            _ = model2(data)

        # Analyze representations for each layer
        for i in range(len(model1.intermediate_outputs)):
            layer_name = f"Layer {i+1}"
            results['layer_names'].append(layer_name)

            # Get hidden representations
            hidden1 = model1.intermediate_outputs[i].cpu().numpy()
            hidden2 = model2.intermediate_outputs[i].cpu().numpy()

            # Reshape to 2D if needed
            hidden1_2d = hidden1.reshape(hidden1.shape[0], -1)
            hidden2_2d = hidden2.reshape(hidden2.shape[0], -1)

            # Calculate correlation
            correlations = calculate_representation_correlation(hidden1_2d, hidden2_2d)
            results['layer_correlation'].append(correlations)

            # Calculate KS test for distribution similarity
            ks_results = calculate_distribution_similarity(hidden1_2d, hidden2_2d)
            results['layer_ks_test'].append(ks_results)

            # Calculate mutual information
            mi = calculate_mutual_information(hidden1_2d, hidden2_2d)
            results['layer_mutual_info'].append(mi)

        # Only analyze one batch for efficiency
        break

    # Remove hooks
    model1.remove_hooks()
    model2.remove_hooks()

    return results

def calculate_representation_correlation(hidden1, hidden2):
    """Calculate correlation between hidden representations"""
    # For efficiency, sample a subset of neurons if there are too many
    max_neurons = 100
    if hidden1.shape[1] > max_neurons:
        indices = np.random.choice(hidden1.shape[1], max_neurons, replace=False)
        hidden1 = hidden1[:, indices]
        hidden2 = hidden2[:, indices]

    # Calculate average activation for each neuron across the batch
    mean_act1 = np.mean(hidden1, axis=0)
    mean_act2 = np.mean(hidden2, axis=0)

    # Calculate Pearson correlation
    corr, p_value = pearsonr(mean_act1, mean_act2)

    # Calculate CKA similarity (a more robust measure for neural networks)
    cka_sim = centered_kernel_alignment(hidden1, hidden2)

    return {
        'pearson_correlation': corr,
        'pearson_p_value': p_value,
        'cka_similarity': cka_sim
    }

def centered_kernel_alignment(X, Y):
    """Calculate Centered Kernel Alignment (CKA) between two representations"""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)

    # Linear kernels
    XTX = X.T @ X
    YTY = Y.T @ Y

    # CKA score
    XTY = X.T @ Y
    cka = (np.linalg.norm(XTY, 'fro') ** 2) / (np.linalg.norm(XTX, 'fro') * np.linalg.norm(YTY, 'fro'))

    return cka

def calculate_distribution_similarity(hidden1, hidden2):
    """Test if the representations come from the same distribution"""
    results = {
        'neurons_from_same_dist_percent': 0,
        'average_ks_statistic': 0,
        'average_ks_pvalue': 0
    }

    # Limit the number of neurons to test for efficiency
    max_neurons = 50
    n_neurons = min(hidden1.shape[1], max_neurons)
    neurons_indices = np.random.choice(hidden1.shape[1], n_neurons, replace=False)

    same_dist_count = 0
    ks_stats = []
    ks_pvals = []

    for idx in neurons_indices:
        # Get activations for a single neuron across all samples
        neuron1 = hidden1[:, idx]
        neuron2 = hidden2[:, idx]

        # KS test to check if distributions are the same
        ks_stat, p_value = ks_2samp(neuron1, neuron2)
        ks_stats.append(ks_stat)
        ks_pvals.append(p_value)

        # If p-value > 0.05, we fail to reject null hypothesis that distributions are the same
        if p_value > 0.05:
            same_dist_count += 1

    results['neurons_from_same_dist_percent'] = (same_dist_count / n_neurons) * 100
    results['average_ks_statistic'] = np.mean(ks_stats)
    results['average_ks_pvalue'] = np.mean(ks_pvals)

    return results

def calculate_mutual_information(hidden1, hidden2):
    """Calculate mutual information between representations"""

    # Ensure there are enough samples for PCA and MI calculation
    if hidden1.shape[0] <= 1 or hidden2.shape[0] <= 1:
        return {
            'avg_mutual_info': np.nan,
            'max_mutual_info': np.nan
        }


    # Dynamically determine n_components based on the minimum dimension
    # n_components cannot be greater than min(n_samples, n_features)
    max_pca_components = 50 # Still limit to 50 if possible for performance
    n_samples = hidden1.shape[0] # Batch size
    n_features = hidden1.shape[1] # Number of features
    n_components = min(n_samples, n_features, max_pca_components)

    # If n_components is 0 or 1, PCA might not be meaningful or possible.
    # Mutual information regression also requires enough samples relative to features.
    if n_components < 2 or n_samples <= n_components:
        # Fallback or return nan if PCA or MI is not applicable
         return {
            'avg_mutual_info': np.nan,
            'max_mutual_info': np.nan
        }


    # Dimensionality reduction if needed (PCA)
    # Only apply PCA if the number of features is large AND we have enough samples
    if n_features > n_components: # Check against the determined n_components
        pca = PCA(n_components=n_components)
        hidden1_reduced = pca.fit_transform(hidden1)
        hidden2_reduced = pca.fit_transform(hidden2)
    else:
        hidden1_reduced = hidden1
        hidden2_reduced = hidden2


    # Calculate mutual information for a subset of neuron pairs for efficiency
    mi_scores = []
    # Ensure we don't try to index more neurons than available in reduced data
    num_neurons_to_compare = min(10, hidden1_reduced.shape[1], hidden2_reduced.shape[1])


    # Ensure we have enough samples for mutual_info_regression
    # It generally requires n_samples > n_features, which is guaranteed by how n_components was chosen for PCA,
    # but still good to be cautious. For a single feature, it's n_samples > 1.
    if hidden1_reduced.shape[0] <= 1 or hidden2_reduced.shape[0] <= 1:
        return {
            'avg_mutual_info': np.nan,
            'max_mutual_info': np.nan
        }


    for i in range(num_neurons_to_compare):
        # mutual_info_regression expects X to be 2D (n_samples, n_features) and y to be 1D (n_samples,)
        # We are comparing neuron i from hidden1 with neuron i from hidden2
        X_mi = hidden1_reduced[:, i:i+1] # Make it 2D (n_samples, 1)
        y_mi = hidden2_reduced[:, i]     # Keep it 1D (n_samples,)

        # Check if X_mi has enough samples
        if X_mi.shape[0] > 1:
            try:
                mi = mutual_info_regression(X_mi, y_mi, random_state=42) # Add random_state for reproducibility
                mi_scores.append(mi[0])
            except ValueError as e:
                print(f"Could not calculate mutual_info_regression for neuron pair {i}: {e}")
                # Append np.nan or skip this pair if calculation fails
                mi_scores.append(np.nan)
        else:
            mi_scores.append(np.nan) # Cannot calculate MI with only one sample


    # Filter out potential NaN scores before calculating mean/max
    valid_mi_scores = [score for score in mi_scores if not np.isnan(score)]


    return {
        'avg_mutual_info': np.mean(valid_mi_scores) if valid_mi_scores else np.nan,
        'max_mutual_info': np.max(valid_mi_scores) if valid_mi_scores else np.nan
    }

def visualize_results(results):
    """Visualize analysis results"""
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot correlations
    correlations = [r['pearson_correlation'] for r in results['layer_correlation']]
    cka_similarities = [r['cka_similarity'] for r in results['layer_correlation']]

    axs[0].bar(results['layer_names'], correlations, alpha=0.7, label='Pearson Correlation')
    axs[0].bar(results['layer_names'], cka_similarities, alpha=0.5, label='CKA Similarity')
    axs[0].set_xlabel('Layer')
    axs[0].set_ylabel('Correlation/Similarity')
    axs[0].set_title('Hidden Representation Correlation by Layer')
    axs[0].legend()

    # Plot KS test results
    same_dist_percent = [r['neurons_from_same_dist_percent'] for r in results['layer_ks_test']]
    axs[1].bar(results['layer_names'], same_dist_percent)
    axs[1].set_xlabel('Layer')
    axs[1].set_ylabel('% Neurons from Same Distribution')
    axs[1].set_title('KS Test Results by Layer')

    # Plot mutual information
    mi_scores = [r['avg_mutual_info'] for r in results['layer_mutual_info']]
    axs[2].bar(results['layer_names'], mi_scores)
    axs[2].set_xlabel('Layer')
    axs[2].set_ylabel('Average Mutual Information')
    axs[2].set_title('Mutual Information by Layer')

    plt.tight_layout()
    return fig

# Example usage
def main():
    # This assumes you have two LeNet models initialized and a data loader

    results = analyze_hidden_representations(linear_net,mp_net, test_loader)
    fig = visualize_results(results)
    plt.show()

    # You can also compare specific metrics
    print(f"Layer 1 correlation: {results['layer_correlation'][0]['pearson_correlation']}")
    print(f"Layer 1 CKA similarity: {results['layer_correlation'][0]['cka_similarity']}")
    print(f"Layer 1 distribution similarity: {results['layer_ks_test'][0]['neurons_from_same_dist_percent']}%")
    pass

def compare_model_distributions(model1, model2, data_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Run a complete comparison between two models to check if their hidden representations
    come from the same distribution.

    Args:
        model1: First LeNet model
        model2: Second LeNet model
        data_loader: DataLoader containing input data
        device: Device to run models on

    Returns:
        Dictionary with analysis results and a conclusion
    """
    # Run the analysis
    results = analyze_hidden_representations(model1, model2, data_loader, device)

    # Calculate average metrics across all layers
    avg_correlation = np.mean([r['pearson_correlation'] for r in results['layer_correlation']])
    avg_cka = np.mean([r['cka_similarity'] for r in results['layer_correlation']])
    avg_same_dist = np.mean([r['neurons_from_same_dist_percent'] for r in results['layer_ks_test']])
    avg_mutual_info = np.mean([r['avg_mutual_info'] for r in results['layer_mutual_info']])

    # Determine if models have similar representations
    similar_correlation = avg_correlation > 0.7
    similar_cka = avg_cka > 0.7
    similar_distribution = avg_same_dist > 70
    high_mutual_info = avg_mutual_info > 0.5  # Threshold depends on your specific use case

    # Overall conclusion
    is_similar = sum([similar_correlation, similar_cka, similar_distribution, high_mutual_info]) >= 3

    conclusion = {
        'average_correlation': avg_correlation,
        'average_cka': avg_cka,
        'average_same_distribution_percent': avg_same_dist,
        'average_mutual_information': avg_mutual_info,
        'models_have_similar_representations': is_similar,
        'conclusion': "The models appear to have hidden representations from the same distribution." if is_similar
                      else "The models have hidden representations from different distributions."
    }

    return {**results, **{'summary': conclusion}}

if __name__ == "__main__":
    main()