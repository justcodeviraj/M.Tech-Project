# -*- coding: utf-8 -*-
"""
"""

import torch
import torch.nn as nn
import math
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import scipy.stats as stats
import types

torch.manual_seed(42)

in_channels = 1
img_size = 28
input_dim = in_channels*img_size*img_size
num_hidden_layers = 4
hidden_dims = [1024, 512, 256, 128]
output_dim = 10
batch_size = 32
K = 5
gamma_values = [5, 10, 20, 25, 30, 40, 50]
epochs = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MPLinear(nn.Module):
    def __init__(self, h1, h2, gamma, K):
        super().__init__()
        self.gamma = gamma
        self.K = K
        self.weight = torch.nn.Parameter(torch.empty(h2, h1), requires_grad=True)
        self.k_values = None
        self.mask_percentage = None
        self.total_dims = None

    def MP(self, x):
        B, H, D = x.shape
        sorted_vals, _ = torch.sort(x, dim=1, descending=True)   # (B, H, D)
        cumsums = torch.cumsum(sorted_vals, dim=1)               # (B, H, D)

        idx = torch.arange(1, H+1, device=x.device, dtype=x.dtype).view(1, H, 1)
        thetas = (cumsums - self.gamma) / idx                          # (B, H, D)

        mask = sorted_vals - thetas > 0                           # (B, H, D) bool
        k = mask.sum(dim=1)                                       # (B, D), # of valid i's

        self.k_values = k
        self.mask_percentage = mask.float().mean(dim=1)  # Average over H dimension
        self.total_dims = H

        k_idx = (k - 1).clamp(min=0).unsqueeze(1)                  # (B,1,D)
        alpha = thetas.gather(dim=1, index=k_idx).squeeze(1)      # (B, D)
        alpha = torch.where(k > 0, alpha, torch.zeros_like(alpha))

        return alpha

    def forward(self, x):
        B, D = x.shape
        x = x.unsqueeze(-1)
        w = self.weight.t().unsqueeze(0)
        p = torch.cat([w + x, - w - x], dim=1)
        q = torch.cat([w - x, - w + x], dim=1)
        alpha = self.MP(p)
        beta = self.MP(q)
        wtx = 0.5*self.gamma*D*(alpha-beta)
        return wtx

class NetworkWithHooks(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, K, network="Linear", gamma=1, scale=1):
        super().__init__()
        self.K = K
        self.gamma = gamma
        self.scale = scale
        dims = [input_dim] + hidden_dims + [output_dim]
        if network=="MP":
            self.layers = nn.ModuleList([MPLinear(dims[i], dims[i+1], gamma, K) for i in range(len(dims) - 1)])
        else:
            self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=False) for i in range(len(dims) - 1)])
        self.activations = nn.ModuleList([nn.ReLU() for _ in range(len(self.layers) - 1)])

        # Store intermediate outputs
        self.intermediate_outputs = []
        self.hooks = []

        self.k_values_history = []
        self.mask_percentages_history = []
        self.mp_layers_indices = [i for i, layer in enumerate(self.layers) if isinstance(layer, MPLinear)]

    def normalize(self, x):
        x = self.K*x/(torch.norm(x, dim=1, keepdim=True) + 1e-8)
        return x

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

        # Register hooks after each normalization step
        for i in range(len(self.layers)):
            hook = self.layers[i].register_forward_hook(self._hook_function(i))
            self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_layers(self):
        return list(self.layers)

    def forward(self, x):
        self.k_values_history = []
        self.mask_percentages_history = []

        x = x.view(x.size(0), -1)
        x = self.normalize(x)
        for i in range(len(self.activations)):
            x = self.scale*self.layers[i](x)

            if isinstance(self.layers[i], MPLinear):
                self.k_values_history.append(self.layers[i].k_values.detach())
                self.mask_percentages_history.append(self.layers[i].mask_percentage.detach())

            x = self.activations[i](x)
            x = self.normalize(x)
        x = self.scale*self.layers[-1](x)

        if isinstance(self.layers[-1], MPLinear):
            self.k_values_history.append(self.layers[-1].k_values.detach())
            self.mask_percentages_history.append(self.layers[-1].mask_percentage.detach())

        return x

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

            # Weight normalization
            for layer in model.get_layers():
                with torch.no_grad():
                    W = layer.weight
                    norms = torch.norm(W, p=2, dim=1, keepdim=True)
                    layer.weight.copy_(K*W / (norms + 1e-8))

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
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy
def plot_gamma_results(results, gamma_values, linear_accuracy):

    accuracies = [acc for _, acc in results]

    plt.figure(figsize=(10, 6))

    # Set global font sizes
    plt.rcParams.update({'font.size': 14})

    plt.plot(gamma_values, accuracies, 'o-', color='blue', linewidth=2, markersize=8)
    plt.axhline(y=linear_accuracy, color='r', linestyle='--',
                label=f'MLP Network: {linear_accuracy:.2f}%')

    plt.title(f'MP Network Accuracy vs. Gamma for K={K}', fontsize=18)
    plt.xlabel('Gamma', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)

    # Annotate points
    for gamma, acc in results:
        plt.annotate(f'{acc:.2f}%',
                     (gamma, acc),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center',
                     fontsize=12)

    plt.tight_layout()
    plt.savefig('gamma_K2.png')
    plt.close()
# Function to find optimal gamma for MP network using fixed K
def find_optimal_gamma(linear_net, test_loader, K):

    results = []
    best_accuracy = 0
    best_gamma = 0

    linear_accuracy = evaluate(linear_net, test_loader)
    print(f"Linear Network Accuracy: {linear_accuracy:.2f}%")
    print(f"Using fixed K value: {K}")

    for gamma in gamma_values:
        mp_net = NetworkWithHooks(input_dim, hidden_dims, output_dim, K, network="MP", gamma=gamma, scale=0.01).to(device)
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

train_loader, test_loader = load_mnist()

linear_net = NetworkWithHooks(input_dim, hidden_dims, output_dim, K).to(device)
print(f"- Number of parameters: {sum(p.numel() for p in linear_net.parameters())}")

print("Training Linear Network...")
print(f"Using K = {K} for weight normalization")

train_losses, train_accs = train_linear_network(linear_net, train_loader, epochs, K=K)

test_accuracy = evaluate(linear_net, test_loader)
print(f"\nLinear Network Test Accuracy: {test_accuracy:.2f}%")

print("Finding optimal gamma for MP Network...")
best_gamma, best_accuracy = find_optimal_gamma(linear_net, test_loader, K)


mp_net = NetworkWithHooks(input_dim, hidden_dims, output_dim, K, network="MP", gamma=best_gamma, scale=0.01).to(device)
mp_net.load_state_dict(linear_net.state_dict())

print(f"\nFinal Results:")
print(f"Linear Network (K={K}): {test_accuracy:.2f}%")
print(f"MP Network (gamma={best_gamma}, K={K}): {best_accuracy:.2f}%")
print(f"Difference: {best_accuracy - test_accuracy:.2f}%")


def compare_hidden_representations(model1, model2, data_loader):
    """
    Compare hidden representations of two models layer by layer

    Args:
        model1, model2: Two models with the same structure
        data_loader: DataLoader containing input data
        device: Device to run the models on

    Returns:
        Dictionary with comparison metrics between layers
    """
    model1.to(device)
    model2.to(device)
    model1.eval()
    model2.eval()

    # Register hooks to capture intermediate outputs
    model1.register_hooks()
    model2.register_hooks()

    # Get a batch of data
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(device)
            _ = model1(data)
            _ = model2(data)
            break  # We only need one batch for analysis

    results = {}

    # Compare layer by layer
    for layer_idx in range(len(model1.intermediate_outputs)):
        layer_name = f"layer_{layer_idx}"
        results[layer_name] = {}

        activations1 = model1.intermediate_outputs[layer_idx].cpu().numpy()
        activations2 = model2.intermediate_outputs[layer_idx].cpu().numpy()

        # 1. Check if dimensions match
        results[layer_name]["dims_match"] = (activations1.shape == activations2.shape)

        if not results[layer_name]["dims_match"]:
            continue

        # 2. Compute correlations between corresponding neurons
        neuron_correlations = []
        for neuron_idx in range(activations1.shape[1]):
            corr, p_value = stats.pearsonr(activations1[:, neuron_idx], activations2[:, neuron_idx])
            neuron_correlations.append((corr, p_value))

        results[layer_name]["neuron_correlations"] = neuron_correlations
        results[layer_name]["avg_correlation"] = np.mean([c for c, _ in neuron_correlations])
        results[layer_name]["median_correlation"] = np.median([c for c, _ in neuron_correlations])
        results[layer_name]["significant_correlations"] = sum(1 for _, p in neuron_correlations if p < 0.05)

        # 3. Compute CKA (Centered Kernel Alignment) similarity
        results[layer_name]["cka_similarity"] = compute_cka(activations1, activations2)

        # 4. Linear regression: can we predict activations of model2 from model1?
        X_flat = activations1.reshape(activations1.shape[0], -1)
        y_flat = activations2.reshape(activations2.shape[0], -1)

        reg = LinearRegression().fit(X_flat, y_flat)
        y_pred = reg.predict(X_flat)
        r2 = r2_score(y_flat, y_pred)
        results[layer_name]["linear_relationship_r2"] = r2

        # 5. Distribution comparison (Kolmogorov-Smirnov test)
        ks_stats = []
        for neuron_idx in range(activations1.shape[1]):
            stat, p_value = stats.ks_2samp(activations1[:, neuron_idx], activations2[:, neuron_idx])
            ks_stats.append((stat, p_value))

        results[layer_name]["ks_stats"] = ks_stats
        results[layer_name]["same_distribution_percentage"] = 100 * sum(1 for _, p in ks_stats if p > 0.05) / len(ks_stats)

    # Clean up hooks
    model1.remove_hooks()
    model2.remove_hooks()

    return results


def compute_cka(X, Y):
    """
    Compute Centered Kernel Alignment (CKA) between two matrices
    """
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)

    XXT = X @ X.T
    YYT = Y @ Y.T

    return (np.trace(XXT @ YYT)) / (np.linalg.norm(XXT, 'fro') * np.linalg.norm(YYT, 'fro'))


def plot_comparison_results(results):
    """
    Visualize the comparison results
    """
    layer_names = list(results.keys())

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Comparison of Hidden Layer Representations', fontsize=16)

    # 1. Average correlation per layer
    avg_corrs = [results[layer]["avg_correlation"] for layer in layer_names]
    axes[0, 0].bar(layer_names, avg_corrs)
    axes[0, 0].set_title('Average Correlation Between Corresponding Neurons')
    axes[0, 0].set_ylim(-1, 1)
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. CKA similarity
    cka_sims = [results[layer]["cka_similarity"] for layer in layer_names]
    axes[0, 1].bar(layer_names, cka_sims)
    axes[0, 1].set_title('CKA Similarity')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. Linear relationship R²
    r2s = [results[layer]["linear_relationship_r2"] for layer in layer_names]
    axes[1, 0].bar(layer_names, r2s)
    axes[1, 0].set_title('Linear Relationship R²')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 4. Same distribution percentage
    same_dist = [results[layer]["same_distribution_percentage"] for layer in layer_names]
    axes[1, 1].bar(layer_names, same_dist)
    axes[1, 1].set_title('Neurons with Same Distribution (%)')
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    return fig


def visualize_neuron_activations(model1, model2, data_loader, layer_idx=0, num_neurons=5):
    """
    Visualize activations of specific neurons from both models
    """
    model1.to(device)
    model2.to(device)
    model1.eval()
    model2.eval()

    # Register hooks
    model1.register_hooks()
    model2.register_hooks()

    # Get activations
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(device)
            _ = model1(data)
            _ = model2(data)
            break

    activations1 = model1.intermediate_outputs[layer_idx].cpu().numpy()
    activations2 = model2.intermediate_outputs[layer_idx].cpu().numpy()

    # Clean up hooks
    model1.remove_hooks()
    model2.remove_hooks()

    # Visualize selected neurons
    fig, axes = plt.subplots(num_neurons, 1, figsize=(10, 3*num_neurons))

    for i in range(min(num_neurons, activations1.shape[1])):
        ax = axes[i] if num_neurons > 1 else axes

        ax.scatter(activations1[:, i], activations2[:, i], alpha=0.6)

        # Add correlation information
        corr, p_value = stats.pearsonr(activations1[:, i], activations2[:, i])

        # Add regression line
        x_line = np.linspace(min(activations1[:, i]), max(activations1[:, i]), 100)
        reg = LinearRegression().fit(activations1[:, i].reshape(-1, 1), activations2[:, i])
        y_line = reg.predict(x_line.reshape(-1, 1))

        ax.plot(x_line, y_line, 'r-', linewidth=2)
        ax.set_title(f"Neuron {i}: r={corr:.3f}, p={p_value:.3e}, slope={reg.coef_[0]:.3f}")
        ax.set_xlabel(f"Model 1 - Layer {layer_idx}, Neuron {i}")
        ax.set_ylabel(f"Model 2 - Layer {layer_idx}, Neuron {i}")

    plt.tight_layout()
    return fig


def plot_activation_distributions(model1, model2, data_loader, layer_idx=0, num_neurons=5):
    """
    Plot activation distributions for each model
    """
    model1.to(device)
    model2.to(device)
    model1.eval()
    model2.eval()

    # Register hooks
    model1.register_hooks()
    model2.register_hooks()

    # Get activations
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(device)
            _ = model1(data)
            _ = model2(data)
            break

    activations1 = model1.intermediate_outputs[layer_idx].cpu().numpy()
    activations2 = model2.intermediate_outputs[layer_idx].cpu().numpy()

    # Clean up hooks
    model1.remove_hooks()
    model2.remove_hooks()

    # Visualize distributions
    fig, axes = plt.subplots(num_neurons, 1, figsize=(10, 3*num_neurons))

    for i in range(min(num_neurons, activations1.shape[1])):
        ax = axes[i] if num_neurons > 1 else axes

        # Plot histograms
        ax.hist(activations1[:, i], bins=30, alpha=0.5, label="Model 1")
        ax.hist(activations2[:, i], bins=30, alpha=0.5, label="Model 2")

        # KS test for distribution comparison
        ks_stat, p_value = stats.ks_2samp(activations1[:, i], activations2[:, i])

        ax.set_title(f"Neuron {i}: KS stat={ks_stat:.3f}, p={p_value:.3e}", fontsize=16)
        ax.set_xlabel(f"Activation value", fontsize=16)
        ax.set_ylabel("Frequency", fontsize=16)
        ax.legend()

    plt.tight_layout()
    return fig


def perform_pca_analysis(model1, model2, data_loader, layer_idx=0, n_components=2):
    """
    Perform PCA on layer activations and visualize
    """
    model1.to(device)
    model2.to(device)
    model1.eval()
    model2.eval()

    # Register hooks
    model1.register_hooks()
    model2.register_hooks()

    # Get activations
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(device)
            _ = model1(data)
            _ = model2(data)
            break

    activations1 = model1.intermediate_outputs[layer_idx].cpu().numpy()
    activations2 = model2.intermediate_outputs[layer_idx].cpu().numpy()

    # Clean up hooks
    model1.remove_hooks()
    model2.remove_hooks()

    # Perform PCA
    pca1 = PCA(n_components=n_components)
    pca2 = PCA(n_components=n_components)

    reduced1 = pca1.fit_transform(activations1)
    reduced2 = pca2.fit_transform(activations2)

    # Plot PCA results
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    if n_components >= 2:
        axes[0].scatter(reduced1[:, 0], reduced1[:, 1])
        axes[0].set_title(f"Model 1 - Layer {layer_idx} PCA")
        axes[0].set_xlabel(f"PC1 ({pca1.explained_variance_ratio_[0]:.2%})")
        axes[0].set_ylabel(f"PC2 ({pca1.explained_variance_ratio_[1]:.2%})")

        axes[1].scatter(reduced2[:, 0], reduced2[:, 1])
        axes[1].set_title(f"Model 2 - Layer {layer_idx} PCA")
        axes[1].set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.2%})")
        axes[1].set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.2%})")

    plt.tight_layout()

    # Also compute canonical correlation between the two representations
    if activations1.shape[0] >= activations1.shape[1] and activations2.shape[0] >= activations2.shape[1]:
        from sklearn.cross_decomposition import CCA

        min_dim = min(activations1.shape[1], activations2.shape[1])
        n_components_cca = min(n_components, min_dim)

        cca = CCA(n_components=n_components_cca)
        cca.fit(activations1, activations2)
        X_c, Y_c = cca.transform(activations1, activations2)

        # Calculate correlations between canonical variables
        correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components_cca)]

        # Create a figure for CCA
        fig_cca, ax_cca = plt.subplots(figsize=(10, 6))

        for i in range(n_components_cca):
            ax_cca.scatter(X_c[:, i], Y_c[:, i], label=f"Component {i+1}, r={correlations[i]:.3f}")

        ax_cca.set_title("Canonical Correlation Analysis")
        ax_cca.set_xlabel("Model 1 Canonical Variables")
        ax_cca.set_ylabel("Model 2 Canonical Variables")
        ax_cca.legend()

        return fig, fig_cca, correlations
    else:
        return fig, None, None

import types

# Example usage:
def main():

    # Compare hidden representations
    results = compare_hidden_representations(linear_net, mp_net, test_loader)

    # Plot the results
    plot_comparison_results(results)

    # Visualize neuron activations
    visualize_neuron_activations(linear_net, mp_net, train_loader, layer_idx=0, num_neurons=3)

    # Plot activation distributions
    plot_activation_distributions(linear_net, mp_net, train_loader, layer_idx=0, num_neurons=3)

    # Perform PCA analysis
    perform_pca_analysis(linear_net, mp_net, train_loader, layer_idx=0)

    plt.show()


if __name__ == "__main__":
    main()