import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_kan_first_layer(model, n_points=100, grid_range=[-1, 1], feature_names=None, output_names=None):

    kan_layer = model.layers[0]
    in_features = kan_layer.in_features
    out_features = kan_layer.out_features
    x = torch.linspace(grid_range[0], grid_range[1], n_points)
    n_cols = min(5, out_features)  # 每行最多5个子图
    n_rows = (out_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if out_features == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # 设置默认特征名称
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(in_features)]

    # 设置默认输出名称
    if output_names is None:
        output_names = [f'Output {j}' for j in range(out_features)]

    for j in range(out_features):
        ax = axes[j]
        total_output = torch.zeros(n_points)
        feature_contributions = []

        for i in range(in_features):

            inputs = torch.zeros(n_points, in_features)
            inputs[:, i] = x


            base_val = kan_layer.base_activation(inputs)
            base_weight = kan_layer.base_weight[j, i]
            base_contribution = base_weight * base_val[:, i]


            spline_basis = kan_layer.b_splines(inputs)
            spline_weight = kan_layer.spline_weight[j, i]

            if kan_layer.enable_standalone_scale_spline:
                spline_scaler = kan_layer.spline_scaler[j, i]
                spline_contribution = spline_scaler * torch.matmul(
                    spline_basis[:, i, :], spline_weight
                )
            else:
                spline_contribution = torch.matmul(
                    spline_basis[:, i, :], spline_weight
                )


            feature_contribution = base_contribution + spline_contribution
            feature_contributions.append(feature_contribution.detach().numpy())

            total_output += feature_contribution


        for i, contrib in enumerate(feature_contributions):
            ax.plot(x.numpy(), contrib, label=f'{feature_names[i]}', alpha=0.7)


        ax.plot(x.numpy(), total_output.detach().numpy(), 'k--', label='Total Output')

        ax.set_title(f'{output_names[j]}')
        ax.set_xlabel('Input Value')
        ax.set_ylabel('Output Contribution')
        ax.grid(True)

        if j == 0:  # 只在第一个子图显示图例
            ax.legend(fontsize=8, loc='best')

    plt.tight_layout()
    plt.suptitle('KAN First Layer Visualization', fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.show()


def visualize_kan_functions(model, layer_idx=0, feature_idx=0, n_points=100, grid_range=[-1, 1]):

    kan_layer = model.layers[layer_idx]
    x = torch.linspace(grid_range[0], grid_range[1], n_points)


    inputs = torch.zeros(n_points, kan_layer.in_features)
    inputs[:, feature_idx] = x

    # 计算基函数部分
    base_val = kan_layer.base_activation(inputs)
    base_vals = base_val[:, feature_idx].detach().numpy()

    # 计算样条函数部分
    spline_basis = kan_layer.b_splines(inputs)
    spline_basis = spline_basis[:, feature_idx, :].detach().numpy()

    # 准备画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 绘制基函数和样条基函数
    ax1.plot(x.numpy(), base_vals, 'b-', label='Base Function (SiLU)')
    ax1.set_title('Base Activation Function')
    ax1.set_xlabel('Input Value')
    ax1.set_ylabel('Activation Output')
    ax1.grid(True)
    ax1.legend()

    for i in range(spline_basis.shape[1]):
        ax2.plot(x.numpy(), spline_basis[:, i], label=f'B-spline {i + 1}')

    ax2.set_title('B-spline Basis Functions')
    ax2.set_xlabel('Input Value')
    ax2.set_ylabel('Basis Value')
    ax2.grid(True)
    ax2.legend()

    plt.suptitle(f'Function Components for Feature {feature_idx} in Layer {layer_idx}', fontsize=16)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))

    for j in range(kan_layer.out_features):
        base_weight = kan_layer.base_weight[j, feature_idx].item()
        base_contribution = base_weight * base_vals

        spline_weight = kan_layer.spline_weight[j, feature_idx].detach().numpy()
        spline_contribution = np.matmul(spline_basis, spline_weight)

        if kan_layer.enable_standalone_scale_spline:
            spline_scaler = kan_layer.spline_scaler[j, feature_idx].item()
            spline_contribution *= spline_scaler

        total_contribution = base_contribution + spline_contribution
        ax.plot(x.numpy(), total_contribution, label=f'Output {j}')

    ax.set_title(f'Output Contributions for Feature {feature_idx} in Layer {layer_idx}')
    ax.set_xlabel('Input Value')
    ax.set_ylabel('Output Contribution')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

from model.kan import KAN
model = KAN([4 * 64, 64, 2])
modelpath = './Model/best_model.pth'
cheakPoint = torch.load(modelpath)
model.load_state_dict(cheakPoint)
model.eval()

# 可视化第一层
#visualize_kan_first_layer(
     #model,
    #feature_names=[f'Feat_{i}' for i in range(256)],  # 输入特征名称
     #output_names=[f'Neuron_{j}' for j in range(64)]  # 输出神经元名称
#)

# 可视化特定层和特征的函数组件
visualize_kan_functions(
    model,
    layer_idx=1,  # 第一层
    feature_idx=1,  # 第10个输入特征
    grid_range=[-1, 1]
)