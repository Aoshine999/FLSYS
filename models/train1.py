import torch
import os
import wandb
import torch.optim as optim
import torchvision # <--- 已添加
import torchvision.transforms as transforms # <--- 已添加特定导入
from torch.nn import GroupNorm
from torchvision.models import resnet18 # <--- 如未使用，已注释掉
from torch.utils.data import DataLoader, Subset, random_split # <--- 已添加 Subset, random_split
# from datasets import load_dataset, DatasetDict # <--- 已移除 HuggingFace datasets
from tqdm import tqdm
from models.MobileNetV3 import get_mobilenet_v3_small_model # 假设此路径正确
import numpy as np # <--- 已添加

# 假设 get_mobilenet_v3_small_model 在指定路径中已正确定义

# 如果您之后可能使用resnet18，请保留此函数，否则可以移除
def getresnet18_model(num_classes = 10):
    model = resnet18(
        norm_layer=lambda x: GroupNorm(2, x), num_classes=num_classes
    )
    
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    return model

# 修改后的 Collate Function 以处理 torchvision 数据集返回的 (image, label) 元组
def pytorch_collate_fn(batch):
    """
    将一批 (image, label) 元组整理成训练循环期望的字典格式。
    """
    try:
        # batch 是一个包含 (image_tensor, label_integer) 元组的列表
        pixel_values = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        # print("Problematic batch content:", batch) # 取消注释以进行调试
        raise


def load_data(batch_size = 32, image_size = 32, val_ratio = 0.2, data_root='./data', seed=42):
    """
    使用 torchvision 加载 CIFAR-10 数据集。
    Args:
        batch_size (int): 每个批次的大小。
        image_size (int): 图像的目标尺寸 (CIFAR-10 通常是 32)。
        val_ratio (float): 从训练集中划分多少比例作为验证集。
        data_root (str): 下载/存储 CIFAR-10 数据集的根目录。
        seed (int): 用于可复现分割的随机种子。
    Returns:
        tuple: 包含 train_loader, val_loader, test_loader, class_names 的元组。
    """
    # 使用标准的 CIFAR-10 均值和标准差进行归一化
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616) # 您代码中原始的标准差
    # cifar10_std = (0.2023, 0.1994, 0.2010) # 备选标准差

    # 定义转换 - 如果不使用 32x32，请调整 crop/padding
    # 训练转换（带数据增强）
    train_transform = transforms.Compose([
        transforms.RandomCrop(image_size, padding=4), # 填充后随机裁剪
        transforms.RandomHorizontalFlip(),            # 随机水平翻转
        transforms.ToTensor(),                        # 转换为张量
        transforms.Normalize(cifar10_mean, cifar10_std), # 归一化
    ])

    # 测试/验证转换（不带数据增强）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    # --- 加载数据集 ---
    # 加载完整的训练数据集 (应用训练转换)
    train_dataset_full_train_tf = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform # 应用训练转换
    )
    # 再次加载训练数据集，但应用测试转换，用于创建验证集
    train_dataset_full_test_tf = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=False, # 已经下载过了
        transform=test_transform # 应用测试转换 (用于验证)
    )
    # 加载测试数据集 (应用测试转换)
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform # 应用测试转换
    )

    # --- 分割训练集和验证集 ---
    num_train = len(train_dataset_full_train_tf)
    num_val = int(np.floor(val_ratio * num_train))
    num_train_split = num_train - num_val

    # 设置随机种子以保证分割的可复现性
    generator = torch.Generator().manual_seed(seed)
    # 使用相同的生成器分割索引
    indices = list(range(num_train))
    train_subset_indices, val_subset_indices = random_split(
        indices, [num_train_split, num_val], generator=generator
    )

    # 创建训练子集 (使用带 train_transform 的数据集)
    train_subset = Subset(train_dataset_full_train_tf, train_subset_indices)

    # 创建验证子集 (使用带 test_transform 的数据集 和 相同的验证索引)
    val_subset = Subset(train_dataset_full_test_tf, val_subset_indices)

    # --- 创建 DataLoaders ---
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True, # 打乱训练数据
        collate_fn=pytorch_collate_fn, # 使用自定义的 collate 函数
        num_workers=4,
        pin_memory=True # 开启内存锁定，可能加速数据传输到 GPU
    )

    val_loader = DataLoader(
        val_subset, # 使用带测试转换的验证子集
        batch_size=batch_size,
        shuffle=False, # 验证集不需要打乱
        collate_fn=pytorch_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, # 测试集不需要打乱
        collate_fn=pytorch_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # 从数据集对象获取类别名称
    class_names = train_dataset_full_train_tf.classes # 或 test_dataset.classes

    print(f"数据加载完成: 训练集={len(train_subset)}, 验证集={len(val_subset)}, 测试集={len(test_dataset)}")
    print(f"类别名称: {class_names}")

    return train_loader, val_loader, test_loader, class_names


def train(config):
    # 初始化 WandB
    wandb.init(project="MobileNetV3_small_CIFAR10_central", config=config) # 更新了项目名称

    # 加载数据 (使用新的基于 PyTorch 的函数)
    train_loader, val_loader, test_loader, class_names = load_data(
        batch_size=config["batch_size"],
        image_size=config["image_size"], # 传递 image_size 如果转换需要它
        val_ratio=config.get("val_ratio", 0.2),
        seed=config.get("seed", 42) # 传递种子以实现可复现分割
    )

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    num_classes = len(class_names)
    print(f"类别数量: {num_classes}")

    # 获取模型实例
    model = get_mobilenet_v3_small_model(num_classes=num_classes, pretrained=config["pretrained"])
    # model = getresnet18_model(num_classes=num_classes)
    model.to(device)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config.get("momentum", 0.9), # 使用配置中的 momentum 或默认值 0.9
        weight_decay=config["weight_decay"]
    )

    # # 定义学习率调度器
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='max',       # 当验证准确率不再提升时降低 LR
    #     factor=0.1,       # 将 LR 乘以 0.1
    #     patience=5,       # 在降低 LR 前等待 5 个没有提升的 epoch
    #     verbose=True      # 打印学习率变化信息
    # )

    best_acc = 0.0  # 初始化最佳验证准确率
    # 定义模型保存路径
    path = os.path.join("global_model_param", "central_model")
    os.makedirs(path, exist_ok=True) # 确保目录存在

    # --- 开始训练循环 ---
    for epoch in range(config["num_epochs"]):
        model.train() # 设置模型为训练模式
        training_loss = 0.0
        correct = 0
        total = 0
        # 使用 enumerate 获取批次索引，方便可能的日志记录
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} Train")):
            # 获取输入和标签，并移动到指定设备
            inputs = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # 清零梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            # --- 统计训练指标 ---
            training_loss += loss.item() * inputs.size(0) # 累加批次损失 (乘以批次大小)
            _, preds = torch.max(outputs.data, 1) # 获取预测结果
            total += labels.size(0) # 累加样本总数
            correct += (preds == labels).sum().item() # 累加预测正确的样本数

            # 可以选择性地记录批次指标 (可能会产生大量日志)
            # if i % 100 == 0: # 每 100 个批次记录一次
            #     wandb.log({
            #         "batch_loss": loss.item(),
            #         "batch_acc": (preds == labels).float().mean().item()
            #     })

        # --- 计算整个 epoch 的训练指标 ---
        epoch_loss = training_loss / total
        epoch_acc = correct / total

        # --- 验证阶段 ---
        avg_val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # --- 更新学习率调度器 (基于验证准确率) ---
        #scheduler.step(val_acc) # <--- 将验证准确率传递给调度器

        # --- 记录 epoch 指标到 WandB ---
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_acc": epoch_acc, # 使用 train_acc 更清晰
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            #"lr": optimizer.param_groups[0]['lr'] # 记录当前学习率
        }, step=epoch + 1) # 使用 epoch 编号作为 step

        # --- 保存最佳模型 (基于验证准确率) ---
        if val_acc > best_acc:
            print(f"验证准确率提升 ({best_acc:.4f} --> {val_acc:.4f}). 保存模型...")
            best_acc = val_acc
            #best_model_path = os.path.join(path, f"best_model_res18.pth")
            best_model_path = os.path.join(path, f"best_model_mbv3sml.pth")
            torch.save(model.state_dict(), best_model_path)
            # 如果需要，可以将最佳模型保存为 WandB 的 artifact
            # wandb.save(best_model_path)

    # --- 训练结束，进行最终测试 ---
    print("\n训练完成。加载最佳模型进行最终测试...")
    #best_model_path = os.path.join(path, f"best_model_res18.pth")
    best_model_path = os.path.join(path, f"best_model_mbv3sml.pth")
    if os.path.exists(best_model_path):
        # 加载验证集上表现最好的模型权重
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("警告：未找到最佳模型检查点。将使用最后一个 epoch 的模型进行测试。")

    # 在测试集上评估模型性能
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

    # --- 记录最终测试结果到 WandB ---
    wandb.log({
        "final_test_loss": test_loss,
        "final_test_acc": test_acc,
        "best_val_acc": best_acc # 记录训练过程中达到的最佳验证准确率
    })

    # --- 打印最终结果 ---
    print(f"\n训练期间最佳验证准确率: {best_acc:.2%}")
    print(f"最终测试准确率 (使用最佳模型): {test_acc:.2%}")

    # 结束 WandB 运行
    wandb.finish()


def evaluate_model(model, dataloader, criterion, device):
    """ 在给定数据集上评估模型 """
    model.eval() # 设置模型为评估模式
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad(): # 在评估阶段不计算梯度
        # 为 tqdm 添加描述信息
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0) # 累加批次损失
            _, preds = torch.max(outputs, 1) # 获取预测类别
            total += labels.size(0) # 累加样本总数
            correct += (preds == labels).sum().item() # 累加正确预测数

    avg_loss = total_loss / total # 计算平均损失
    accuracy = correct / total # 计算准确率
    return avg_loss, accuracy


# 运行配置
if __name__ == "__main__":
    config = {
        # "dataset_name": "CIFAR10", # 数据集名称 (主要用于标识)
        "image_size": 32,         # CIFAR-10 的标准图像尺寸
        "batch_size": 64,         # 批处理大小
        "num_epochs": 80,         # 训练的总轮数
        "lr": 0.01,               # 初始学习率
        "momentum": 0.9,          # SGD 优化器的动量因子
        "weight_decay": 1e-4,     # 权重衰减 (L2 正则化)
        "val_ratio": 0.2,         # 验证集占训练集的比例 (例如 0.2 表示 20%)
        "pretrained": False,      # 是否使用预训练权重 (CIFAR-10 通常设为 False)
        "seed": 42                # 固定随机种子以保证实验可复现性
    }
    # 启动训练过程
    #train(config)

    # 调试路径的打印输出 (通常在最终脚本中不需要)
    # print(os.getcwd())
    # path = os.path.join("global_model_param","central_model")
    # print(os.chdir(path)) # 全局改变当前目录需要小心
    # print(os.getcwd())