import torch
import os
import wandb
import torch.optim as optim
from torchvision import transforms
from torch.nn import GroupNorm
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
from flsys.models.MobileNetV3 import get_mobilenet_v3_small_model
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)



def getresnet18_model(num_classes = 10):
    model = resnet18(
        norm_layer=lambda x: GroupNorm(2, x), num_classes=num_classes
    )

    return model
# 将 collate_fn 定义移到这里 (顶层)
def top_level_collate_fn(batch):
    # 确保 batch 中的元素确实是字典，并且有 'pixel_values' 和 'label'键
    # 如果 set_format 工作正常，这里应该没问题
    try:
        pixel_values = torch.stack([x["pixel_values"] for x in batch])
        labels = torch.tensor([x["label"] for x in batch])
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        # 可以打印 batch 内容帮助调试
        # print("Problematic batch content:", batch)
        raise



def load_data(dataset_name = "uoft-cs",batch_size = 32,image_size = 224,val_ratio = 0.2):
    """
    使用datasets库加载数据集
    dataset_name: 可以是Hugging Face Hub上的数据集名称,或本地路径
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),  # 调整图像大小
        #transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # 归一化 
    ])

    train_transform = Compose(
        [RandomCrop(24), RandomHorizontalFlip(), ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )
    test_transform = Compose([CenterCrop(24), ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    dataset = load_dataset(dataset_name)  # 加载训练集

    test_val = dataset["train"].train_test_split(test_size=val_ratio, seed=42)  # 划分验证集和测试集


    dataset = DatasetDict({
        "train": test_val["train"],  # 训练集
        "val": test_val["test"],  # 测试集
        "test": dataset["test"]  # 验证集
    })




        # 定义转换应用函数
    def apply_transforms(examples):
        examples["pixel_values"] = [
            transform(image.convert("RGB")) 
            for image in examples["img"]
        ]
        return examples
        

        # 应用数据转换
    dataset = dataset.map(
        apply_transforms,
        batched=True,
        batch_size=32,
        remove_columns=["img"],  # 移除原始图像列
        desc="Applying transforms"
    )


    dataset.set_format("torch", columns=["pixel_values", "label"])  # 设置数据集格式为PyTorch


        # 创建DataLoader
    def collate_fn(batch):
        return {
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "labels": torch.tensor([x["label"] for x in batch])
        }

    train_loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=top_level_collate_fn,
        num_workers=4
    )

    val_loader = DataLoader(
        dataset["val"],
        batch_size=batch_size,
        collate_fn=top_level_collate_fn,
        num_workers=4
    )

    test_loader = DataLoader(
        dataset["test"],
        batch_size=batch_size,
        collate_fn=top_level_collate_fn,
        num_workers=4
    )
    class_names = dataset["train"].features["label"].names  # 获取类别名称

    return train_loader, val_loader, test_loader, class_names


def train(config):
    wandb.init(project="MobileNetV3_samll centralization train", config=config)

    # 加载数据
    train_loader,val_loader, test_loader, class_names = load_data(
        config["dataset_name"], config["batch_size"], config["image_size"],
        val_ratio=config.get("val_ratio", 0.2)
    )

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_mobilenet_v3_small_model(num_classes=len(class_names), pretrained=config["pretrained"])

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],momentum=0.9,weight_decay=1e-5)


    # 学习率调度
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.1,
        patience=3,
        verbose=True
    )
   
    best_acc = 0.0  # 初始化最佳准确率
    # 训练模型
    for epoch in range(config["num_epochs"]):
        model.train()
        training_loss = 0.0
        correct = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
            inputs = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #统计指标
            training_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs.data, 1)
            correct += (preds == labels).sum().item()

            # 记录batch指标
            wandb.log({
                "batch_loss": loss.item(),
                "batch_acc": (preds == labels).float().mean().item()
            })

        # 记录epoch指标
        epoch_loss = training_loss / len(train_loader.dataset)
        epoch_acc = correct / len(train_loader.dataset)

        # 验证阶段
        avg_val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        # 学习率调度
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode='max',
        #     factor=0.1,
        #     patience=3,
        #     verbose=True
        # )

        
        # 记录epoch指标
        wandb.log({
            "train_loss": epoch_loss,  # 保持一致，或者根据需要调整
            "epoch_acc": epoch_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            #"lr": optimizer.param_groups[0]['lr']
        },step= epoch + 1)

        
        path = os.path.join("global_model_param","central_model")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(path,f"best_model.pth"))
            wandb.save("best_model.pth")

    # 最终测试
    model.load_state_dict(torch.load(os.path.join(path,f"best_model.pth")))
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

    # 根据验证准确率更新调度器
    #scheduler.step(val_acc) # <--- 添加这一行
    # 记录最终结果
    wandb.log({
        "final_test_loss": test_loss,
        "final_test_acc": test_acc
    })

    # 打印结果对比
    print(f"\nBest Validation Accuracy: {best_acc:.2%}")
    print(f"Final Test Accuracy: {test_acc:.2%}")
    
    # 保存完整报告
    wandb.finish()

def evaluate_model(model, dataloader,criterion, device):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    return avg_loss, accuracy            

# 运行配置
if __name__ == "__main__":
    config = {
        "dataset_name": "uoft-cs/cifar10",
        "image_size": 32,
        "batch_size": 64,
        "num_epochs": 50,
        "lr": 0.01,
        "weight_decay": 1e-4,
        "val_ratio": 0.3,  
        "pretrained": True,
        "seed": 42        # 固定随机种子
    }
    train(config)

    # print(os.getcwd())
    # path = os.path.join("global_model_param","central_model")
    # print(os.chdir(path))
    # print(os.getcwd())

