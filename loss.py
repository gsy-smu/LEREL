import torch
import torch.nn as nn
import torch.nn.functional as F
class CustomLoss(nn.Module):
    def __init__(self, alpha=0.0, beta=5.0):
        super(CustomLoss, self).__init__()
        self.alpha = alpha  # 惩罚项的权重系数
        self.beta = beta    # 目标类别差距的权重系数
        self.closs=nn.CrossEntropyLoss()

    def forward(self, output, target):
        """
        计算预测编码和标签编码之间的加权平方差之和
        :param output: 模型预测的独热编码，形状为 (batch_size, num_classes)
        :param target: 真实标签的独热编码，形状为 (batch_size, num_classes)
        :return: 损失值
        """

        # 计算交叉熵损失
        cross_entropy = self.closs(output,target)  # 按照类别维度求和，得到每个样本的交叉熵损失

        # 计算这一批次的正确率
        _, predicted = torch.max(output, 1)
        target_indices = torch.argmax(target, dim=1)
        correct = (predicted == target_indices).sum().item()
        accuracy = correct / len(predicted)
        penalty = 2.0 - accuracy

        # 总损失 = 交叉熵损失 + 惩罚项
        total_loss = cross_entropy * penalty

        # 返回平均损失
        return torch.mean(total_loss)