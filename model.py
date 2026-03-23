import torch
import torch.nn as nn


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CustomLSTM, self).__init__()

        # 1. 优化 LSTM 定义
        # num_layers=2: 2层 LSTM 通常就够用了，既能学到复杂动作，又不容易过拟合
        # dropout=0.3: 在 LSTM 层之间丢弃 30% 的连接，防止死记硬背
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.3)

        # 2. 简化全连接层 (从 6 层砍到 2 层)
        # 结构：Hidden -> 64 -> Output
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # ⚠️ 关键：在全连接层也加 Dropout
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)

        # LSTM 输出
        # out shape: (batch_size, sequence_length, hidden_size)
        # _ (hidden_state, cell_state) 我们不需要，用 _ 忽略
        out, _ = self.lstm(x)

        # 取最后一个时间步 (Last Time Step)
        # 因为我们是看完整个视频动作后才分类，所以只取最后一步的输出
        out = out[:, -1, :]

        # 放入分类器
        out = self.classifier(out)
        return out