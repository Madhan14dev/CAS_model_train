# # models/quartznet_encoder.py

# import torch.nn as nn
# from .quartznet_blocks import QuartzBlock


# class QuartzNetEncoder(nn.Module):
#     def __init__(self, n_mels=80):
#         super().__init__()

#         self.C1 = nn.Sequential(
#             nn.Conv1d(n_mels, 256, kernel_size=33, stride=2, padding=16),
#             nn.BatchNorm1d(256),
#             nn.ReLU()
#         )

#         self.B1 = QuartzBlock(256, 256, kernel_size=33, R=5)
#         self.B2 = QuartzBlock(256, 256, kernel_size=39, R=5)
#         self.B3 = QuartzBlock(256, 512, kernel_size=51, R=5)
#         self.B4 = QuartzBlock(512, 512, kernel_size=63, R=5)
#         self.B5 = QuartzBlock(512, 512, kernel_size=75, R=5)

#         self.C2 = nn.Sequential(
#             nn.Conv1d(512, 512, kernel_size=87, padding=43),
#             nn.BatchNorm1d(512),
#             nn.ReLU()
#         )

#         self.C3 = nn.Sequential(
#             nn.Conv1d(512, 1024, kernel_size=1),
#             nn.BatchNorm1d(1024),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         # x: (B, n_mels, T)

#         x = self.C1(x)
#         x = self.B1(x)
#         x = self.B2(x)
#         x = self.B3(x)
#         x = self.B4(x)
#         x = self.B5(x)
#         x = self.C2(x)
#         x = self.C3(x)

#         # return (B, T, C)
#         return x.transpose(1, 2)

# models/quartznet_encoder.py

import torch.nn as nn
from .quartznet_blocks import QuartzBlock


class QuartzNetEncoder(nn.Module):
    def __init__(self, n_mels=80, repeat=1):
        """
        repeat = 1 → QuartzNet 5x5
        repeat = 2 → QuartzNet 10x5
        repeat = 3 → QuartzNet 15x5
        """
        super().__init__()

        self.C1 = nn.Sequential(
            nn.Conv1d(n_mels, 256, kernel_size=33, stride=2, padding=16),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # Block configs (same as paper)
        block_cfg = [
            (256, 256, 33),  # B1
            (256, 256, 39),  # B2
            (256, 512, 51),  # B3
            (512, 512, 63),  # B4
            (512, 512, 75),  # B5
        ]

        blocks = []
        for in_ch, out_ch, k in block_cfg:
            for _ in range(repeat):
                blocks.append(
                    QuartzBlock(in_ch, out_ch, kernel_size=k, R=5)
                )
                in_ch = out_ch

        self.blocks = nn.Sequential(*blocks)

        self.C2 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=87, padding=43),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.C3 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.C1(x)
        x = self.blocks(x)
        x = self.C2(x)
        x = self.C3(x)

        return x.transpose(1, 2)