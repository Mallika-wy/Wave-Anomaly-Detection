from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TimeDistributed(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps = x.shape[:2]
        y = self.module(x.reshape(batch * steps, *x.shape[2:]))
        return y.reshape(batch, steps, *y.shape[1:])


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        gates = self.gates(torch.cat([x, h_prev], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(
        self,
        batch_size: int,
        spatial_size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = spatial_size
        zeros = torch.zeros(batch_size, self.hidden_channels, height, width, device=device, dtype=dtype)
        return zeros, zeros.clone()


class ConvLSTM(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, steps, _, height, width = x.shape
        h, c = self.cell.init_state(batch, (height, width), x.device, x.dtype)
        outputs = []
        for step in range(steps):
            h, c = self.cell(x[:, step], (h, c))
            outputs.append(h)
        return torch.stack(outputs, dim=1), (h, c)


class EncoderBranch(nn.Module):
    def __init__(self, in_channels: int, channels: list[int], dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        self.temporal = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        current = in_channels
        for idx, out_channels in enumerate(channels):
            self.blocks.append(TimeDistributed(ConvBlock(current, out_channels, dropout=dropout)))
            self.temporal.append(ConvLSTM(out_channels, out_channels))
            if idx < len(channels) - 1:
                self.downsamples.append(TimeDistributed(nn.MaxPool2d(kernel_size=2, stride=2)))
            current = out_channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        skips: list[torch.Tensor] = []
        seq = x
        for idx, block in enumerate(self.blocks):
            seq = block(seq)
            seq, (h, _) = self.temporal[idx](seq)
            skips.append(h)
            if idx < len(self.downsamples):
                seq = self.downsamples[idx](seq)
        return skips


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.block = ConvBlock(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class DualBranchConvLSTMUNet(nn.Module):
    def __init__(
        self,
        wind_channels: int = 3,
        wave_channels: int = 4,
        base_channels: int = 32,
        depth: int = 3,
        fusion_type: str = "concat",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if fusion_type != "concat":
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
        channels = [base_channels * (2**idx) for idx in range(depth)]

        self.wind_encoder = EncoderBranch(wind_channels, channels, dropout=dropout)
        self.wave_encoder = EncoderBranch(wave_channels, channels, dropout=dropout)
        self.fusion_layers = nn.ModuleList(
            [nn.Conv2d(ch * 2, ch, kernel_size=1) for ch in channels]
        )
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(channels[idx] + channels[idx - 1], channels[idx - 1], dropout) for idx in range(depth - 1, 0, -1)]
        )
        self.head = nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(self, x_wind: torch.Tensor, x_wave: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        wind_skips = self.wind_encoder(x_wind)
        wave_skips = self.wave_encoder(x_wave)
        fused = [
            fusion(torch.cat([wind_skip, wave_skip], dim=1))
            for fusion, wind_skip, wave_skip in zip(self.fusion_layers, wind_skips, wave_skips)
        ]
        x = fused[-1]
        for block, skip in zip(self.decoder_blocks, reversed(fused[:-1])):
            x = block(x, skip)
        logits = self.head(x)
        if return_logits:
            return logits
        return torch.sigmoid(logits)


def build_model(config: dict[str, Any]) -> DualBranchConvLSTMUNet:
    return DualBranchConvLSTMUNet(
        wind_channels=3,
        wave_channels=4,
        base_channels=int(config["model"]["base_channels"]),
        depth=int(config["model"]["depth"]),
        fusion_type=str(config["model"]["fusion_type"]),
        dropout=float(config["model"].get("dropout", 0.1)),
    )
