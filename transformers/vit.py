from __future__ import annotations
from torch.nn import MultiheadAttention, Linear, LayerNorm, Dropout, GELU
import torch as th
from utils import patchify


class ViT(th.nn.Module):
    def __init__(
        self,
        image_shape: tuple[int, int],
        in_channels: int,
        patch_size: int,
        embed_dim: int,
        n_heads: int,
        n_layers: int,
        n_classes: int,
        linear_dims: int,
    ):
        super().__init__()
        self.embedder = PatchEmbedder(
            (in_channels, *image_shape), patch_size, embed_dim
        )
        self.drop = th.nn.Dropout(p=0.2)

        n_patches = (
            (image_shape[0] // patch_size)
            * (image_shape[1] // patch_size)
        )
        in_shape = (n_patches + 1, embed_dim)  # 1 extra patch for class embeding
        self.transformers_block = th.nn.Sequential(
            *[
                Transformer(in_shape, embed_dim, n_heads, linear_dims)
                for _ in range(n_layers)
            ]
        )

        self.classifier = th.nn.Sequential(
            th.nn.LayerNorm(embed_dim),
            th.nn.Linear(embed_dim, n_classes),
        )

    def forward(self, X: th.Tensor) -> th.Tensor:
        X = self.embedder(X)
        X = self.drop(X)
        X = self.transformers_block(X)
        classs_embedding = X[:, 0]
        return self.classifier(classs_embedding)


class PatchEmbedder(th.nn.Module):
    def __init__(
        self,
        image_shape: tuple,
        patch_size: int,
        emb_dim: int,
    ):
        super().__init__()
        self._patch_size = patch_size
        channels, height, width = image_shape
        n_patches = (height // patch_size) * (width // patch_size)

        self.linear = th.nn.Linear(channels * patch_size**2, emb_dim)
        self.class_token = th.nn.Parameter(th.randn(1, 1, emb_dim) / 10)
        self.pos_embedding = th.nn.Parameter(th.randn(1, n_patches + 1, emb_dim) / 10)

    def forward(self, x):
        x = patchify(x, self._patch_size)
        x = self.linear(x)
        x = self.add_class_embedding(x)
        out = x + self.pos_embedding
        return out

    def add_class_embedding(self, tokens: th.Tensor):
        class_tokens = self.class_token.expand(tokens.shape[0], -1, -1)
        x = th.cat((class_tokens, tokens), dim=1)
        return x


class Transformer(th.nn.Module):
    def __init__(
        self,
        input_shape,
        embed_dim: int,
        n_heads: int,
        linear_dims: int,
    ):
        super().__init__()
        self.norm = th.nn.LayerNorm(input_shape)
        self.attention = th.nn.MultiheadAttention(embed_dim, n_heads)

        self.mlp = th.nn.Sequential(
            th.nn.LayerNorm(embed_dim),
            th.nn.Linear(embed_dim, linear_dims),
            th.nn.GELU(),
            th.nn.Dropout(p=0.2),
            th.nn.Linear(linear_dims, embed_dim),
            th.nn.Dropout(p=0.2),
        )

    def forward(self, X: th.Tensor) -> th.Tensor:
        residual1 = X
        X = self.norm(residual1)
        X, _ = self.attention(X, X, X, need_weights=False)

        residual2 = X + residual1
        return self.mlp(residual2) + residual2

from torchviz import make_dot

if __name__ == "__main__":
    vit = ViT(
        image_shape=(32, 32),
        in_channels=3,
        patch_size=4,
        embed_dim=256,
        n_heads=8,
        n_layers=6,
        n_classes=10,
        linear_dims=512
    )
    sample = th.ones((64, 3, 32, 32))
    out = vit(sample)
    make_dot(out).render("rnn_torchviz", format="png")

