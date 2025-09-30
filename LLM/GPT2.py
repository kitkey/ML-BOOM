import math
import torch
import torch.nn as nn
import torch.utils.data
from typing import Optional, Tuple, List


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(torch.sqrt(torch.Tensor([2 / math.pi])) * (x + 0.044715 * x.pow(3))))


class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class PositionalEmbeddings(nn.Module):
    def __init__(self, max_seq_len: int, emb_size: int):
        super().__init__()
        self.pos_embeddings = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=emb_size)

    def forward(self, seq_len: int, start_pos: int = 0) -> torch.Tensor:
        return self.pos_embeddings.weight[start_pos:start_pos + seq_len]


class HeadAttention(nn.Module):
    def __init__(self, emb_size: int, head_size: int, max_seq_len: int):
        super().__init__()
        self.K = nn.Linear(emb_size, head_size)
        self.Q = nn.Linear(emb_size, head_size)
        self.V = nn.Linear(emb_size, head_size)
        self.triangle_mask = torch.tril(torch.ones(size=(max_seq_len, max_seq_len), dtype=torch.uint8))
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len

    def forward(
            self,
            x: torch.Tensor,
            use_cache: bool = True,
            cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:

        key, query, value = self.K(x), self.Q(x), self.V(x)

        if cache is not None:
            key = torch.cat([cache[0], key], dim=1)  # cache_key
            value = torch.cat([cache[1], value], dim=1)  # cache_value

        attention_matrix = query @ key.transpose(1, 2) / math.sqrt(self.head_size)

        masked_attention_matrix = attention_matrix

        if cache is None:
            masked_attention_matrix = attention_matrix.masked_fill(self.triangle_mask[:x.shape[1], :x.shape[1]] == 0,
                                                                   float('-inf'))

        softmax_attention = nn.functional.softmax(masked_attention_matrix, dim=-1)

        output = softmax_attention @ value
        if cache is not None or use_cache:
            return (output, (key, value))
        else:
            return (output, None)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            num_heads: int,
            emb_size: int,
            head_size: int,
            max_seq_len: int,
            dropout: float = 0.1
    ):
        super().__init__()
        self.module_list = nn.ModuleList(
            [HeadAttention(emb_size, head_size, max_seq_len) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(head_size * num_heads, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, use_cache: bool = True, cache: Optional[List[Tuple[torch.Tensor]]] = None):
        layer_outputs = []
        cache_new = []

        for i, layer in enumerate(self.module_list):
            head_cache = cache[i] if cache is not None else None
            y_i, c_i = layer(x, use_cache=use_cache, cache=head_cache)
            layer_outputs.append(y_i)
            if use_cache:
                cache_new.append(c_i)

        concat = torch.cat(layer_outputs, dim=-1)

        out = self.linear(concat)
        drop_output = self.dropout(out)
        if use_cache:
            return drop_output, cache_new
        else:
            return drop_output, None


class FeedForward(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            GELU(),
            nn.Linear(4 * emb_size, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class Decoder(nn.Module):
    def __init__(
            self,
            num_heads: int,
            emb_size: int,
            head_size: int,
            max_seq_len: int,
            dropout: float = 0.1
    ):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads, emb_size, head_size, max_seq_len, dropout)
        self.ff = FeedForward(emb_size, dropout)
        self.first_ln = nn.LayerNorm(emb_size)
        self.second_ln = nn.LayerNorm(emb_size)

    def forward(
            self, x: torch.Tensor,
            use_cache: bool = True,
            cache: Optional[List[Tuple[torch.Tensor]]] = None):
        ln_first = self.first_ln(x)
        mha_out, attention_cache = self.mha(ln_first, use_cache=use_cache, cache=cache)
        mha_res = mha_out + x
        ln_second = self.second_ln(mha_res)
        ff_out = self.ff(ln_second)
        ff_res = ff_out + mha_res
        return ff_res, attention_cache


class GetData(torch.utils.data.Dataset):
    def __init__(self, data: str, seq_len: int, device: str):
        super().__init__()

        self.data = data
        self.seq_len = seq_len
        self.device = device

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx: int):
        return (torch.tensor(self.data[idx:idx + self.seq_len]),
                torch.tensor(self.data[idx + 1:idx + self.seq_len + 1])
                )


class GPT2(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            max_seq_len: int,
            emb_size: int,
            num_heads: int,
            head_size: int,
            num_layers: int,
            device: str = 'cpu',
            dropout: float = 0.1
    ):
        super().__init__()
        self.token_emb = TokenEmbeddings(vocab_size, emb_size).to(device)
        self.pos_emb = PositionalEmbeddings(max_seq_len, emb_size).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.decoder_blocks = nn.ModuleList(
            [Decoder(num_heads, emb_size, head_size, max_seq_len, dropout) for _ in range(num_layers)]
        ).to(device)
        self.decoder_norm_layer = nn.LayerNorm(emb_size).to(device)
        self.proj = nn.Linear(emb_size, vocab_size).to(device)
        self.device = device
        self.val_losses = []
        self.train_losses = []

    def forward(self, x: torch.Tensor, use_cache: bool = True, cache: Optional[List[List[Tuple[torch.Tensor]]]] = None):
        token_emb = self.token_emb(x)

        if use_cache:
            if cache is None:
                pos_emb = self.pos_emb(x.size(1))

            else:
                key, value = cache[0][-1]
                start_pos = key.shape[-2]

                pos_emb = self.pos_emb(seq_len=x.size(1), start_pos=start_pos)
        else:
            pos_emb = self.pos_emb(x.size(1))
        unified_emb = token_emb + pos_emb
        drop_out = self.dropout(unified_emb)
        decoder_out = drop_out
        decoder_output_hashes = []
        for i, decoder_block in enumerate(self.decoder_blocks):
            decoder_cache = None
            if cache is not None:
                decoder_cache = cache[i]
            decoder_out, cache_output = decoder_block(decoder_out, use_cache=use_cache, cache=decoder_cache)

            if cache_output is not None:
                decoder_output_hashes.append(cache_output)

        decoder_norm = self.decoder_norm_layer(decoder_out)
        output = self.proj(decoder_norm)

        return (output, decoder_output_hashes) if use_cache else (output, None)

    def generate(
            self,
            x: torch.Tensor,
            max_new_tokens: int,
            do_sample: bool = False,
            temperature: float = 1.0,
            top_k: int = None,
            top_p: int = None,
            use_cache: bool = True
    ):

        for iteration in range(max_new_tokens):
            # x_restr = x[:, -self.pos_emb.pos_embeddings.num_embeddings:]
            if use_cache:
                if iteration == 0:
                    logits, cache = self.forward(x, use_cache=use_cache)
                else:
                    logits, cache = self.forward(x[:, -1].unsqueeze(1), use_cache=use_cache, cache=cache)
            else:
                x_restr = x[:, -self.pos_emb.pos_embeddings.num_embeddings:]
                logits = self.forward(x_restr, use_cache=use_cache)[0] / temperature
            logits = logits / temperature

            if not do_sample:
                next_token = torch.argmax(torch.softmax(logits[:, -1, :], dim=-1), dim=-1, keepdim=True)
            else:
                probas = logits[:, -1, :]

                if top_k is not None:
                    sort_val, sort_ind = torch.sort(probas, descending=True)
                    inf_arr = torch.empty(size=probas.shape).fill_(float("-Inf"))

                    # probas.scatter_(dim=-1, index=sort_ind[:, top_k:],  src=inf_arr)
                    probas.scatter_(dim=-1, index=sort_ind[:, top_k:],
                                    src=torch.full_like(sort_ind[:, top_k:], float('-inf'), dtype=torch.float))

                if top_p is not None:
                    real_probas = torch.softmax(probas, dim=-1)
                    sort_val, sort_ind = torch.sort(real_probas, descending=True)
                    cumsum_arr = torch.cumsum(sort_val, dim=-1)
                    mask = cumsum_arr < top_p
                    mask[:, 0] = 1
                    filtered_logits = torch.full_like(probas, float('-Inf'))
                    probas = filtered_logits.scatter_(
                        dim=-1,
                        index=sort_ind,
                        src=torch.where(mask, probas.gather(-1, sort_ind), torch.full_like(sort_val, float('-Inf')))
                    )
                next_token = torch.multinomial(torch.softmax(probas, dim=-1), num_samples=1)
            x = torch.cat([x, next_token], dim=1)
        return x

    def fit(
            self,
            train_loader: torch.utils.data.DataLoader,
            valid_loader: torch.utils.data.DataLoader,
            num_epoch: int,
            learning_rate: float) -> None:
        self.to(self.device)

        optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss = nn.CrossEntropyLoss()

        for _ in range(num_epoch):
            self.train()
            for batch in train_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                logits, _ = self.forward(inputs, use_cache=False)
                logits_shape = logits.shape
                logits = logits.view(logits_shape[0] * logits_shape[1], logits_shape[2])
                targets = targets.flatten()

                loss_value = loss(logits, targets)

                self.train_losses.append(loss_value.item())
                optim.zero_grad()
                loss_value.backward()
                optim.step()

            self.eval()
            with torch.no_grad():
                for batch in valid_loader:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    logits, _ = self(inputs, use_cache=False)
                    logits_shape = logits.shape
                    logits = logits.view(logits_shape[0] * logits_shape[1], logits_shape[2])
                    targets = targets.flatten()
                    loss_value = loss(logits, targets)
                    self.val_losses.append(loss_value.item())








