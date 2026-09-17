"""Four independent parameter groups with token-time, not depth-time, supervision."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.func import functional_call
from .layers import TransformerBlock, TransformerConfig, RMSNorm, init_weights
from .recal_model import DriftEstimator
from recal.training.losses import state_distillation_loss


class BlockStack(nn.Module):
    def __init__(self, config, layers):
        super().__init__()
        tc = TransformerConfig(**{k: config[k] for k in
            ('vocab_size', 'context_length', 'hidden_size', 'num_heads', 'ffn_dim')},
            dropout=config.get('dropout', 0.0))
        self.blocks = nn.ModuleList([TransformerBlock(tc) for _ in range(layers)])
        self.norm = RMSNorm(config['hidden_size'])
        self.gradient_checkpointing = config.get('gradient_checkpointing', False)

    def forward(self, state, positions, use_checkpoint=True):
        for block in self.blocks:
            if use_checkpoint and self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                state = checkpoint(block, state, positions, use_reentrant=False)
            else:
                state = block(state, positions)
        return self.norm(state)


class AttentionState(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.stack = BlockStack(config, config['front_layers'])

    def forward(self, ids):
        return self.stack(self.embedding(ids), torch.arange(ids.shape[1], device=ids.device))


class RecurrentCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.input_projection = nn.Linear(config['hidden_size'], config['hidden_size'], bias=False)
        self.input_norm = RMSNorm(config['hidden_size'])
        self.stack = BlockStack(config, config['recurrent_layers'])

    def forward(self, previous, new_ids, positions):
        state = self.input_norm(previous + self.input_projection(self.embedding(new_ids)))
        return self.stack(state, positions)


class StateDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stack = BlockStack(config, config['back_layers'])
        self.head = nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)

    def forward(self, state, positions, use_checkpoint=True):
        return self.head(self.stack(state, positions, use_checkpoint=use_checkpoint))


class FourModelLM(nn.Module):
    def __init__(self, config, stage='early'):
        super().__init__()
        if stage not in ('early', 'late'):
            raise ValueError(stage)
        self.config = dict(config)
        self.stage = stage
        self.attention = AttentionState(config)
        self.core = RecurrentCore(config)
        self.decoder = StateDecoder(config)
        self.drift = DriftEstimator(config['hidden_size'], config.get('drift_hidden_size', 128))
        self.apply(init_weights)
        # Separate embeddings/head avoid accidentally training another module through a tied weight.
        self.teacher = copy.deepcopy(self.attention) if stage == 'early' else None
        if self.teacher is not None:
            self.teacher.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def enter_late(self):
        self.stage = 'late'
        self.teacher = None  # Removes parameters, state_dict entries, and the module's device memory.

    @torch.no_grad()
    def update_teacher(self):
        if self.teacher is None:
            return
        decay = float(self.config.get('ema_decay', 0.9995))
        for t, a in zip(self.teacher.parameters(), self.attention.parameters()):
            t.lerp_(a, 1.0 - decay)
        for t, a in zip(self.teacher.buffers(), self.attention.buffers()):
            t.copy_(a)

    def forward(self, ids, labels=None, *, decoder_source=None, decoder_step=None):
        """At offset s, state j has consumed tokens through j+s; its label is x[j+s+1].

        Vectorized streams share a causal prefix, never future positions. The next token
        is an observed input to R, not a guessed token. No extra Router is trained.
        """
        hops = int(self.config.get('rollout_steps', 2))
        if not 1 <= hops < ids.shape[1] <= self.config['context_length']:
            raise ValueError('Require 1 <= rollout_steps < sequence length <= context_length')
        attention = self.attention(ids)
        width = ids.shape[1] - hops
        previous = attention[:, :width]
        if self.stage == 'early':
            previous = previous.detach()
            with torch.no_grad():
                reference = self.teacher(ids)
        else:
            reference = attention.detach()
        core_states, core_losses, attention_losses, drift_losses = [], [], [], []
        drift_values = []
        for s in range(1, hops + 1):
            positions = torch.arange(s, s + width, device=ids.device)
            previous = self.core(previous, ids[:, s:s + width], positions)
            core_states.append(previous)
            target = reference[:, s:s + width]
            if self.stage == 'early':
                core_losses.append(state_distillation_loss(previous.float(), target.float()))
                attention_losses.append(state_distillation_loss(
                    attention[:, s:s + width].float(), previous.detach().float()))
            # Stop gradients on both sides: P cannot move R/A to make drift artificially small.
            actual = (1 - F.cosine_similarity(previous.detach().float(), target.float(), dim=-1)).clamp(0, 1)
            predicted = self.drift(previous.detach()).float()
            drift_losses.append(F.mse_loss(predicted, actual))
            drift_values.append(actual.mean())
        if decoder_step is None:
            decoder_step = int(torch.randint(1, hops + 1, ()).item()) if self.training and self.stage == 'early' else hops
        if not 1 <= decoder_step <= hops:
            raise ValueError('Invalid decoder step')
        if decoder_source is None:
            decoder_source = ('attention' if torch.rand(()).item() < 0.5 else 'core') if self.training and self.stage == 'early' else 'core'
        if decoder_source not in ('attention', 'core') or (self.stage == 'late' and decoder_source != 'core'):
            raise ValueError('Late stage uses the connected A -> R -> D path')
        s = decoder_step
        state = attention[:, s:s + width] if decoder_source == 'attention' else core_states[s - 1]
        if self.stage == 'early':
            state = state.detach()
        logits = self.decoder(state, torch.arange(s, s + width, device=ids.device))
        zero = logits.new_zeros((), dtype=torch.float32)
        losses = {
            'loss_core': torch.stack(core_losses).mean() if core_losses else zero,
            'loss_attention': torch.stack(attention_losses).mean() if attention_losses else zero,
            'loss_drift': torch.stack(drift_losses).mean(),
            'loss_lm': F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                labels[:, s:s + width].reshape(-1)) if labels is not None else zero,
        }
        # A receives direct next-token supervision through the current decoder, but
        # this auxiliary path cannot update D or R. Disable decoder checkpointing in
        # this functional call so recomputation cannot escape its detached weights.
        attention_lm_weight = float(self.config.get('lambda_attention_lm', 0.0))
        if attention_lm_weight < 0:
            raise ValueError('lambda_attention_lm must be nonnegative')
        losses['loss_attention_lm'] = zero
        if self.stage == 'late' and labels is not None and attention_lm_weight > 0:
            auxiliary_positions = torch.arange(hops, hops + width, device=ids.device)
            fixed_decoder = {name: parameter.detach() for name, parameter in self.decoder.named_parameters()}
            auxiliary_logits = functional_call(self.decoder, fixed_decoder,
                (attention[:, hops:hops + width], auxiliary_positions), {'use_checkpoint': False})
            losses['loss_attention_lm'] = F.cross_entropy(
                auxiliary_logits.float().reshape(-1, auxiliary_logits.shape[-1]),
                labels[:, hops:hops + width].reshape(-1))
        loss = losses['loss_lm'] + self.config.get('lambda_drift', 0.05) * losses['loss_drift']
        loss = loss + attention_lm_weight * losses['loss_attention_lm']
        if self.stage == 'early':
            loss = loss + self.config.get('lambda_core', 1.0) * losses['loss_core']
            loss = loss + self.config.get('lambda_attention', 1.0) * losses['loss_attention']
        return dict(losses, loss=loss, logits=logits, drift_target=torch.stack(drift_values).mean(),
                    decoder_source=decoder_source, decoder_step=s)
