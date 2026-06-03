import torch
from einops import rearrange
from easydict import EasyDict
from torch import nn

from .conformer import CondConformerEncoder
from muq.muq.models.muq_model import MuQModel

class MLENModel(MuQModel):
    def __init__(
            self, 
            cond_dim=None, 
            cond_target_dim=512, 
            flash_attention=True, 
            **kwargs
        ):
        super().__init__( **kwargs)
        config = EasyDict(kwargs.get("w2v2_config", {}))
        config.num_hidden_layers = kwargs.get("encoder_depth", 12)
        config.hidden_size = kwargs.get("encoder_dim", 1024)
        if cond_dim is not None:
            self.cond = True
            self.cond_proj = nn.Linear(kwargs["encoder_dim"], cond_target_dim)
            self.cond_loss = nn.MSELoss()
            config.cond_dim = cond_dim
            config.flash_attention = flash_attention
            self.conformer = CondConformerEncoder(config)
        else:
            self.cond = False

    @torch.no_grad()
    def preprocessing(self, x, features):
        """extract classic audio features"""
        # check precision
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            precision = 16
        else:
            precision = 32

        out = {}
        for key in features:
            layer = getattr(self, "preprocessor_%s" % key)
            layer.to(x.device)
            dtype = x.dtype
            out[key] = layer(x.float())[..., :-1]
            if precision == 16:
                out[key] = out[key].half()
            if out[key].dtype != dtype:
                out[key] = out[key].to(dtype=dtype)
        return out

    def get_targets(self, x, label=None):
        if label is None:
            return super().get_targets(x, label=label)

        x = self.preprocessing(x, features=self.features)
        x = self.normalize(x)
        x = self.rearrange(x)
        melspec = x['melspec_2048']

        if label.dim() == 2:
            target = label.long()
        elif label.dim() == 3:
            target = rearrange(label, "b n s -> b (n s)").long()
        else:
            raise ValueError(f"Expected label with 2 or 3 dims, got shape {tuple(label.shape)}")
        return {'melspec_2048': target}, melspec

    def encoder_cond(self, x, cond, *, attention_mask=None, is_features_only=False):
        """2-layer conv + w2v-conformer"""
        x = self.conv(x)
        mask_indices = None
        if attention_mask is None:
            out = self.conformer(x, cond=cond, output_hidden_states=True)
        else:
            attention_mask = attention_mask.bool()
            skip_n = int(attention_mask.size(-1) / x.size(1))
            attention_mask = attention_mask[:, ::skip_n]
            attention_mask = attention_mask[:, :x.size(1)]
            out = self.conformer(x, cond=cond, attention_mask=attention_mask, output_hidden_states=True)
        hidden_emb = out["hidden_states"]
        last_emb = out["last_hidden_state"]
        logits = self.linear(last_emb)
        interval = self.codebook_size
        logits = {
            key: logits[:, :, i * interval : (i + 1) * interval]
            for i, key in enumerate(self.features)
        }
        return logits, hidden_emb, mask_indices

    def encoder(self, x, *args, attention_mask=None, is_features_only=False):
        if self.cond:
            assert len(args) > 0 and args[0] is not None, "cond is required for MLENModel when self.cond is True"
            return self.encoder_cond(x, args[0], attention_mask=attention_mask, is_features_only=is_features_only)
        else:
            return super().encoder(x, attention_mask=attention_mask, is_features_only=is_features_only)

    def get_predictions(self, x, *args, mask=None, attention_mask=None, return_new_mask=False, is_features_only=False):
        # preprocessing
        x = self.preprocessing(x, features=["melspec_2048"])
        x = self.normalize(x)

        # encoding
        logits, hidden_emb, new_mask = self.encoder(x["melspec_2048"], *args, attention_mask=attention_mask, is_features_only=is_features_only)

        if return_new_mask:
            return logits, hidden_emb, mask if new_mask is None else new_mask
        else:
            return logits, hidden_emb

    def get_latent(self, x, *args, layer_ix=12):
        _, hidden_states = self.get_predictions(x, *args)
        emb = hidden_states[layer_ix]
        return emb

    def get_loss(self, logits, target_tokens, masked_indices):
        losses = {}
        accuracies = {}
        for key in logits.keys():
            if not self.use_rvq_like_target:
                masked_logits = logits[key][tuple(masked_indices.t())]
                masked_tokens = target_tokens[key][tuple(masked_indices.t())]
            else:
                Batch, SeqLen, N_Codebook_x_CodebookSize = logits[key].shape
                Batch, N_Codebook_x_SeqLen = target_tokens[key].shape 
                N_Codebook = int(N_Codebook_x_SeqLen // SeqLen)
                target_tokens[key] = rearrange(target_tokens[key], "b (n s) -> b s n", n=N_Codebook) # Batch, SeqLen=750, N_Codebook=4
                masked_logits = logits[key][tuple(masked_indices.t())] 
                masked_tokens = target_tokens[key][tuple(masked_indices.t())] 
                masked_logits = rearrange(masked_logits, "b (n c) -> (b n) c", n=N_Codebook)
                masked_tokens = rearrange(masked_tokens, "b n -> (b n)", n=N_Codebook) 

            losses[key] = self.loss(masked_logits, masked_tokens)
            accuracies[key] = (
                torch.sum(masked_logits.argmax(-1) == masked_tokens)
                / masked_tokens.numel()
            )
        return losses, accuracies

    def get_cond_loss(self, last_hidden_emb, cond_target):
        cond_pred = self.cond_proj(last_hidden_emb).mean(dim=1)
        cond_loss = self.cond_loss(cond_pred, cond_target)
        return cond_loss


    def forward(self, x, *args, attention_mask=None, label=None, cond_label=None):
        # get target feature tokens
        if self.training:
            target_tokens, melspec = self.get_targets(x, label=label) 
            # masking
            x, masked_indices = self.masking(x, attention_mask=attention_mask) 
        else:
            masked_indices = None

        # forward
        logits, hidden_emb, masked_indices = self.get_predictions(x, *args, mask=masked_indices, attention_mask=attention_mask, return_new_mask=True) 
        
        if not self.training:
            return logits, hidden_emb

        # get loss
        losses, accuracies = self.get_loss(logits, target_tokens, masked_indices)

        if self.recon_loss_ratio:
            losses["recon_loss"] = self.get_recon_loss(hidden_emb[-1], melspec, masked_indices) * self.recon_loss_ratio
        
        if self.cond:
            assert cond_label is not None, "cond_label is required for MLENModel in conditional mode"
            losses['cond_loss'] = self.get_cond_loss(hidden_emb[-1], cond_label)

        return logits, hidden_emb, losses, accuracies

