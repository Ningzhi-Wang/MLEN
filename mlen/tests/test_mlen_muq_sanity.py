from dataclasses import asdict

import pytest
import torch

from mlen.models import MLENModel
from muq import MuQ


MODEL_NAME = "OpenMuQ/MuQ-large-msd-iter"
COND_DIM = 8
LAYER_IX = 12
WAVEFORM_SECONDS = 1
SAMPLE_RATE = 24_000


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _random_waveform(batch_size=1, device=None, dtype=torch.float32):
    generator = torch.Generator(device="cpu").manual_seed(1234)
    wav = torch.randn(
        batch_size,
        WAVEFORM_SECONDS * SAMPLE_RATE,
        generator=generator,
        dtype=torch.float32,
    )
    return wav.to(device=device or _device(), dtype=dtype)

def _load_muq():
    try:
        return MuQ.from_pretrained(MODEL_NAME)
    except Exception as exc:
        pytest.skip(f"MuQ checkpoint {MODEL_NAME!r} is not available: {exc}")


def _mlen_from_muq_config(config, use_flash_attention=False):
    values = asdict(config)
    return MLENModel(
        num_codebooks=values["num_codebooks"],
        codebook_dim=values["codebook_dim"],
        codebook_size=values["codebook_size"],
        features=values["features"],
        hop_length=values["hop_length"],
        n_mels=values["n_mels"],
        conv_dim=values["conv_dim"],
        encoder_dim=values["encoder_dim"],
        encoder_depth=values["encoder_depth"],
        mask_hop=values["mask_hop"],
        mask_prob=values["mask_prob"],
        is_flash=values["is_flash"],
        stat=values["stat"],
        w2v2_config=values["w2v2_config"],
        use_rvq_target=values["use_rvq_target"],
        use_vq_target=values["use_vq_target"],
        use_encodec_target=values["use_encodec_target"],
        rvq_ckpt_path=values["rvq_ckpt_path"],
        recon_loss_ratio=values["recon_loss_ratio"],
        label_rate=values["label_rate"],
        rvq_n_codebooks=values["rvq_n_codebooks"],
        rvq_multi_layer_num=values["rvq_multi_layer_num"],
        cond_target_dim=512,
        cond_dim=COND_DIM,
        flash_attention=use_flash_attention,
    )


def _load_muq_weights_into_mlen_model(muq, mlen):
    src = muq.model.state_dict()
    dst = mlen.state_dict()
    loadable = {}
    skipped = []

    for key, value in src.items():
        is_replaced_adaln = (
            key.startswith("conformer.layers.")
            and ".self_attn_layer_norm." in key
        )

        if is_replaced_adaln:
            if key.endswith(".weight"):
                key = key.removesuffix(".weight") + ".gamma"
            elif key.endswith(".bias"):
                key = key.removesuffix(".bias") + ".beta"
            else:
                skipped.append(key)
                continue

        if key == "conformer.pos_conv_embed.conv.parametrizations.weight.original0":
            key = "conformer.pos_conv_embed.conv.weight_g"
        elif key == "conformer.pos_conv_embed.conv.parametrizations.weight.original1":
            key = "conformer.pos_conv_embed.conv.weight_v"

        if key in dst and dst[key].shape == value.shape:
            loadable[key] = value
        else:
            raise AssertionError(
                f"Unexpected MuQ key {key!r} with shape {tuple(value.shape)}"
            )

    missing, unexpected = mlen.load_state_dict(loadable, strict=False)
    return missing, unexpected, skipped


def _clear_rotary_cache(model):
    embed_positions = getattr(model.conformer, "embed_positions", None)
    if hasattr(embed_positions, "cached_rotary_positional_embedding"):
        embed_positions.cached_rotary_positional_embedding = None
        embed_positions.cached_sequence_length = None


@pytest.fixture(scope="module")
def pretrained_muq():
    return _load_muq()


@pytest.mark.parametrize(
    ("use_flash_attention", "dtype"),
    [
        pytest.param(False, torch.float32, id="float32-no-flash"),
        pytest.param(
            True,
            torch.bfloat16,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="flash attention requires CUDA",
            ),
            id="bfloat16-flash",
        ),
    ],
)
def test_initialized_mlen_latents_are_condition_invariant(
    pretrained_muq,
    use_flash_attention,
    dtype,
):
    device = _device()
    torch.manual_seed(0)
    mlen = (
        _mlen_from_muq_config(
            pretrained_muq.config,
            use_flash_attention=use_flash_attention,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    wav = _random_waveform(device=device, dtype=dtype)
    conds = torch.eye(COND_DIM, device=device, dtype=dtype)

    with torch.inference_mode():
        latents_a = mlen.get_latent(wav, conds[0:1], layer_ix=LAYER_IX)
        latents_b = mlen.get_latent(wav, conds[1:2], layer_ix=LAYER_IX)

    torch.testing.assert_close(latents_a, latents_b, atol=1e-6, rtol=0.0)


def test_mlen_loaded_with_muq_weights_matches_muq_latents(pretrained_muq):
    device = _device()
    muq_model = pretrained_muq.model.to(device).float().eval()
    mlen = _mlen_from_muq_config(pretrained_muq.config, use_flash_attention=False).to(device).float().eval()
    missing, unexpected, skipped = _load_muq_weights_into_mlen_model(
        pretrained_muq,
        mlen,
    )
    assert unexpected == []
    assert skipped == []
    assert all(
        key.startswith("cond_proj.")
        or (
            key.startswith("conformer.layers.")
            and ".self_attn_layer_norm.cond_mlp." in key
        )
        for key in missing
    ), missing

    _clear_rotary_cache(mlen)
    wav = _random_waveform(device=device)
    cond = torch.eye(COND_DIM, device=device, dtype=torch.float32)[0:1]

    with torch.inference_mode():
        mlen_latents = mlen.get_latent(wav, cond, layer_ix=LAYER_IX)
        muq_latents = muq_model.get_latent(wav, layer_ix=LAYER_IX)

    torch.testing.assert_close(mlen_latents, muq_latents, atol=1e-6, rtol=0.0)
