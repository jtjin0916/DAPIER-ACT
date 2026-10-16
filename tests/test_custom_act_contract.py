import torch

from dapier_act.model_multimodal import DapierACTMultimodal


def test_multimodal_inference_contract_cpu():
    torch.manual_seed(0)

    model = DapierACTMultimodal(
        qpos_dim=12,
        action_dim=12,
        chunk_size=20,
        hidden_dim=256,
        latent_dim=32,
        num_point_samples=512,
        candidate_points=8192,
        point_min=[0.0, -0.25, 0.005],
        point_max=[0.4, 0.25, 0.30],
        nheads=8,
        encoder_layers=4,
        decoder_layers=4,
        ff_dim=1024,
        dropout=0.0,
        pretrained_wrist_backbone=False,
    ).eval()

    qpos = torch.zeros(1, 12)

    # >512 points exercises candidate/FPS behavior without requiring a dataset.
    n = 520
    xyz = torch.stack(
        [
            torch.linspace(0.01, 0.39, n),
            torch.linspace(-0.24, 0.24, n),
            torch.linspace(0.01, 0.29, n),
        ],
        dim=1,
    )

    left = torch.zeros(1, 3, 240, 320)
    right = torch.zeros(1, 3, 240, 320)

    with torch.no_grad():
        action, mu, logvar, feat = model(
            qpos=qpos,
            pointcloud=[xyz],
            left_wrist=left,
            right_wrist=right,
            actions=None,
            is_pad=None,
            return_features=True,
        )

    assert action.shape == (1, 20, 12)
    assert mu is None
    assert logvar is None

    assert feat["sampled_xyz"].shape == (1, 512, 3)
    assert feat["point_features"].shape == (1, 1, 256)
    assert feat["head_tokens"].shape == (1, 1, 256)
    assert feat["left_tokens"].shape == (1, 80, 256)
    assert feat["right_tokens"].shape == (1, 80, 256)
    assert feat["context"].shape == (1, 163, 256)
    assert torch.isfinite(action).all()
