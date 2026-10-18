"""Compatibility regressions for the public Latex2im inference path."""
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import GPTNeoConfig, GPTNeoModel

from diffusers import DDPMPipeline, DDPMScheduler, UNet2DConditionModel
from diffusers.utils.hub_compat import _download


class HubCompatibilityTests(unittest.TestCase):
    def test_new_hub_preserves_auth_cache_and_revision(self):
        def download(repo_id, *, token=None, revision=None, cache_dir=None, local_files_only=False):
            return repo_id, token, revision, cache_dir, local_files_only
        self.assertEqual(_download(download, 'owner/model', use_auth_token=False,
                         resume_download=False, proxies=None, revision='commit',
                         cache_dir='/test/cache', local_files_only=True),
                         ('owner/model', False, 'commit', '/test/cache', True))

    def test_old_hub_preserves_explicit_controls(self):
        def download(*, token=None, resume_download=False, proxies=None):
            return token, resume_download, proxies
        proxies = {'https': 'https://proxy.example'}
        self.assertEqual(_download(download, use_auth_token='test-token',
                         resume_download=True, proxies=proxies), ('test-token', True, proxies))

    def test_unsupported_proxy_is_not_silently_discarded(self):
        def download(*, token=None):
            self.fail('Request must not be sent without the requested proxy')
        with self.assertRaisesRegex(ValueError, 'per-call proxies'):
            _download(download, proxies={'https': 'https://proxy.example'})

    def test_resume_change_is_reported(self):
        with patch('diffusers.utils.hub_compat.warnings.warn') as warn:
            self.assertEqual(_download(lambda: 'ok', resume_download=True), 'ok')
        self.assertIn('automatically', warn.call_args.args[0])
        self.assertIs(warn.call_args.args[1], FutureWarning)

    def test_conflicting_tokens_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'only one'):
            _download(lambda **kw: kw, token=False, use_auth_token=True)


class PipelineCompatibilityTests(unittest.TestCase):
    def test_conditioned_pipeline_reload_and_clean_image_stream(self):
        torch.manual_seed(5)
        encoder = GPTNeoModel(GPTNeoConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_layers=1,
            num_heads=2, attention_types=[[['global'], 1]], max_position_embeddings=64,
            bos_token_id=1, eos_token_id=2,
        )).eval()
        with patch('transformers.AutoModel.from_pretrained', return_value=encoder):
            model = UNet2DConditionModel(
                sample_size=(8, 8), in_channels=3, out_channels=3,
                block_out_channels=(32,), down_block_types=('CrossAttnDownBlock2D',),
                up_block_types=('CrossAttnUpBlock2D',), layers_per_block=1,
                cross_attention_dim=16, create_text_encoder=True,
            ).eval()
            pipeline = DDPMPipeline(model, DDPMScheduler())
            with tempfile.TemporaryDirectory() as directory:
                pipeline.save_pretrained(directory)
                restored = DDPMPipeline.from_pretrained(directory)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, restored.unet.state_dict()[key]), key)
        with torch.no_grad():
            states = restored.unet.text_encoder(torch.tensor([[1, 2, 3]])).last_hidden_state
        original_set_timesteps = restored.scheduler.set_timesteps
        def two_steps(count):
            original_set_timesteps(count)
            restored.scheduler.timesteps = restored.scheduler.timesteps[-2:]
        with patch.object(restored.scheduler, 'set_timesteps', side_effect=two_steps):
            frames = list(restored.run_clean(encoder_hidden_states=states,
                          generator=torch.Generator().manual_seed(0), output_type='numpy'))
        self.assertEqual(len(frames), 2)
        for noisy, clean in frames:
            self.assertEqual(clean.shape, (1, 8, 8, 3))
            self.assertTrue(np.isfinite(clean).all())
            self.assertTrue((clean >= 0).all() and (clean <= 1).all())


if __name__ == '__main__':
    unittest.main()
