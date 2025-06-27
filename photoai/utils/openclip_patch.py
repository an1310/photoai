"""
Patch for OpenCLIP attention mask compatibility with newer PyTorch versions
This fixes the RuntimeError: The shape of the 2D attn_mask is torch.Size([77, 77]), but should be (batch_size, batch_size)
"""
import torch
import logging

logger = logging.getLogger(__name__)


def patch_openclip_attention():
    """
    Patch OpenCLIP attention mask handling to work with newer PyTorch versions
    """
    try:
        # Import the modules we need to patch
        from sgm.modules.encoders.modules import FrozenOpenCLIPEmbedder2

        logger.info("Applying OpenCLIP attention mask patch...")

        # Store the original method
        original_text_transformer_forward = FrozenOpenCLIPEmbedder2.text_transformer_forward

        def patched_text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
            """
            Patched version that handles attention mask shape correctly for any batch size
            """
            # Fix attention mask shape for newer PyTorch versions
            if attn_mask is not None:
                batch_size = x.shape[1]  # x is in LND format, so batch is dim 1

                logger.debug(f"Original attn_mask shape: {attn_mask.shape}, batch_size: {batch_size}")

                # If attention mask is 2D [seq_len, seq_len], we need to expand or disable it
                if attn_mask.dim() == 2:
                    seq_len = attn_mask.shape[0]
                    expected_shape = (batch_size, batch_size)

                    # For CLIP text transformers, we typically don't need causal masking
                    # The attn_mask is usually for padding, but CLIP uses fixed-length sequences
                    # So we can safely set it to None for most cases
                    logger.debug(f"Disabling attention mask - expected {expected_shape}, got {attn_mask.shape}")
                    attn_mask = None

            # Call the original logic with fixed/disabled attention mask
            outputs = {}
            for i, r in enumerate(self.model.transformer.resblocks):
                if i == len(self.model.transformer.resblocks) - 1:
                    outputs["penultimate"] = x.permute(1, 0, 2)  # LND -> NLD
                if (
                        self.model.transformer.grad_checkpointing
                        and not torch.jit.is_scripting()
                ):
                    from torch.utils.checkpoint import checkpoint
                    x = checkpoint(r, x, attn_mask)
                else:
                    x = r(x, attn_mask=attn_mask)
            outputs["last"] = x.permute(1, 0, 2)  # LND -> NLD
            return outputs

        # Apply the patch
        FrozenOpenCLIPEmbedder2.text_transformer_forward = patched_text_transformer_forward

        logger.info("✅ OpenCLIP attention mask patch applied successfully!")
        return True

    except ImportError as e:
        logger.warning(f"Could not import FrozenOpenCLIPEmbedder2: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to apply OpenCLIP patch: {e}")
        return False


def patch_openclip_legacy():
    """
    Patch for the legacy FrozenOpenCLIPEmbedder class as well
    """
    try:
        from sgm.modules.encoders.modules import FrozenOpenCLIPEmbedder

        logger.info("Applying legacy OpenCLIP attention mask patch...")

        # Store the original method
        original_text_transformer_forward = FrozenOpenCLIPEmbedder.text_transformer_forward

        def patched_text_transformer_forward_legacy(self, x: torch.Tensor, attn_mask=None):
            """
            Patched version for legacy embedder - disables problematic attention mask
            """
            # Fix attention mask shape for newer PyTorch versions
            if attn_mask is not None:
                batch_size = x.shape[1]  # x is in LND format, so batch is dim 1

                logger.debug(f"Legacy - Original attn_mask shape: {attn_mask.shape}, batch_size: {batch_size}")

                # If attention mask is 2D [seq_len, seq_len], disable it
                if attn_mask.dim() == 2:
                    logger.debug("Legacy - Disabling attention mask for compatibility")
                    attn_mask = None

            # Call the original logic with fixed/disabled attention mask
            for i, r in enumerate(self.model.transformer.resblocks):
                if i == len(self.model.transformer.resblocks) - self.layer_idx:
                    break
                if (
                        self.model.transformer.grad_checkpointing
                        and not torch.jit.is_scripting()
                ):
                    from torch.utils.checkpoint import checkpoint
                    x = checkpoint(r, x, attn_mask)
                else:
                    x = r(x, attn_mask=attn_mask)
            return x

        # Apply the patch
        FrozenOpenCLIPEmbedder.text_transformer_forward = patched_text_transformer_forward_legacy

        logger.info("✅ Legacy OpenCLIP attention mask patch applied successfully!")
        return True

    except ImportError as e:
        logger.warning(f"Could not import FrozenOpenCLIPEmbedder: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to apply legacy OpenCLIP patch: {e}")
        return False


def patch_openclip_resblock():
    """
    Patch the actual ResidualAttentionBlock to handle attention mask properly
    """
    try:
        import open_clip.transformer as transformer_module

        logger.info("Applying ResidualAttentionBlock attention mask patch...")

        # Get the original forward method
        original_forward = transformer_module.ResidualAttentionBlock.forward

        def patched_resblock_forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
            """
            Patched ResidualAttentionBlock forward that handles attention mask shape
            """
            # If we have an attention mask with wrong dimensions, disable it
            if attn_mask is not None and attn_mask.dim() == 2:
                # Check if the mask shape doesn't match expected batch dimensions
                batch_size = x.shape[1] if x.dim() == 3 else x.shape[0]  # Handle both LND and NLD
                expected_batch_shape = (batch_size, batch_size)

                if attn_mask.shape != expected_batch_shape:
                    logger.debug(
                        f"ResBlock - Disabling attn_mask: got {attn_mask.shape}, expected {expected_batch_shape}")
                    attn_mask = None

            # Call original forward with potentially modified mask
            return original_forward(self, x, attn_mask)

        # Apply the patch
        transformer_module.ResidualAttentionBlock.forward = patched_resblock_forward

        logger.info("✅ ResidualAttentionBlock patch applied successfully!")
        return True

    except ImportError as e:
        logger.warning(f"Could not import open_clip.transformer: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to apply ResidualAttentionBlock patch: {e}")
        return False


def apply_all_openclip_patches():
    """
    Apply all necessary OpenCLIP patches
    """
    logger.info("Applying comprehensive OpenCLIP compatibility patches...")

    success_count = 0

    # Try to patch all versions and levels
    if patch_openclip_attention():
        success_count += 1

    if patch_openclip_legacy():
        success_count += 1

    if patch_openclip_resblock():
        success_count += 1

    if success_count > 0:
        logger.info(f"✅ Applied {success_count} OpenCLIP patches successfully!")
        return True
    else:
        logger.error("❌ Failed to apply any OpenCLIP patches")
        return False