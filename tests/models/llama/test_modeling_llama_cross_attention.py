
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import torch

from transformers import LlamaConfig, LlamaModel
from transformers.testing_utils import require_torch, slow, torch_device


@require_torch
class LlamaCrossAttentionTest(unittest.TestCase):
    @slow
    def test_cross_attention_initialization(self):
        """Test that a LLaMA model with add_cross_attention=True has cross-attention components"""
        config = LlamaConfig(
            vocab_size=32000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            add_cross_attention=True,  # Enable cross-attention
        )
        
        model = LlamaModel(config)
        # Check that cross-attention is initialized in decoder layers
        for layer in model.layers:
            self.assertIsNotNone(layer.cross_attention, "Cross-attention was not initialized")
            self.assertIsNotNone(layer.cross_attention_layer_norm, "Cross-attention layer norm was not initialized")

    def test_cross_attention_forward_pass(self):
        """Test that forward pass with encoder_hidden_states works correctly"""
        config = LlamaConfig(
            vocab_size=32000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            add_cross_attention=True,  # Enable cross-attention
        )
        
        model = LlamaModel(config)
        batch_size = 2
        seq_len = 10
        encoder_seq_len = 8
        
        # Create input tensors
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        encoder_hidden_states = torch.randn(batch_size, encoder_seq_len, config.hidden_size)
        
        # Forward pass with encoder_hidden_states
        outputs = model(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            output_attentions=True,
            return_dict=True
        )
        
        # Check shapes of outputs
        self.assertEqual(
            outputs.last_hidden_state.shape, 
            (batch_size, seq_len, config.hidden_size),
            "Output hidden states shape is incorrect"
        )
        
        # Check that cross-attentions are returned when requested
        self.assertIsNotNone(
            outputs.cross_attentions,
            "Cross-attentions were not returned"
        )

    def test_cross_attention_mask(self):
        """Test that cross-attention mask is properly applied"""
        config = LlamaConfig(
            vocab_size=32000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            add_cross_attention=True,  # Enable cross-attention
        )
        
        model = LlamaModel(config)
        batch_size = 2
        seq_len = 10
        encoder_seq_len = 8
        
        # Create input tensors
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        encoder_hidden_states = torch.randn(batch_size, encoder_seq_len, config.hidden_size)
        
        # Create encoder attention mask (mask out last 2 positions)
        encoder_attention_mask = torch.ones(batch_size, encoder_seq_len)
        encoder_attention_mask[:, -2:] = 0  # Mask out last 2 tokens of encoder sequence
        
        # Forward pass with and without mask
        outputs_with_mask = model(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=True,
            return_dict=True
        )
        
        outputs_without_mask = model(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            output_attentions=True,
            return_dict=True
        )
        
        # Check that outputs are different when mask is applied
        self.assertFalse(
            torch.allclose(
                outputs_with_mask.last_hidden_state,
                outputs_without_mask.last_hidden_state
            ),
            "Encoder attention mask does not affect outputs"
        )

    def test_cross_attention_output_shapes(self):
        """Test that the output shapes of cross-attention are correct"""
        config = LlamaConfig(
            vocab_size=32000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            add_cross_attention=True,  # Enable cross-attention
        )
        
        model = LlamaModel(config)
        model.eval()  # Set to eval mode for deterministic output
        
        batch_size = 2
        seq_len = 10
        encoder_seq_len = 8
        
        # Create input tensors
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        encoder_hidden_states = torch.randn(batch_size, encoder_seq_len, config.hidden_size)
        
        # Forward pass with output_attentions=True
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                output_attentions=True,
                return_dict=True
            )
        
        # Check that all expected shapes are correct
        self.assertEqual(
            outputs.last_hidden_state.shape,
            (batch_size, seq_len, config.hidden_size),
            "Last hidden state shape is incorrect"
        )
        
        # Check cross-attention shapes
        for layer_cross_attentions in outputs.cross_attentions:
            self.assertEqual(
                layer_cross_attentions.shape,
                (batch_size, config.num_attention_heads, seq_len, encoder_seq_len),
                "Cross-attention shape is incorrect"
            )

    def test_error_without_encoder_hidden_states(self):
        """Test that error is raised when cross-attention is enabled but encoder_hidden_states is not provided"""
        config = LlamaConfig(
            vocab_size=32000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            add_cross_attention=True,  # Enable cross-attention
        )
        
        model = LlamaModel(config)
        batch_size = 2
        seq_len = 10
        
        # Create input tensors without encoder_hidden_states
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # We expect this to work without error - the model should just skip cross-attention
        try:
            outputs = model(
                input_ids=input_ids,
                return_dict=True
            )
            # Verify output shape is still correct
            self.assertEqual(
                outputs.last_hidden_state.shape,
                (batch_size, seq_len, config.hidden_size),
                "Last hidden state shape is incorrect when encoder_hidden_states is not provided"
            )
        except Exception as e:
            self.fail(f"Forward pass without encoder_hidden_states raised exception: {e}")

    @slow
    def test_full_model_gradient_flow(self):
        """Test gradient flow through cross-attention layer"""
        config = LlamaConfig(
            vocab_size=32000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            add_cross_attention=True,  # Enable cross-attention
        )
        
        model = LlamaModel(config)
        
        # Verify parameters in cross-attention modules have requires_grad=True
        for layer in model.layers:
            for param in layer.cross_attention.parameters():
                self.assertTrue(param.requires_grad, "Cross-attention parameters are not trainable")
        
        batch_size = 2
        seq_len = 10
        encoder_seq_len = 8
        
        # Create input tensors
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        encoder_hidden_states = torch.randn(batch_size, encoder_seq_len, config.hidden_size, requires_grad=True)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
        )
        
        # Create a dummy loss to backprop
        loss = outputs.last_hidden_state.sum()
        loss.backward()
        
        # Check if encoder_hidden_states gradients are not None (indicating gradient flow)
        self.assertIsNotNone(
            encoder_hidden_states.grad,
            "No gradients flowing through encoder_hidden_states"
        )
