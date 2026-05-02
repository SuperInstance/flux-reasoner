"""Tests for FluxReasoner gradient computation and API calls."""

import pytest
from unittest.mock import patch, MagicMock
from flux_reasoner import FluxReasoner


class TestFluxReasoner:
    """Test suite for FluxReasoner."""

    def test_init_default_key(self):
        """Test initialization with default key from env."""
        with patch.dict('os.environ', {'DEEPINFRA_API_KEY': 'test-key'}):
            reasoner = FluxReasoner()
            assert reasoner.deepinfra_key == 'test-key'

    def test_init_custom_key(self):
        """Test initialization with custom key."""
        reasoner = FluxReasoner(deepinfra_key='custom-key')
        assert reasoner.deepinfra_key == 'custom-key'

    def test_compute_gradient_high_novelty(self):
        """Test gradient when creative output has many unique words."""
        reasoner = FluxReasoner()
        # With 50+ words, novelty approaches 1.0, gradient ~0.5
        creative = "foo bar baz qux quux corge grault garply waldo fred plugh xyzzy thud one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twentyone twentytwo twentythree twentyfour twentyfive"
        logical = "some constraints limiting options"
        gradient = reasoner.compute_gradient(creative, logical)
        assert gradient > 0.3  # positive gradient

    def test_compute_gradient_low_novelty(self):
        """Test gradient when creative output has low novelty."""
        reasoner = FluxReasoner()
        creative = "the and a to in of is it"
        logical = "the and a to in of is it that this those"
        gradient = reasoner.compute_gradient(creative, logical)
        assert gradient < 0.3  # low novelty, high constraint overlap

    def test_compute_gradient_clamped_to_one(self):
        """Test gradient is clamped to 1.0 maximum."""
        reasoner = FluxReasoner()
        # 100 unique words → novelty=2.0, clamped to 1.0, constraint=0 → gradient=1.0
        creative = " ".join([f"unique{i}" for i in range(100)])
        logical = ""
        gradient = reasoner.compute_gradient(creative, logical)
        assert gradient == 1.0

    def test_compute_gradient_clamped_to_zero(self):
        """Test gradient is clamped to 0.0 minimum."""
        reasoner = FluxReasoner()
        creative = "the the the the the the the the the the the the the the the"
        logical = "the the the the the the the the the the the the the the the"
        gradient = reasoner.compute_gradient(creative, logical)
        assert gradient == 0.0

    @patch('flux_reasoner.requests.post')
    def test_call_deepinfra_seed_mini(self, mock_post):
        """Test DeepInfra Seed-2.0-mini API call."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "creative output"}}]
        }
        mock_post.return_value = mock_response

        reasoner = FluxReasoner()
        result = reasoner.call_deepinfra_seed_mini("test prompt", temperature=0.85)

        assert result == "creative output"
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs['json']['model'] == "ByteDance/Seed-2.0-mini"
        assert call_kwargs['json']['temperature'] == 0.85

    @patch('flux_reasoner.requests.post')
    def test_call_deepseek(self, mock_post):
        """Test DeepSeek-v4-flash API call via SiliconFlow."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "logical evaluation"}}]
        }
        mock_post.return_value = mock_response

        reasoner = FluxReasoner()
        with patch.dict('os.environ', {'SILICONFLOW_KEY': 'sf-key'}):
            result = reasoner.call_deepseek("test prompt")

        assert result == "logical evaluation"

    @patch.object(FluxReasoner, 'call_deepinfra_seed_mini')
    @patch.object(FluxReasoner, 'call_deepseek')
    def test_reason_adopt_creative(self, mock_deepseek, mock_deepinfra):
        """Test ADOPT_CREATIVE decision when gradient > threshold."""
        mock_deepinfra.return_value = "unique creative divergent words option1 option2 option3 option4 option5 option6 option7 option8 option9 option10 option11 option12 option13 option14 option15 option16 option17 option18 option19 option20"
        mock_deepseek.return_value = "some constraints"

        reasoner = FluxReasoner()
        result = reasoner.reason("test input", threshold=0.35)

        assert result['decision'] == "ADOPT_CREATIVE"
        assert result['gradient'] > 0.35

    @patch.object(FluxReasoner, 'call_deepinfra_seed_mini')
    @patch.object(FluxReasoner, 'call_deepseek')
    def test_reason_adopt_logical(self, mock_deepseek, mock_deepinfra):
        """Test ADOPT_LOGICAL decision when gradient < threshold * 0.5."""
        mock_deepinfra.return_value = "the and the and the and the and the"
        mock_deepseek.return_value = "the and the and the and the and the the the"

        reasoner = FluxReasoner()
        result = reasoner.reason("test input", threshold=0.35)

        assert result['decision'] == "ADOPT_LOGICAL"
        assert result['gradient'] < 0.35 * 0.5

    @patch.object(FluxReasoner, 'call_deepinfra_seed_mini')
    @patch.object(FluxReasoner, 'call_deepseek')
    def test_reason_hold(self, mock_deepseek, mock_deepinfra):
        """Test HOLD decision when gradient is between thresholds."""
        mock_deepinfra.return_value = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12 word13 word14 word15"
        mock_deepseek.return_value = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        reasoner = FluxReasoner()
        result = reasoner.reason("test input", threshold=0.35)

        assert result['decision'] == "HOLD"
        assert 0.35 * 0.5 <= result['gradient'] <= 0.35

    @patch.object(FluxReasoner, 'call_deepinfra_seed_mini')
    @patch.object(FluxReasoner, 'call_deepseek')
    def test_reason_with_iterations(self, mock_deepseek, mock_deepinfra):
        """Test iterative reasoning stops at convergence."""
        mock_deepinfra.side_effect = [
            "first creative output with many unique creative divergent words option1 option2 option3 option4 option5 option6 option7 option8 option9 option10 option11 option12 option13 option14 option15",
            "refined creative output with different unique approach option1 option2 option3 option4 option5 option6 option7 option8 option9 option10 option11 option12 option13 option14 option15",
        ]
        mock_deepseek.side_effect = [
            "some critique of the first output",
            "another critique with more constraints but still some overlap",
        ]

        reasoner = FluxReasoner()
        result = reasoner.reason_with_iterations("test input", iterations=3, threshold=0.35)

        assert len(result['iterations']) >= 1
        assert 'final_gradient' in result
        assert 'converged' in result

    @patch.object(FluxReasoner, 'call_deepinfra_seed_mini')
    @patch.object(FluxReasoner, 'call_deepseek')
    def test_reason_with_iterations_max_iterations(self, mock_deepseek, mock_deepinfra):
        """Test iterative reasoning respects max iterations."""
        mock_deepinfra.return_value = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12 word13 word14 word15 word16 word17 word18 word19 word20"
        mock_deepseek.return_value = "word1 word2 word3"

        reasoner = FluxReasoner()
        result = reasoner.reason_with_iterations("test input", iterations=3, threshold=0.99)

        assert len(result['iterations']) == 3
        assert result['converged'] is False


class TestGradientComputation:
    """Test gradient computation in isolation."""

    def test_empty_strings(self):
        """Test gradient with empty strings - no novelty."""
        reasoner = FluxReasoner()
        gradient = reasoner.compute_gradient("", "")
        # Empty creative: novelty=0, constraint=0, gradient=0.0
        assert gradient == 0.0

    def test_perfect_overlap(self):
        """Test gradient when creative and logical are identical."""
        reasoner = FluxReasoner()
        text = "word1 word2 word3"
        gradient = reasoner.compute_gradient(text, text)
        # novelty=3/50=0.06, constraint=1.0, gradient=0.06-0.5=0 (clamped)
        assert gradient < 0.1

    def test_no_overlap(self):
        """Test gradient when there is no overlap."""
        reasoner = FluxReasoner()
        creative = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
        logical = "kilo lima mike november oscar papa quebec romeo sierra tango"
        gradient = reasoner.compute_gradient(creative, logical)
        # novelty=10/50=0.2, constraint=0, gradient=0.2-0=0.2
        assert 0.15 <= gradient <= 0.25  # modest novelty with 10 words
