"""Regression tests for the AOTI compilation path.

The bug these guard against: ``three_body_indices`` has shape ``[T, 2]`` where
``T`` is the number of angle (three-body) terms, which changes with the
topology on every MD step. An op that materialized ``T`` as a Python ``int``
during tracing baked the compile-time value into the AOTI artifact. At runtime
that crashed with a CUDA device-side assert for smaller cells and silently
returned wrong energies and forces for larger ones.
"""

import numpy as np
import pytest
import torch

from mattersim.datasets.utils.build import build_dataloader
from mattersim.forcefield.aoti_compile import (
    _THREEBODY_INPUT_INDEX,
    MATTERSIM_DYNAMIC_SHAPES,
    AOTISettings,
    M3GNetForAOTI,
    _get_example_inputs,
    _make_fx,
    assert_threebody_dim_is_dynamic,
    is_dynamic_dim,
)

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

# Number of three-body terms in the compile-time example
# (``_get_example_inputs``: two 8-atom Si-diamond-cubic cells).
EXAMPLE_NUM_TRIPLES = 3840


def _export(potential, device: str):
    """FX-trace and export the AOTI wrapper, mirroring ``compile_m3gnet_aoti``."""
    m3gnet = potential.model
    wrapper = M3GNetForAOTI(m3gnet, device=device, settings=AOTISettings(enabled=True))
    for p in wrapper.parameters():
        p.requires_grad_(False)
    wrapper.eval().to(device)

    example_inputs = _get_example_inputs(
        m3gnet.model_args["cutoff"],
        m3gnet.model_args["threebody_cutoff"],
        torch.device(device),
    )
    assert example_inputs[_THREEBODY_INPUT_INDEX].shape[0] == EXAMPLE_NUM_TRIPLES
    fx_model = _make_fx(wrapper, example_inputs)
    return torch.export.export(
        fx_model, example_inputs, dynamic_shapes=MATTERSIM_DYNAMIC_SHAPES
    )


class _FakeShape:
    """Minimal stand-in for a tensor whose ``.shape`` we control in tests."""

    def __init__(self, shape):
        self.shape = shape


def _threebody_placeholder(exported_model):
    """The ``three_body_indices`` placeholder node of an exported program."""
    target = exported_model.graph_signature.user_inputs[_THREEBODY_INPUT_INDEX]
    return next(
        n
        for n in exported_model.graph.nodes
        if n.op == "placeholder" and n.name == target
    )


def _placeholder_shapes(exported_model):
    """Map each user input name to the shape of its placeholder."""
    by_name = {
        node.name: node
        for node in exported_model.graph.nodes
        if node.op == "placeholder"
    }
    return {
        name: tuple(by_name[name].meta["val"].shape)
        for name in exported_model.graph_signature.user_inputs
    }


@pytest.fixture(scope="module")
def mattersim_potential_aoti():
    """MatterSim 1M potential with its model swapped for the AOTI artifact.

    Compiles on first use (or reuses the on-disk cache) and is therefore
    module-scoped. Loads its own ``Potential`` because ``_apply_aoti``
    replaces ``potential.model`` in place.
    """
    from mattersim.forcefield.potential import Potential
    from mattersim.torchsim.model_loading import _apply_aoti

    potential = Potential.from_checkpoint(device="cuda", load_training_state=False)
    _apply_aoti(potential, AOTISettings(enabled=True))
    return potential


class TestExportKeepsShapesDynamic:
    """Export-level tests: fast, no full AOTI compile, CPU is enough."""

    def test_threebody_dim_stays_symbolic(self, mattersim_potential_cpu):
        """The three-body dimension must not be specialized to a constant.

        This is the direct regression for the root cause: a specialized ``T``
        makes the compiled artifact valid only for structures with exactly
        that many angle terms.
        """
        exported_model = _export(mattersim_potential_cpu, "cpu")
        shapes = _placeholder_shapes(exported_model)
        three_body = shapes[
            exported_model.graph_signature.user_inputs[_THREEBODY_INPUT_INDEX]
        ]

        assert is_dynamic_dim(three_body[0]), (
            f"three_body_indices dim 0 was specialized to {three_body[0]}; "
            "the AOTI artifact would be wrong for any other three-body count"
        )
        # dim 1 is the (i, k) edge pair and is genuinely static
        assert three_body[1] == 2

    def test_all_variable_dims_stay_symbolic(self, mattersim_potential_cpu):
        """Atom, edge, batch and three-body counts must all remain dynamic."""
        exported_model = _export(mattersim_potential_cpu, "cpu")
        shapes = _placeholder_shapes(exported_model)

        specialized = {
            name: shape
            for name, shape in shapes.items()
            # every user input has at least one variable-length dim except the
            # scalar ``num_graphs``
            if shape and not any(is_dynamic_dim(s) for s in shape)
        }
        assert not specialized, f"unexpectedly specialized inputs: {specialized}"

    def test_assert_threebody_dim_is_dynamic_passes(self, mattersim_potential_cpu):
        """The compile-time tripwire accepts a correctly exported model."""
        assert_threebody_dim_is_dynamic(_export(mattersim_potential_cpu, "cpu"))

    def test_assert_threebody_dim_is_dynamic_raises_on_int(
        self, mattersim_potential_cpu
    ):
        """The tripwire rejects a three-body dim that is a plain ``int``."""
        exported_model = _export(mattersim_potential_cpu, "cpu")
        node = _threebody_placeholder(exported_model)
        node.meta["val"] = torch.empty(
            EXAMPLE_NUM_TRIPLES, 2, dtype=torch.long, device="meta"
        )

        with pytest.raises(RuntimeError, match="specialized the three-body"):
            assert_threebody_dim_is_dynamic(exported_model)

    def test_assert_threebody_dim_is_dynamic_raises_on_constant_symint(
        self, mattersim_potential_cpu
    ):
        """The tripwire rejects a ``SymInt`` refined to a constant.

        This is what a real specialization looks like: the guard
        ``Eq(s52, 3840)`` leaves a ``SymInt`` in place whose sympy expression
        is the constant ``3840``, so it is *not* a plain ``int``. An
        ``isinstance(dim, int)`` check would let it through.
        """
        exported_model = _export(mattersim_potential_cpu, "cpu")
        node = _threebody_placeholder(exported_model)
        symbolic_dim = node.meta["val"].shape[0]
        # cancels the symbol, leaving sympy Integer(EXAMPLE_NUM_TRIPLES)
        constant_dim = symbolic_dim * 0 + EXAMPLE_NUM_TRIPLES

        assert not isinstance(constant_dim, int), "expected a SymInt, not an int"
        assert not is_dynamic_dim(constant_dim)

        node.meta["val"] = _FakeShape((constant_dim, 2))
        with pytest.raises(RuntimeError, match="specialized the three-body"):
            assert_threebody_dim_is_dynamic(exported_model)


class TestSphericalHarmonicsScripting:
    """``_spherical_harmonics`` must stay traceable *and* scriptable.

    It is deliberately not decorated with ``@torch.jit.script``: the
    TorchScript interpreter emits ``aten::size``, which pins the three-body
    count during symbolic tracing. Scripting an enclosing module still works
    because TorchScript compiles called functions recursively.
    """

    def test_matches_scripted_version_bitwise(self):
        from mattersim.forcefield.m3gnet.modules.angle_encoding import (
            _spherical_harmonics,
        )

        scripted = torch.jit.script(_spherical_harmonics)
        x = torch.linspace(-1.0, 1.0, 257)
        for lmax in range(4):
            assert torch.equal(_spherical_harmonics(lmax, x), scripted(lmax, x))

    def test_enclosing_layer_is_still_scriptable(self):
        from mattersim.forcefield.m3gnet.modules.angle_encoding import (
            SphericalBasisLayer,
        )

        layer = SphericalBasisLayer(max_n=4, max_l=4, cutoff=5.0).eval()
        scripted = torch.jit.script(layer)
        r = torch.linspace(0.5, 4.5, 64)
        theta = torch.linspace(0.0, np.pi, 64)
        assert torch.equal(layer(r, theta), scripted(r, theta))


@requires_gpu
class TestAOTIMatchesEager:
    """AOTI-vs-eager numeric parity across three-body counts.

    AOTI only runs on CUDA, so this is the layer that exercises the compiled
    artifact itself. Structures are chosen so that ``T`` straddles the
    compile-time example's 3840 in both directions.
    """

    @staticmethod
    def _dataloader(atoms, potential):
        return build_dataloader(
            [atoms],
            batch_size=1,
            model_type="m3gnet",
            shuffle=False,
            only_inference=True,
            cutoff=potential.model.model_args["cutoff"],
            threebody_cutoff=potential.model.model_args["threebody_cutoff"],
        )

    @pytest.mark.parametrize(
        "repeat,displacement",
        [
            ((1, 1, 1), 0.0),  # T = 1920  < 3840
            ((2, 2, 2), 0.0),  # T = 15360 > 3840
            ((2, 2, 2), 0.05),  # rattled: irregular topology, as in MD
        ],
        ids=["T_below_example", "T_above_example", "T_above_example_rattled"],
    )
    def test_energies_forces_stresses_match_eager(
        self,
        mattersim_potential_aoti,
        mattersim_potential_best_device,
        si_diamond_cubic,
        perturb,
        repeat,
        displacement,
    ):
        atoms = si_diamond_cubic.repeat(repeat)
        if displacement:
            atoms = perturb(atoms, displacement=displacement)

        eager = mattersim_potential_best_device
        aoti = mattersim_potential_aoti

        num_triples = int(
            next(iter(self._dataloader(atoms, eager))).three_body_indices.shape[0]
        )
        assert num_triples != EXAMPLE_NUM_TRIPLES, (
            "test structure must not have the compile-time three-body count, "
            "otherwise a specialized artifact would pass"
        )

        e_ref, f_ref, s_ref = eager.predict_properties(
            self._dataloader(atoms, eager),
            include_forces=True,
            include_stresses=True,
        )
        e_aoti, f_aoti, s_aoti = aoti.predict_properties(
            self._dataloader(atoms, aoti),
            include_forces=True,
            include_stresses=True,
        )

        assert abs(e_ref[0] - e_aoti[0]) / len(atoms) < 1e-4
        np.testing.assert_allclose(f_aoti[0], f_ref[0], atol=1e-3)
        np.testing.assert_allclose(s_aoti[0], s_ref[0], atol=1e-2)
