"""Test basic package installation and imports."""

import importlib

import pytest


class TestPackageInstallation:
    """Test that spaceKLIP is properly installed and importable."""

    def test_package_importable(self):
        """Test that spaceKLIP can be imported."""
        import spaceKLIP

        assert spaceKLIP is not None

    def test_package_has_version(self):
        """Test that package version is accessible."""
        import spaceKLIP

        assert hasattr(spaceKLIP, "__version__")
        assert isinstance(spaceKLIP.__version__, str)
        assert len(spaceKLIP.__version__) > 0

    def test_version_format(self):
        """Test that version follows semantic versioning."""
        import re

        import spaceKLIP

        # Match semantic versioning pattern (X.Y.Z or X.Y.Z.devN, etc.)
        version_pattern = r"^\d+\.\d+\.\d+.*$"
        assert re.match(version_pattern, spaceKLIP.__version__)


class TestCoreModuleImports:
    """Test that all core modules can be imported."""

    @pytest.mark.parametrize(
        "module_name",
        [
            "spaceKLIP.analysistools",
            "spaceKLIP.coron1pipeline",
            "spaceKLIP.coron2pipeline",
            "spaceKLIP.coron3pipeline",
            "spaceKLIP.database",
            "spaceKLIP.imagetools",
            "spaceKLIP.psf",
            "spaceKLIP.psflib",
            "spaceKLIP.pyklippipeline",
            "spaceKLIP.utils",
            "spaceKLIP.wcs_utils",
            "spaceKLIP.plotting",
            "spaceKLIP.starphot",
            "spaceKLIP.mast",
        ],
    )
    def test_module_import(self, module_name):
        """Test individual module imports."""
        try:
            module = importlib.import_module(module_name)
            assert module is not None
        except ImportError as e:
            pytest.fail(f"Failed to import {module_name}: {e}")

    def test_xara_subpackage_import(self):
        """Test xara subpackage import."""
        from spaceKLIP.xara import core

        assert core is not None


class TestModuleAttributes:
    """Test that modules expose expected public APIs."""

    def test_main_package_exports(self):
        """Test that main package has expected attributes."""
        import spaceKLIP

        # Check for common attributes that should exist
        expected_attrs = ["__version__", "__name__"]
        for attr in expected_attrs:
            assert hasattr(spaceKLIP, attr), f"Missing attribute: {attr}"

    def test_no_private_exports_in_main(self):
        """Test that main __init__ doesn't export private modules."""
        import spaceKLIP

        # Get all public attributes (not starting with _)
        public_attrs = [attr for attr in dir(spaceKLIP) if not attr.startswith("_")]

        # Should have reasonable number of exports (not everything)
        # This is a sanity check - adjust threshold as needed
        assert (
            len(public_attrs) < 50
        ), "Too many exports - check __init__.py"  # Should have reasonable number of exports (not everything)
        # This is a sanity check - adjust threshold as needed
        assert len(public_attrs) < 50, "Too many exports - check __init__.py"
