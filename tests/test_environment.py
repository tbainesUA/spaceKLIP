"""Test environment variables and configuration."""

import os
import sys
from pathlib import Path

import pytest


class TestEnvironmentVariables:
    """Test required environment variables."""

    def test_crds_environment_variables(self):
        """Test CRDS environment variables are set."""
        # These are typically required for JWST pipeline
        crds_vars = ["CRDS_PATH", "CRDS_SERVER_URL"]

        missing_vars = []
        for var in crds_vars:
            if var not in os.environ:
                missing_vars.append(var)

        if missing_vars:
            pytest.skip(
                f"CRDS environment variables not set: {missing_vars}. "
                "This is expected in some test environments."
            )

    def test_crds_path_exists(self):
        """Test CRDS_PATH points to valid directory if set."""
        if "CRDS_PATH" in os.environ:
            crds_path = Path(os.environ["CRDS_PATH"])
            assert (
                crds_path.exists()
            ), f"CRDS_PATH points to non-existent directory: {crds_path}"

    def test_webbpsf_data_path(self):
        """Test WebbPSF data path environment variable."""
        if "WEBBPSF_PATH" in os.environ:
            webbpsf_path = Path(os.environ["WEBBPSF_PATH"])
            assert (
                webbpsf_path.exists()
            ), f"WEBBPSF_PATH points to non-existent directory: {webbpsf_path}"
        else:
            pytest.skip("WEBBPSF_PATH not set - this is optional")

    def test_pysyn_cdbs_environment(self):
        """Test pysynphot/stsynphot CDBS environment variable."""
        if "PYSYN_CDBS" in os.environ:
            pysyn_path = Path(os.environ["PYSYN_CDBS"])
            assert (
                pysyn_path.exists()
            ), f"PYSYN_CDBS points to non-existent directory: {pysyn_path}"
        else:
            pytest.skip("PYSYN_CDBS not set - this is optional")


class TestResourceFiles:
    """Test that package resource files are accessible."""

    def test_resources_directory_exists(self):
        """Test that resources directory is included in package."""
        import spaceKLIP

        package_dir = Path(spaceKLIP.__file__).parent
        resources_dir = package_dir / "resources"

        assert resources_dir.exists(), "resources directory not found in package"

    @pytest.mark.parametrize(
        "resource_file",
        [
            "crpix_jarron.json",
            "filter_shifts_jarron.json",
            "svo_filter_table.dat",
        ],
    )
    def test_resource_file_exists(self, resource_file):
        """Test that specific resource files exist."""
        import spaceKLIP

        package_dir = Path(spaceKLIP.__file__).parent
        resource_path = package_dir / "resources" / resource_file

        assert resource_path.exists(), f"Resource file {resource_file} not found"

    @pytest.mark.parametrize(
        "resource_dir",
        [
            "miri_bg_masks",
            "PCEs",
            "transmissions",
        ],
    )
    def test_resource_subdirectory_exists(self, resource_dir):
        """Test that resource subdirectories exist."""
        import spaceKLIP

        package_dir = Path(spaceKLIP.__file__).parent
        subdir_path = package_dir / "resources" / resource_dir

        assert subdir_path.exists(), f"Resource subdirectory {resource_dir} not found"

    def test_style_file_exists(self):
        """Test that matplotlib style file exists."""
        import spaceKLIP

        package_dir = Path(spaceKLIP.__file__).parent
        style_file = package_dir / "sk_style.mplstyle"

        assert style_file.exists(), "Matplotlib style file not found"


class TestConfiguration:
    """Test package configuration and setup."""

    def test_logging_configuration(self):
        """Test that logging can be configured."""
        from spaceKLIP import logging_tools

        # Test basic logging setup works
        assert hasattr(logging_tools, "configure_logging") or callable(
            getattr(logging_tools, "setup_logging", None)
        ), "No logging configuration function found"

    def test_package_data_accessible(self):
        """Test that package data is accessible via importlib.resources."""
        try:
            if sys.version_info >= (3, 9):
                from importlib.resources import files

                resources = files("spaceKLIP") / "resources"
            else:
                from importlib.resources import path

                with path("spaceKLIP", "resources") as p:
                    resources = p

            # Just check it doesn't raise an error
            assert resources is not None
        except Exception as e:
            pytest.fail(f"Cannot access package resources: {e}")
