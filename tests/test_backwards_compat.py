"""Tests for backward compatibility with the old import paths."""

import pytest
import warnings


class TestBackwardCompat:
    """Test that old import paths still work with deprecation warnings."""

    def test_src_import_warns(self):
        """Importing from 'src' should emit a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from src import SymanticModel
            assert len(w) >= 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

    def test_src_import_works(self):
        """Importing from 'src' should still provide the correct class."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from src import SymanticModel
            assert SymanticModel is not None

    def test_new_import_path(self):
        """New import path should work without warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from symantic import SymanticModel
            # Filter only DeprecationWarnings from our code
            our_warnings = [x for x in w if "deprecated" in str(x.message).lower() and "src" in str(x.message).lower()]
            assert len(our_warnings) == 0
