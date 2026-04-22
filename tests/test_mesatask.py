"""
Unit tests for MesaTask pure functions.
No Blender or GPU required.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mesatask'))

from instruction_templates import TEMPLATES


def test_templates_top_level_keys():
    expected = {"move_right", "move_left", "move_forward", "move_backward"}
    assert expected.issubset(set(TEMPLATES.keys()))


def test_templates_have_zh_en():
    for key, val in TEMPLATES.items():
        assert "zh" in val, f"Missing 'zh' in TEMPLATES['{key}']"
        assert "en" in val, f"Missing 'en' in TEMPLATES['{key}']"


def test_templates_have_placeholders():
    for key, val in TEMPLATES.items():
        for lang in ("zh", "en"):
            for tpl in val[lang]:
                has_ph = "{obj}" in tpl or "{value}" in tpl
                assert has_ph, \
                    f"No placeholder in TEMPLATES['{key}']['{lang}']: '{tpl}'"


def test_templates_min_variants():
    for key, val in TEMPLATES.items():
        for lang in ("zh", "en"):
            assert len(val[lang]) >= 2, \
                f"Too few variants for TEMPLATES['{key}']['{lang}']: {len(val[lang])}"
