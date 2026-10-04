import math

from src.utils import load_aliases, normalise_skill, parse_experience, parse_skills


def test_normalise_spacing_and_case():
    assert normalise_skill("  C + + ") == "c++"
    assert normalise_skill("CI / CD") == "ci/cd"
    assert normalise_skill("E - Commerce") == "e-commerce"
    assert normalise_skill(".NET") == ".net"
    assert normalise_skill("English Proficiency (Spoken)") == "english proficiency (spoken)"


def test_aliases_map_variants_to_one_name():
    aliases = load_aliases()
    for variant in ["ReactJS", "react js", "React"]:
        assert normalise_skill(variant, aliases) == "react.js"
    assert normalise_skill("Tally ERP 9", aliases) == "tally"
    assert normalise_skill("c sharp", aliases) == "c#"
    assert normalise_skill("java", aliases) == "java"  # untouched


def test_parse_skills_splits_dedupes_keeps_order():
    aliases = load_aliases()
    assert parse_skills(" Python| SQL |python|", r"\|", aliases) == ["python", "sql"]
    assert parse_skills("React,ReactJS,Node JS", r",", aliases) == ["react.js", "node.js"]
    assert parse_skills(float("nan"), r",") == []


def test_parse_experience():
    assert parse_experience("2 - 5 yrs") == (2.0, 5.0)
    assert parse_experience("0-3 Yrs") == (0.0, 3.0)
    assert all(math.isnan(x) for x in parse_experience("Not mentioned"))
