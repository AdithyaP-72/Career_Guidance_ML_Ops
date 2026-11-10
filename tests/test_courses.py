"""Unit tests for Objective O3: Learning path course recommendations (src/models/courses.py).

Uses tiny synthetic/made-up course catalogs to verify matching rules, deduplication,
boundary safety, and priority ordering without relying on external files.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.courses import (
    deduplicate_nptel,
    make_skill_pattern,
    map_skills_to_courses,
    parse_coursera_courses,
)


@pytest.fixture
def tiny_nptel_df() -> pd.DataFrame:
    """Tiny synthetic NPTEL/SWAYAM catalog with duplicate runs and various titles."""
    return pd.DataFrame(
        [
            {
                "id": "1",
                "title": "NOC:Introduction to Machine Learning",
                "provider": "NPTEL",
                "url": "https://nptel.ac.in/courses/101",
                "summary": "Short summary",
            },
            {
                "id": "2",
                "title": "Introduction to Machine Learning",
                "provider": "SWAYAM",
                "url": "https://swayam.gov.in/courses/101",
                "summary": (
                    "Comprehensive long summary with details about regression, classification, and neural networks."
                ),
            },
            {
                "id": "3",
                "title": "Introduction to GST",
                "provider": "SWAYAM",
                "url": "https://swayam.gov.in/courses/gst",
                "summary": "Goods and services tax law and compliances.",
            },
            {
                "id": "4",
                "title": "Programming in C++",
                "provider": "NPTEL",
                "url": "https://nptel.ac.in/courses/cpp",
                "summary": "Object oriented programming using C++.",
            },
            {
                "id": "5",
                "title": "Problem Solving through Programming in C",
                "provider": "NPTEL",
                "url": "https://nptel.ac.in/courses/c",
                "summary": "Learn C language fundamentals, pointers, and memory.",
            },
            {
                "id": "6",
                "title": "History of Modern Art (c. 1900-1950)",
                "provider": "SWAYAM",
                "url": "https://swayam.gov.in/courses/art",
                "summary": "History and movements of modern art in early 20th century.",
            },
        ]
    )


@pytest.fixture
def tiny_coursera_df() -> pd.DataFrame:
    """Tiny synthetic Coursera catalog with skill tags."""
    return pd.DataFrame(
        [
            {
                "name": "Machine Learning Specialization",
                "skills": "Machine Learning, Deep Learning, Supervised Learning",
                "url": "https://coursera.org/learn/ml-spec",
            },
            {
                "name": "Front-End Web Development with React",
                "skills": "React, JavaScript, HTML, CSS",
                "url": "https://coursera.org/learn/react-web",
            },
            {
                "name": "Cloud Architecture with AWS",
                "skills": "Amazon Web Services, Cloud Computing",
                "url": "https://coursera.org/learn/aws-cloud",
            },
            {
                "name": "C for Everyone: Fundamentals",
                "skills": "C (Programming Language), Algorithms",
                "url": "https://coursera.org/learn/c-everyone",
            },
        ]
    )


@pytest.fixture
def sample_aliases() -> dict[str, str]:
    """Sample synonym mappings: variant -> canonical."""
    return {
        "react": "react.js",
        "reactjs": "react.js",
        "react js": "react.js",
        "amazon web services": "aws",
        "c (programming language)": "c",
        "goods and services tax": "gst",
    }


def test_deduplicate_nptel(tiny_nptel_df):
    """Test that duplicate runs with NOC: prefix are collapsed, keeping the longest summary."""
    dedup = deduplicate_nptel(tiny_nptel_df)
    # The two "Introduction to Machine Learning" runs should become 1
    ml_courses = dedup[dedup["clean_title"].str.contains("Machine Learning", case=False)]
    assert len(ml_courses) == 1
    assert "Comprehensive long summary" in ml_courses.iloc[0]["clean_summary"]


def test_coursera_tag_normalization(tiny_coursera_df, sample_aliases):
    """Test that Coursera skill tags normalize using aliases."""
    lookup = parse_coursera_courses(tiny_coursera_df, sample_aliases)
    assert "react.js" in lookup
    assert lookup["react.js"][0]["title"] == "Front-End Web Development with React"
    assert "aws" in lookup
    assert lookup["aws"][0]["title"] == "Cloud Architecture with AWS"
    assert "c" in lookup


def test_make_skill_pattern_boundary_safety():
    """Verify regex whole-word boundaries and punctuation safety."""
    p_c = make_skill_pattern("c")
    assert p_c.search("Programming in C")
    assert p_c.search("C Programming")
    assert not p_c.search("Programming in C++")
    assert not p_c.search("C# Developer")
    assert not p_c.search("History (c. 1900-1950)")

    p_cpp = make_skill_pattern("c++")
    assert p_cpp.search("Programming in C++")
    assert not p_cpp.search("Programming in C")

    p_net = make_skill_pattern(".net")
    assert p_net.search("Introduction to .NET Core")
    assert not p_net.search("Network administration")


def test_map_skills_to_courses_prefers_free(tiny_nptel_df, tiny_coursera_df, sample_aliases):
    """Test that free Indian courses (NPTEL/SWAYAM) come first, Coursera as fallback."""
    dedup_nptel = deduplicate_nptel(tiny_nptel_df)
    coursera_lookup = parse_coursera_courses(tiny_coursera_df, sample_aliases)

    params = {
        "max_courses_per_skill": 3,
        "min_nptel_similarity": 0.35,
        "vague_skills": ["core", "development"],
    }

    core_skills = ["machine learning", "react.js", "gst", "core", "unknown_skill_xyz"]
    skill_courses = map_skills_to_courses(
        core_skills, dedup_nptel, coursera_lookup, tiny_coursera_df, sample_aliases, params
    )

    # 1. "machine learning" has both NPTEL and Coursera -> NPTEL must be first
    ml_matches = skill_courses["machine learning"]
    assert len(ml_matches) >= 2
    assert ml_matches[0]["provider"] in ("NPTEL", "SWAYAM")
    assert any(c["provider"] == "Coursera" for c in ml_matches)

    # 2. "react.js" only has Coursera match
    react_matches = skill_courses["react.js"]
    assert len(react_matches) == 1
    assert react_matches[0]["provider"] == "Coursera"

    # 3. "gst" matches NPTEL
    assert len(skill_courses["gst"]) == 1
    assert skill_courses["gst"][0]["provider"] == "SWAYAM"

    # 4. Vague skills return empty list
    assert skill_courses["core"] == []

    # 5. Unknown skill returns empty list
    assert skill_courses["unknown_skill_xyz"] == []


def test_max_courses_per_skill_limit(tiny_nptel_df, sample_aliases):
    """Test that returned course list respects max_courses_per_skill limit."""
    dedup_nptel = deduplicate_nptel(tiny_nptel_df)
    params = {
        "max_courses_per_skill": 1,
        "min_nptel_similarity": 0.35,
        "vague_skills": [],
    }
    skill_courses = map_skills_to_courses(["machine learning"], dedup_nptel, {}, pd.DataFrame(), sample_aliases, params)
    assert len(skill_courses["machine learning"]) == 1
