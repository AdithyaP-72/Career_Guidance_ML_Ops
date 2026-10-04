import pytest

from src.data.label import assign_area_family, assign_role, compile_area_map, compile_rules

RULES = compile_rules()
AREAS = compile_area_map()


@pytest.mark.parametrize(
    "title, role",
    [
        ("Senior React Developer", "Frontend Developer"),
        ("Android Developer - Kotlin", "Android Developer"),
        ("Accounts Executive", "Accountant"),
        ("GST Executive", "Tax / GST Executive"),
        ("Staff Nurse (GNM)", "Staff Nurse"),
        ("TGT Mathematics Teacher", "School Teacher"),
        ("Java Full Stack Developer", "Full Stack Developer"),
        ("Javascript Developer", "Frontend Developer"),  # "java" rule must not catch javascript
        ("Solution Architect", None),  # not a building architect
    ],
)
def test_specific_titles(title, role):
    assert assign_role(title, RULES)[0] == role


@pytest.mark.parametrize(
    "title, wrong_role",
    [
        ("Exciting Opportunity - Team Lead Renewals", "Game Developer"),  # "opportunity" contains "unity"
        ("Hiring for International Voice Process", "Recruiter / Talent Acquisition"),
        ("Company Secretary", "Executive Assistant"),
    ],
)
def test_known_traps_stay_fixed(title, wrong_role):
    assert assign_role(title, RULES)[0] != wrong_role


def test_company_secretary_is_legal():
    assert assign_role("Company Secretary", RULES) == ("Company Secretary / Compliance", "Legal")


@pytest.mark.parametrize(
    "area, family",
    [
        ("IT Software - Application Programming , Maintenance", "Software Development"),
        ("IT Software - QA & Testing", "QA & Testing"),
        ("HR , Recruitment , Administration , IR", "HR & Recruitment"),
        ("HR", "HR & Recruitment"),
        ("TV , Films , Production , Broadcasting", "Content & Media"),
        ("Engineering Design , R&D", "Manufacturing & Core Engg"),
        ("Beauty/Fitness/Spa Services", None),
    ],
)
def test_functional_area_map(area, family):
    assert assign_area_family(area, AREAS) == family
