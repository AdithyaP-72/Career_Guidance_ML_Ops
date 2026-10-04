import pytest

from src.data.common import load_yaml
from src.data.label import compile_rules, label_title

RULES = compile_rules()
FAMILIES = set(load_yaml("configs/role_rules.yaml")["families"])

GOLDEN = [
    ("Java Full Stack Developer", "Full Stack Developer"),
    ("Sr. Android Developer (Kotlin)", "Android Developer"),
    ("Staff Nurse - ICU", "Staff Nurse"),
    ("Tax Consultant - GST", "Tax / GST Executive"),
    ("SAP FICO Consultant", "SAP Functional Consultant"),
    ("SAP ABAP Developer", "SAP ABAP Developer"),
    ("Salesforce Developer", "Salesforce Developer"),
    ("DevOps Engineer", "DevOps Engineer"),
    ("Telecaller", "Telecaller / Telesales"),
    ("Back Office Executive", "Back Office Executive"),
    ("Design Engineer - Mechanical", "Design Engineer (CAD)"),
    ("Product Manager", "Product Manager"),
    ("Senior Accountant", "Accountant"),
    ("Digital Marketing Executive", "Digital Marketing Executive"),
    ("Graphic Designer", "Graphic Designer"),
    ("Data Scientist", "Data Scientist"),
    ("Customer Care Executive", "Customer Support Executive"),
    ("Company Secretary", "Company Secretary"),
    ("Recruiter - IT", "Recruiter / Talent Acquisition"),
]


@pytest.mark.parametrize("title,role", GOLDEN)
def test_golden_titles(title, role):
    assert label_title(title, RULES)[0] == role


def test_no_substring_traps():
    # "unity" inside "Opportunity", "hiring" in generic ads, "secretary" in "Company Secretary"
    assert label_title("Great Opportunity For Freshers", RULES)[0] != "Game Developer"
    assert label_title("Urgent Hiring", RULES)[0] != "Recruiter / Talent Acquisition"
    assert label_title("Company Secretary", RULES)[0] != "Executive Assistant / Admin"


def test_every_rule_has_valid_family_and_compiles():
    for rx, role, family in RULES:
        assert family in FAMILIES, (role, family)


def test_no_generic_catch_all_labels():
    banned = {"manager", "engineer", "developer", "executive"}
    assert not any(role.lower() in banned for _, role, _ in RULES)
