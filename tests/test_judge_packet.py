import re
from pathlib import Path
import pytest


def test_judge_packet_exists_and_size():
    packet_path = Path("reports/JUDGE_PACKET.md")
    assert packet_path.exists(), "reports/JUDGE_PACKET.md does not exist"
    size_kb = packet_path.stat().st_size / 1024
    assert size_kb < 30.0, f"JUDGE_PACKET.md size ({size_kb:.2f} KB) exceeds 30 KB limit"


def test_judge_packet_required_sections():
    packet_path = Path("reports/JUDGE_PACKET.md")
    content = packet_path.read_text(encoding="utf-8")
    required_sections = [
        "Executive Summary",
        "Diagnosis",
        "Dataset",
        "Generalization",
        "Decisions",
    ]
    for section in required_sections:
        assert re.search(section, content, re.IGNORECASE), f"Missing required section: '{section}'"


def test_judge_packet_links():
    packet_path = Path("reports/JUDGE_PACKET.md")
    content = packet_path.read_text(encoding="utf-8")
    
    # Match markdown links: [text](path)
    # Ignore web urls (http/https) and internal anchor links (#...)
    links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
    for text, link in links:
        if link.startswith("http://") or link.startswith("https://") or link.startswith("#"):
            continue
        # Split optional anchor
        file_part = link.split("#")[0]
        if file_part:
            # relative to repo root or reports/
            p1 = Path(file_part)
            p2 = Path("reports") / file_part
            assert p1.exists() or p2.exists(), f"Broken link in JUDGE_PACKET.md: '{link}' (target '{file_part}' not found)"

