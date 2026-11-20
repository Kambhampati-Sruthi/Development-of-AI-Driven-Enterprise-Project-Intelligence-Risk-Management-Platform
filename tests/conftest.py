import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _docx_bytes() -> bytes:
    from docx import Document
    d = Document()
    d.add_heading("Payments Portal", 0)
    d.add_paragraph("Project Statement: The objective of this project is to deliver a payments portal for retail customers "
                    "with secure checkout, reporting and an admin console for the operations team.")
    d.add_heading("Outcomes", 1)
    d.add_paragraph("Deliver a secure checkout flow with card and UPI payments.", style="List Bullet")
    d.add_paragraph("Provide an operations dashboard with daily settlement reports.", style="List Bullet")
    d.add_heading("Plan (Week 1-2 | ~8 Hours)", 1)
    for t in ("Design the database schema and API contracts.", "Develop the login module.", "Implement the checkout API.", "Validate the payment flow."):
        d.add_paragraph(t, style="List Bullet")
    d.add_heading("Plan (Week 3-4 | ~10 Hours)", 1)
    d.add_paragraph("Build the settlement report module.", style="List Bullet")
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def _xlsx_bytes() -> bytes:
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Backlog"
    ws.append(["Story ID", "Story Description", "Assigned To", "Status", "Planned Sprint", "Actual Sprint", "Dependency"])
    ws.append(["US-01", "As a user, I want to log in securely", "Ravi", "Completed", "Sprint 1", "Sprint 1", ""])
    ws.append(["US-02", "As a user, I want to pay by card", "Asha", "Blocked", "Sprint 1", "Sprint 2", "US-01"])
    ws.append(["US-03", "As an admin, I want settlement reports", "Meera", "In Progress", "Sprint 2", "Sprint 2", "US-02"])
    wd = wb.create_sheet("Defects")
    wd.append(["Defect ID", "Description", "Type of Defect", "Status", "Assigned To", "Remarks"])
    wd.append(["D-1", "Checkout page crashes on empty cart", "Functional", "Open", "Asha", ""])
    wd.append(["D-2", "Typo on login screen", "UI", "Closed", "Ravi", "fixed and verified"])
    wt = wb.create_sheet("UT")
    wt.append(["Test Case Name", "Expected Result", "Actual Result"])
    wt.append(["Login with valid user", "User logged in", "User logged in as expected"])
    wt.append(["Pay with expired card", "Error shown", "Payment fails with unhandled exception"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


NOTES = b"""Sprint 4 Meeting Notes - 12 Nov 2026

Blocked: Payment gateway integration is blocked, waiting for credentials from the vendor.
Decision needed: choose between PostgreSQL and MongoDB for the reporting store.
Action: Ravi will finish the login module by 20 Nov 2026.
Risk: Only one developer knows the billing module, a single point of failure.
The release deadline is 15 Dec 2026 and testing has not started.
"""

CSV = b"Task,Owner,Status,Due Date,Priority\nLogin module,Ravi,In Progress,2026-11-20,High\nPayment API,Asha,Blocked,2026-11-05,High\n"


@pytest.fixture(scope="session")
def sample_files():
    return {"brief.docx": _docx_bytes(), "tracker.xlsx": _xlsx_bytes(), "notes.txt": NOTES, "tasks.csv": CSV}


@pytest.fixture()
def kb(sample_files):
    from core.ingestion import parse_file
    from core.rag import KnowledgeBase
    k = KnowledgeBase()
    for n, b in sample_files.items():
        k.add_document(parse_file(n, b))
    k.build_index()
    return k


@pytest.fixture()
def analysis(kb):
    from core.agents import run_pipeline
    from core.health import compute_health
    pipe = run_pipeline(kb)
    return pipe, compute_health(kb, pipe)
