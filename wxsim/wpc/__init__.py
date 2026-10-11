"""WPC-style products.

  rain  - rule-based rainfall field shared by every WPC product (predecessor rain + TC track rain + orography)
  qpf   - rainfall map panels                      python -m wxsim.wpc.qpf <qpf_*.json>
  ero   - Excessive Rainfall Outlook + discussion  python -m wxsim.wpc.ero <ero_*.json>

WPC post-tropical public advisories are built by wxsim.tropical.products when a tc_*.json sets
advisory.center = "WPC" (KWNH header, WPC office line, optional storm-total rainfall list from the field).
"""
