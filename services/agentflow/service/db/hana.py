"""Legacy SAP HANA helpers.

This module now exists solely to guard against lingering imports. Any attempt to
use these helpers indicates that HANA integration code paths still exist and
should be removed.
"""

raise RuntimeError("SAP HANA integration has been retired from AgentFlow")
