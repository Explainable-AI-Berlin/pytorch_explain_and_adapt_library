import os
from diff_cf_ir.debug import register_pdb_hook

if (
    "THESIS_DEBUG" in os.environ
    and os.environ["THESIS_DEBUG"] != "0"
    and os.environ["THESIS_DEBUG"].lower() != "false"
):
    register_pdb_hook()
