"""
crm — Cirkanime CRM module for Paramodus (BonsaiChat).

Provides Leo with an AI-powered outreach operating system for
circus/animation event management.

Usage in your agent setup:

    from crm import ALL_TOOLS, start_scheduler

    agent = Agent(tools=ALL_TOOLS, ...)
    start_scheduler(on_reminder_callback=callback)
"""

from crm.tools import ALL_TOOLS          # noqa: F401
from crm.scheduler import start_scheduler  # noqa: F401
from crm.db import init_db as init_crm_db  # noqa: F401
