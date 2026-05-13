"""
crm/scheduler.py — Background scheduler for the Cirkanime CRM.

Fires daily follow-up checks and monthly seasonal contact briefings.
Optional Google Calendar integration via OAuth 2.0.
"""

import os, json, threading, time
from datetime import datetime, date, timedelta
from typing import Optional, Callable, Dict, Any
from crm import db as crm_db

GOOGLE_CALENDAR_ENABLED = os.environ.get("GOOGLE_CALENDAR_ENABLED", "").lower() in ("1", "true", "yes")
CHECK_INTERVAL_SECONDS = 3600
DAILY_CHECK_HOUR = 9
MONTHLY_CHECK_DAY = 1

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CREDENTIALS_PATH = os.path.join(_base_dir, "credentials.json")
_TOKEN_PATH = os.path.join(_base_dir, "token.json")
_SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
_calendar_service = None


def _init_google_calendar():
    """Init Google Calendar API via OAuth 2.0. Opens browser on first run."""
    global _calendar_service
    if not GOOGLE_CALENDAR_ENABLED:
        return
    if not os.path.isfile(_CREDENTIALS_PATH):
        print(f"[CRM Scheduler] credentials.json not found at {_CREDENTIALS_PATH}")
        return
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
        creds = None
        if os.path.isfile(_TOKEN_PATH):
            creds = Credentials.from_authorized_user_file(_TOKEN_PATH, _SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(_CREDENTIALS_PATH, _SCOPES)
                creds = flow.run_local_server(port=0)
            with open(_TOKEN_PATH, "w") as f:
                f.write(creds.to_json())
        _calendar_service = build("calendar", "v3", credentials=creds)
        print("[CRM Scheduler] Google Calendar connected.")
    except ImportError:
        print("[CRM Scheduler] pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
    except Exception as e:
        print(f"[CRM Scheduler] Google Calendar init failed: {e}")


def _push_calendar_event(title, description, event_date=None, duration_hours=1):
    """Create a Google Calendar event. Returns True on success."""
    if not _calendar_service:
        return False
    if not event_date:
        event_date = date.today().isoformat()
    try:
        start_dt = datetime.fromisoformat(f"{event_date}T{DAILY_CHECK_HOUR:02d}:00:00")
        end_dt = start_dt + timedelta(hours=duration_hours)
        body = {
            "summary": title, "description": description,
            "start": {"dateTime": start_dt.isoformat(), "timeZone": "America/Montreal"},
            "end": {"dateTime": end_dt.isoformat(), "timeZone": "America/Montreal"},
            "reminders": {"useDefault": False, "overrides": [{"method": "popup", "minutes": 30}]},
        }
        created = _calendar_service.events().insert(calendarId="primary", body=body).execute()
        print(f"[CRM Scheduler] Calendar event: {created.get('htmlLink')}")
        return True
    except Exception as e:
        print(f"[CRM Scheduler] Calendar event failed: {e}")
        return False


def _build_followup_reminder(followups):
    lines = []
    for f in followups:
        org_id = f["org_id"]
        org_name = f.get("org_name", f"Org #{org_id}")
        status = f.get("status", "?")
        due = f.get("follow_up_date", "?")
        lines.append(f"  - {org_name} ({status}) -- relance {due}")
    body = "\n".join(lines) if lines else "Aucun suivi en retard."
    return {"type": "followup_reminder", "title": f"[CRM] {len(followups)} suivi(s) en retard",
            "body": body, "count": len(followups), "data": followups}


def _build_seasonal_reminder(targets, month):
    names = {1:"Janvier",2:"Fevrier",3:"Mars",4:"Avril",5:"Mai",6:"Juin",
             7:"Juillet",8:"Aout",9:"Septembre",10:"Octobre",11:"Novembre",12:"Decembre"}
    mn = names.get(month, str(month))
    by_type = {}
    for t in targets:
        by_type.setdefault(t.get("org_type","Autre"), []).append(t)
    lines = [f"Contacts saisonniers pour {mn}:"]
    for otype, orgs in by_type.items():
        lines.append(f"\n  {otype} ({len(orgs)}):")
        for o in orgs[:5]:
            lines.append(f"    - {o['name']} ({o.get('city','?')})")
        if len(orgs) > 5:
            lines.append(f"    ... et {len(orgs)-5} autres")
    return {"type": "seasonal_reminder", "title": f"[CRM] Briefing saisonnier — {mn}",
            "body": "\n".join(lines), "count": len(targets), "data": targets}


_scheduler_thread = None
_stop_event = threading.Event()
_last_daily_check = None
_last_monthly_check = None


def _scheduler_loop(on_reminder_callback=None):
    global _last_daily_check, _last_monthly_check
    if GOOGLE_CALENDAR_ENABLED:
        _init_google_calendar()
    print("[CRM Scheduler] Started.")

    while not _stop_event.is_set():
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")

        if now.hour >= DAILY_CHECK_HOUR and _last_daily_check != today_str:
            _last_daily_check = today_str
            try:
                followups = crm_db.get_followups_due()
                if followups:
                    payload = _build_followup_reminder(followups)
                    if on_reminder_callback: on_reminder_callback(payload)
                    if GOOGLE_CALENDAR_ENABLED and _calendar_service:
                        _push_calendar_event(payload["title"], payload["body"])
                    print(f"[CRM Scheduler] {len(followups)} follow-up(s) due.")
            except Exception as e:
                print(f"[CRM Scheduler] Daily check error: {e}")

        if now.day == MONTHLY_CHECK_DAY and now.hour >= DAILY_CHECK_HOUR and _last_monthly_check != today_str:
            _last_monthly_check = today_str
            try:
                targets = crm_db.get_seasonal_targets(month=now.month)
                if targets:
                    payload = _build_seasonal_reminder(targets, now.month)
                    if on_reminder_callback: on_reminder_callback(payload)
                    if GOOGLE_CALENDAR_ENABLED and _calendar_service:
                        _push_calendar_event(payload["title"], payload["body"], duration_hours=2)
                    print(f"[CRM Scheduler] {len(targets)} seasonal target(s).")
            except Exception as e:
                print(f"[CRM Scheduler] Monthly check error: {e}")

        _stop_event.wait(CHECK_INTERVAL_SECONDS)
    print("[CRM Scheduler] Stopped.")


def start_scheduler(on_reminder_callback=None):
    """Start the background scheduler thread."""
    global _scheduler_thread
    if _scheduler_thread and _scheduler_thread.is_alive():
        # If toggled on while running, initialize the OAuth flow immediately
        if GOOGLE_CALENDAR_ENABLED and not _calendar_service:
            _init_google_calendar()
        return
    _stop_event.clear()
    _scheduler_thread = threading.Thread(target=_scheduler_loop, args=(on_reminder_callback,), daemon=True, name="crm-scheduler")
    _scheduler_thread.start()


def stop_scheduler():
    """Signal the scheduler to stop."""
    _stop_event.set()


if __name__ == "__main__":
    print("=== CRM Scheduler Demo ===")
    crm_db.init_db()
    followups = crm_db.get_followups_due()
    if followups:
        print(json.dumps(_build_followup_reminder(followups), indent=2, ensure_ascii=False))
    targets = crm_db.get_seasonal_targets()
    if targets:
        print(json.dumps(_build_seasonal_reminder(targets, date.today().month), indent=2, ensure_ascii=False))
    def demo_cb(p): print(f"\n--- {p['title']} ---\n{p['body']}\n---")
    start_scheduler(on_reminder_callback=demo_cb)
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        stop_scheduler()
