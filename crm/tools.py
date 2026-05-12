"""
crm/tools.py — LLM-callable tool definitions for the Cirkanime CRM.

Each function is a plain Python callable that Agno can register as a tool.
The agent will call these functions when the user asks CRM-related questions.

Usage:
    from crm.tools import ALL_TOOLS
    agent = Agent(tools=ALL_TOOLS, ...)
"""

import json
from typing import Optional

from crm import db as crm_db


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def tool_add_organisation(
    name: str,
    org_type: str = "",
    city: str = "",
    contact_person: str = "",
    contact_email: str = "",
    contact_phone: str = "",
    activity_tags: str = "",
    notes: str = "",
    potential_value: float = 0,
) -> str:
    """
    Add an organisation (school, camp, municipality, festival, etc.) to the CRM.

    Args:
        name: Organisation name (required).
        org_type: Type — one of: Municipalité, Festival, Camp de jour, École,
                  Parascolaire, Maison des jeunes, Organisme.
        city: City or town.
        contact_person: Name of the contact person.
        contact_email: Email address.
        contact_phone: Phone number.
        activity_tags: Comma-separated activity tags (e.g. "cirque,animation,magie").
        notes: Free-text notes.
        potential_value: Estimated contract value in $CAD.

    Returns:
        JSON confirmation with the new organisation id.
    """
    org_id = crm_db.add_organisation(
        name=name, org_type=org_type, city=city,
        contact_person=contact_person, contact_email=contact_email,
        contact_phone=contact_phone, activity_tags=activity_tags,
        notes=notes, potential_value=potential_value,
    )
    return json.dumps(
        {"status": "ok", "org_id": org_id, "message": f"Organisation '{name}' ajoutée (id={org_id})."},
        ensure_ascii=False,
    )


def tool_find_organisations(
    org_type: str = "",
    city: str = "",
    query: str = "",
) -> str:
    """
    Search organisations by type, city, or free-text name.

    Args:
        org_type: Filter by type (e.g. "Municipalité", "Festival").
        city: Filter by city.
        query: Free-text search on name and notes.

    Returns:
        JSON array of matching organisations.
    """
    results = crm_db.find_organisations(org_type=org_type, city=city, query=query)
    return json.dumps(results, ensure_ascii=False, default=str)


def tool_update_organisation(
    org_id: int,
    name: str = "",
    org_type: str = "",
    city: str = "",
    contact_person: str = "",
    contact_email: str = "",
    contact_phone: str = "",
    activity_tags: str = "",
    notes: str = "",
    potential_value: Optional[float] = None,
) -> str:
    """
    Update fields on an existing organisation.

    Args:
        org_id: ID of the organisation to update (required).
        name: New name (leave empty to keep current).
        org_type: New type.
        city: New city.
        contact_person: New contact person name.
        contact_email: New email.
        contact_phone: New phone.
        activity_tags: New tags.
        notes: New notes.
        potential_value: New potential value in $CAD.

    Returns:
        JSON confirmation.
    """
    fields = {}
    if name:
        fields["name"] = name
    if org_type:
        fields["org_type"] = org_type
    if city:
        fields["city"] = city
    if contact_person:
        fields["contact_person"] = contact_person
    if contact_email:
        fields["contact_email"] = contact_email
    if contact_phone:
        fields["contact_phone"] = contact_phone
    if activity_tags:
        fields["activity_tags"] = activity_tags
    if notes:
        fields["notes"] = notes
    if potential_value is not None:
        fields["potential_value"] = potential_value

    ok = crm_db.update_organisation(org_id, **fields)
    if ok:
        return json.dumps(
            {"status": "ok", "message": f"Organisation {org_id} mise à jour."},
            ensure_ascii=False,
        )
    return json.dumps(
        {"status": "error", "message": "Aucun champ valide à mettre à jour."},
        ensure_ascii=False,
    )


def tool_log_contact(
    org_id: int,
    method: str = "",
    status: str = "",
    summary: str = "",
    follow_up_date: str = "",
    contract_value: float = 0,
    contact_date: str = "",
) -> str:
    """
    Record an interaction with an organisation and set its status.

    Args:
        org_id: ID of the organisation (required).
        method: Contact method — courriel, téléphone, en personne, messenger.
        status: New status — Contacté, Intéressé, Rencontre prévue, À relancer, Refus, Bon potentiel futur.
        summary: Brief summary of the interaction.
        follow_up_date: When to follow up (YYYY-MM-DD).
        contract_value: Contract value in $CAD if applicable.
        contact_date: Date of contact (YYYY-MM-DD, defaults to today).

    Returns:
        JSON confirmation with the new log entry id.
    """
    log_id = crm_db.log_contact(
        org_id=org_id, contact_date=contact_date, method=method,
        status=status, summary=summary, follow_up_date=follow_up_date,
        contract_value=contract_value,
    )
    return json.dumps(
        {"status": "ok", "log_id": log_id, "message": f"Contact enregistré pour org {org_id}."},
        ensure_ascii=False,
    )


def tool_get_history(org_id: int) -> str:
    """
    Get the full contact timeline for an organisation.

    Args:
        org_id: ID of the organisation (required).

    Returns:
        JSON array of contact log entries, newest first.
    """
    history = crm_db.get_history(org_id)
    return json.dumps(history, ensure_ascii=False, default=str)


def tool_get_followups_due() -> str:
    """
    Get all overdue follow-ups (follow_up_date <= today).

    Returns:
        JSON array of follow-up entries with organisation details.
    """
    followups = crm_db.get_followups_due()
    return json.dumps(followups, ensure_ascii=False, default=str)


def tool_add_event(
    event_name: str,
    city: str = "",
    event_type: str = "",
    period: str = "",
    best_contact: str = "",
    org_id: Optional[int] = None,
    notes: str = "",
) -> str:
    """
    Add a community event (festival, fête, marché, etc.) to the CRM.

    Args:
        event_name: Name of the event (required).
        city: City where the event takes place.
        event_type: Type — Festival, Fête de quartier, Marché, etc.
        period: When it happens (e.g. "Juin 2026").
        best_contact: Best time to contact organisers.
        org_id: Link to an existing organisation (optional).
        notes: Free-text notes.

    Returns:
        JSON confirmation with the new event id.
    """
    event_id = crm_db.add_event(
        event_name=event_name, city=city, event_type=event_type,
        period=period, best_contact=best_contact, org_id=org_id, notes=notes,
    )
    return json.dumps(
        {"status": "ok", "event_id": event_id, "message": f"Événement '{event_name}' ajouté."},
        ensure_ascii=False,
    )


def tool_find_events(
    city: str = "",
    event_type: str = "",
    query: str = "",
) -> str:
    """
    Search community events by city, type, or free-text query.

    Args:
        city: Filter by city.
        event_type: Filter by event type.
        query: Free-text search on event name and notes.

    Returns:
        JSON array of matching events.
    """
    results = crm_db.find_events(city=city, event_type=event_type, query=query)
    return json.dumps(results, ensure_ascii=False, default=str)


def tool_get_seasonal_targets(month: Optional[int] = None) -> str:
    """
    Get organisations that should be contacted this month based on seasonal
    contact windows for each org type.

    Seasonal windows:
    - Municipalité: Automne + Hiver
    - Festival: Hiver + Printemps
    - Camp de jour: Janvier–Mars
    - École / Parascolaire: Mai + Novembre
    - Maison des jeunes: Janvier + Septembre
    - Organisme: Automne + Hiver

    Args:
        month: Month number (1-12). Defaults to current month.

    Returns:
        JSON array of organisations to contact.
    """
    targets = crm_db.get_seasonal_targets(month=month)
    return json.dumps(targets, ensure_ascii=False, default=str)


def tool_pipeline_summary() -> str:
    """
    Get a pipeline overview: contacts grouped by status with total $CAD values.

    Returns:
        JSON object with by_status breakdown and totals.
    """
    summary = crm_db.pipeline_summary()
    return json.dumps(summary, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Export all tools as a list for Agno registration
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    tool_add_organisation,
    tool_find_organisations,
    tool_update_organisation,
    tool_log_contact,
    tool_get_history,
    tool_get_followups_due,
    tool_add_event,
    tool_find_events,
    tool_get_seasonal_targets,
    tool_pipeline_summary,
]
