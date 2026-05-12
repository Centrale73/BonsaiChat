"""
crm/cirkanime_seed.py — One-time seed script with Leo's example data.

Run: python -m crm.cirkanime_seed
"""

from crm.db import init_db, add_organisation, add_event, log_contact


def seed():
    """Populate the CRM with Leo's initial Cirkanime data."""
    init_db()

    # --- Organisations ---
    orgs = {}

    orgs["sherbrooke"] = add_organisation(
        name="Ville de Sherbrooke", org_type="Municipalite",
        city="Sherbrooke", contact_person="Service des loisirs",
        contact_email="loisirs@ville.sherbrooke.qc.ca",
        activity_tags="cirque,animation,evenements municipaux",
        notes="Grande ville, plusieurs arrondissements. Budget culture important.",
        potential_value=5000,
    )
    orgs["magog"] = add_organisation(
        name="Ville de Magog", org_type="Municipalite",
        city="Magog", contact_person="Responsable loisirs",
        activity_tags="cirque,animation,fete",
        notes="Petite ville touristique, fete nationale active.",
        potential_value=3000,
    )
    orgs["stoke"] = add_organisation(
        name="Municipalite de Stoke", org_type="Municipalite",
        city="Stoke", activity_tags="animation,evenement communautaire",
        notes="Petite municipalite rurale, budget limite mais ouverte.",
        potential_value=1500,
    )
    orgs["richmond"] = add_organisation(
        name="Ville de Richmond", org_type="Municipalite",
        city="Richmond", activity_tags="cirque,fete",
        potential_value=2000,
    )
    orgs["windsor"] = add_organisation(
        name="Ville de Windsor", org_type="Municipalite",
        city="Windsor", activity_tags="animation,cirque",
        potential_value=2000,
    )
    orgs["fete_lac"] = add_organisation(
        name="Festival du Lac-Memphremagog", org_type="Festival",
        city="Magog", contact_person="Comite organisateur",
        activity_tags="festival,spectacle,cirque",
        notes="Festival d'ete majeur. Cherche toujours des animateurs.",
        potential_value=4000,
    )
    orgs["fete_sherb"] = add_organisation(
        name="Fete du Lac des Nations", org_type="Festival",
        city="Sherbrooke", activity_tags="festival,animation,cirque",
        notes="Gros evenement estival a Sherbrooke.",
        potential_value=5000,
    )
    orgs["camp_magog"] = add_organisation(
        name="Camp de jour de Magog", org_type="Camp de jour",
        city="Magog", contact_person="Coordination camps",
        activity_tags="camp,cirque,animation,enfants",
        notes="Semaines thematiques disponibles.",
        potential_value=2500,
    )
    orgs["camp_sherb"] = add_organisation(
        name="Camp de jour Sherbrooke", org_type="Camp de jour",
        city="Sherbrooke", activity_tags="camp,cirque,enfants",
        potential_value=3000,
    )
    orgs["ecole_mitchell"] = add_organisation(
        name="Ecole Mitchell-Montcalm", org_type="Ecole",
        city="Sherbrooke", contact_person="Direction",
        activity_tags="parascolaire,cirque,ecole",
        notes="Interesse par activites parascolaires cirque.",
        potential_value=2000,
    )
    orgs["ecole_stoke"] = add_organisation(
        name="Ecole de Stoke", org_type="Ecole",
        city="Stoke", activity_tags="ecole,animation,cirque",
        potential_value=1200,
    )
    orgs["mdj_sherb"] = add_organisation(
        name="Maison des jeunes de Sherbrooke", org_type="Maison des jeunes",
        city="Sherbrooke", activity_tags="jeunes,cirque,animation",
        notes="Partenaire potentiel pour ateliers reguliers.",
        potential_value=1800,
    )
    orgs["mdj_magog"] = add_organisation(
        name="Maison des jeunes de Magog", org_type="Maison des jeunes",
        city="Magog", activity_tags="jeunes,cirque",
        potential_value=1500,
    )
    orgs["carrefour"] = add_organisation(
        name="Carrefour jeunesse-emploi", org_type="Organisme",
        city="Sherbrooke", activity_tags="jeunesse,emploi,atelier",
        notes="Possibilite d'ateliers de cirque comme outil d'insertion.",
        potential_value=2000,
    )

    # --- Events ---
    add_event(
        event_name="Fete nationale - Sherbrooke",
        city="Sherbrooke", event_type="Fete de quartier",
        period="Juin 2026", best_contact="Avril",
        org_id=orgs["sherbrooke"],
        notes="Animation de rue, spectacles. Contacter en avril.",
    )
    add_event(
        event_name="Fete nationale - Magog",
        city="Magog", event_type="Fete de quartier",
        period="Juin 2026", best_contact="Avril",
        org_id=orgs["magog"],
    )
    add_event(
        event_name="Festival du Lac-Memphremagog",
        city="Magog", event_type="Festival",
        period="Juillet 2026", best_contact="Mars-Avril",
        org_id=orgs["fete_lac"],
    )
    add_event(
        event_name="Fete du Lac des Nations",
        city="Sherbrooke", event_type="Festival",
        period="Juillet 2026", best_contact="Fevrier-Mars",
        org_id=orgs["fete_sherb"],
    )
    add_event(
        event_name="Marche de Noel de Magog",
        city="Magog", event_type="Marche",
        period="Decembre 2026", best_contact="Septembre-Octobre",
    )
    add_event(
        event_name="Rendez-vous champetres de Stoke",
        city="Stoke", event_type="Fete communautaire",
        period="Aout 2026", best_contact="Mai-Juin",
        org_id=orgs["stoke"],
    )

    # --- Sample contact logs ---
    log_contact(
        org_id=orgs["sherbrooke"], method="courriel",
        status="Contacte", summary="Courriel envoye au Service des loisirs pour ete 2026.",
        follow_up_date="2026-05-20",
    )
    log_contact(
        org_id=orgs["magog"], method="telephone",
        status="Interesse", summary="Appel avec responsable loisirs. Interesse pour fete nationale.",
        follow_up_date="2026-05-15",
    )
    log_contact(
        org_id=orgs["camp_magog"], method="courriel",
        status="A relancer", summary="Courriel envoye, pas de reponse apres 2 semaines.",
        follow_up_date="2026-05-10",
    )
    log_contact(
        org_id=orgs["fete_lac"], method="messenger",
        status="Rencontre prevue",
        summary="Rencontre prevue le 20 mai pour discuter programmation.",
        follow_up_date="2026-05-20", contract_value=4000,
    )
    log_contact(
        org_id=orgs["ecole_mitchell"], method="en personne",
        status="Bon potentiel futur",
        summary="Visite a l'ecole. Interesse mais budget deja alloue pour cette annee.",
        follow_up_date="2026-09-01",
    )

    print(f"[CRM Seed] Done: {len(orgs)} organisations, 6 events, 5 contact logs.")


if __name__ == "__main__":
    seed()
