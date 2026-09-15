"""IAB-based taxonomy v3: 11 labels for political news domain tagging.

Source: IAB Content Taxonomy 3.1 Tier 1 categories from mdonigian/iab-news-classification.
Environment & Climate is extracted from Science articles via keyword filtering.

Everything here is a pure data module -- no I/O, no side effects.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 11-label primary taxonomy (order == label encoding 0..10).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LabelRule:
    name: str
    definition: str
    include: str
    exclude: str
    iab_categories: tuple[str, ...] = ()
    confusion_pairs: tuple[str, ...] = ()


LABEL_RULES: tuple[LabelRule, ...] = (
    LabelRule(
        name="Politics & Government",
        definition="Elections, government, legislation, policy decisions, political figures/parties, foreign policy.",
        include="Elections, government, legislation, policy decisions, political figures/parties, foreign policy.",
        exclude="Market/earnings news -> Economy & Business. War/military operations -> War & Conflicts.",
        iab_categories=("Politics",),
        confusion_pairs=("Economy & Business", "War & Conflicts"),
    ),
    LabelRule(
        name="War & Conflicts",
        definition="Armed conflict, military operations, terrorism, war zones, peace negotiations, defense strategy.",
        include="Armed conflict, military operations, terrorism, war zones, peace negotiations, defense strategy.",
        exclude="Domestic crime/law enforcement -> Crime & Justice. Military spending as policy -> Politics & Government.",
        iab_categories=("War and Conflicts",),
        confusion_pairs=("Crime & Justice", "Politics & Government"),
    ),
    LabelRule(
        name="Crime & Justice",
        definition="Criminal acts, law enforcement, courts, sentencing, investigations, serial crimes, organized crime.",
        include="Criminal acts, law enforcement, courts, sentencing, investigations, serial crimes, organized crime.",
        exclude="International conflict/war -> War & Conflicts. Legal/civil law -> Law & Legal.",
        iab_categories=("Crime",),
        confusion_pairs=("Law & Legal", "War & Conflicts"),
    ),
    LabelRule(
        name="Law & Legal",
        definition="Civil law, lawsuits, legal proceedings, court rulings, regulation, constitutional law.",
        include="Civil law, lawsuits, legal proceedings, court rulings, regulation, constitutional law.",
        exclude="Criminal cases -> Crime & Justice. Government policy/legislation -> Politics & Government.",
        iab_categories=("Law",),
        confusion_pairs=("Crime & Justice", "Politics & Government"),
    ),
    LabelRule(
        name="Economy & Business",
        definition="Companies, markets, finance, the economy, jobs, industry, corporate news, trade.",
        include="Companies, markets, finance, the economy, jobs, industry, corporate news, trade.",
        exclude="Government fiscal policy -> Politics & Government. Personal finance advice -> ignore.",
        iab_categories=("Business and Finance",),
        confusion_pairs=("Politics & Government", "Science & Technology"),
    ),
    LabelRule(
        name="Science & Technology",
        definition="Research findings, space, technology products/companies, engineering, computing, AI, cybersecurity.",
        include="Research findings, space, technology products/companies, engineering, computing, AI, cybersecurity.",
        exclude="Climate/environment science -> Environment & Climate. Tech industry earnings -> Economy & Business.",
        iab_categories=("Technology & Computing",),
        confusion_pairs=("Environment & Climate", "Economy & Business"),
    ),
    LabelRule(
        name="Health",
        definition="Public health, medical conditions, treatments, pharmaceuticals, hospitals, pandemics.",
        include="Public health, medical conditions, treatments, pharmaceuticals, hospitals, pandemics.",
        exclude="Health industry business -> Economy & Business. Healthy lifestyle tips -> ignore.",
        iab_categories=("Medical Health",),
        confusion_pairs=("Science & Technology", "Economy & Business"),
    ),
    LabelRule(
        name="Education",
        definition="K-12/universities, teachers, curricula, student life, admissions, education policy.",
        include="K-12/universities, teachers, curricula, student life, admissions, education policy.",
        exclude="Student debt/finance -> Economy & Business. Education policy debates -> Politics & Government.",
        iab_categories=("Education",),
        confusion_pairs=("Politics & Government", "Economy & Business"),
    ),
    LabelRule(
        name="Disasters & Emergencies",
        definition="Natural disasters, accidents, emergency incidents, crises, rescue operations.",
        include="Natural disasters, accidents, emergency incidents, crises, rescue operations.",
        exclude="War/military attacks -> War & Conflicts. Climate change impacts -> Environment & Climate.",
        iab_categories=("Disasters",),
        confusion_pairs=("War & Conflicts", "Environment & Climate"),
    ),
    LabelRule(
        name="Sports",
        definition="Competitive events, teams, athletes, leagues, results/standings, tournaments.",
        include="Competitive events, teams, athletes, leagues, results/standings, tournaments.",
        exclude="Sports business/finance -> Economy & Business. Athlete legal issues -> Crime & Justice.",
        iab_categories=("Sports",),
        confusion_pairs=("Economy & Business", "Crime & Justice"),
    ),
    LabelRule(
        name="Environment & Climate",
        definition="Climate change, pollution, conservation, wildlife, green energy, environmental policy.",
        include="Climate change, pollution, conservation, wildlife, green energy, environmental policy.",
        exclude="Green-industry business -> Economy & Business. General science -> Science & Technology.",
        iab_categories=(),  # extracted from Science via keywords
        confusion_pairs=("Science & Technology", "Politics & Government"),
    ),
)

LABEL_ORDER: tuple[str, ...] = tuple(r.name for r in LABEL_RULES)

# IAB category -> our label lookup (for direct IAB categories).
IAB_TO_LABEL: dict[str, str] = {}
for _rule in LABEL_RULES:
    for _iab in _rule.iab_categories:
        IAB_TO_LABEL[_iab] = _rule.name

# Environment keywords for extracting Environment articles from Science.
ENVIRONMENT_KEYWORDS = [
    "climate", "environment", "pollution", "carbon", "emissions",
    "green energy", "renewable", "conservation", "wildlife", "fossil fuel",
    "global warming", "deforestation", "biodiversity", "ecosystem",
    "sustainability", "sustainable", "ocean acidification", "sea level",
    "arctic", "antarctic", "amazon rainforest", "coral reef",
    "air quality", "water pollution", "plastic waste", "recycling",
    "electric vehicle", "solar power", "wind power", "nuclear energy",
    "environmental policy", "paris agreement", "cop28", "greenhouse gas",
]


def is_environment_article(text: str) -> bool:
    """Check if a Science article is about Environment & Climate."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in ENVIRONMENT_KEYWORDS)


def lookup_iab(iab_category: str) -> str | None:
    """Return the label for an IAB category, or None if not mapped."""
    return IAB_TO_LABEL.get(iab_category)


def label_index(label: str) -> int:
    return LABEL_ORDER.index(label)
