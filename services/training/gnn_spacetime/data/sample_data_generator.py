"""Generate synthetic narrative data for testing."""

import random
from typing import List, Dict, Any
from datetime import datetime, timedelta

logger = None
try:
    import logging
    logger = logging.getLogger(__name__)
except:
    pass


def generate_synthetic_corporate_merger(
    company_a: str = "CompanyA",
    company_b: str = "CompanyB",
    duration_days: int = 180,
    num_events: int = 20
) -> List[Dict[str, Any]]:
    """Generate synthetic corporate merger narrative.
    
    Args:
        company_a: Name of acquiring company
        company_b: Name of target company
        duration_days: Duration of merger process in days
        num_events: Number of events to generate
        
    Returns:
        List of event dicts
    """
    events = []
    base_time = datetime.now()
    
    # Phase 1: Announcement and initial reactions
    events.append({
        "time": 0.0,
        "source": company_a,
        "target": company_b,
        "event_type": "announces",
        "description": f"{company_a} announces acquisition of {company_b}",
        "source_type": "company",
        "target_type": "company"
    })
    
    events.append({
        "time": 5.0,
        "source": "market_analysts",
        "target": company_a,
        "event_type": "evaluates",
        "description": "Market analysts evaluate merger prospects",
        "source_type": "analyst",
        "target_type": "company"
    })
    
    # Phase 2: Due diligence and negotiations
    for i in range(1, num_events // 2):
        time_offset = 10.0 + i * (duration_days / num_events)
        event_type = random.choice(["negotiates", "evaluates", "discusses"])
        
        events.append({
            "time": time_offset,
            "source": company_a if i % 2 == 0 else company_b,
            "target": company_b if i % 2 == 0 else company_a,
            "event_type": event_type,
            "description": f"{event_type.capitalize()} merger terms",
            "source_type": "company",
            "target_type": "company"
        })
    
    # Phase 3: Challenges emerge
    events.append({
        "time": duration_days * 0.6,
        "source": "cultural_resistance",
        "target": company_a,
        "event_type": "impedes",
        "description": "Cultural resistance emerges in integration",
        "source_type": "factor",
        "target_type": "company"
    })
    
    events.append({
        "time": duration_days * 0.7,
        "source": "regulatory_delays",
        "target": company_a,
        "event_type": "delays",
        "description": "Regulatory approval delays",
        "source_type": "factor",
        "target_type": "company"
    })
    
    # Phase 4: Resolution
    for i in range(num_events // 2, num_events):
        time_offset = duration_days * 0.8 + (i - num_events // 2) * (duration_days * 0.2 / (num_events // 2))
        outcome = random.choice(["succeeds", "struggles", "adapts"])
        
        events.append({
            "time": time_offset,
            "source": company_a,
            "target": company_b,
            "event_type": outcome,
            "description": f"Merger {outcome}",
            "source_type": "company",
            "target_type": "company"
        })
    
    # Sort by time
    events.sort(key=lambda x: x["time"])
    
    if logger:
        logger.info(f"Generated {len(events)} synthetic merger events")
    
    return events


def generate_synthetic_research_discovery(
    researcher: str = "Dr. Smith",
    topic: str = "quantum_computing",
    duration_days: int = 365,
    num_events: int = 15
) -> List[Dict[str, Any]]:
    """Generate synthetic research discovery narrative.
    
    Args:
        researcher: Researcher name
        topic: Research topic
        duration_days: Duration in days
        num_events: Number of events
        
    Returns:
        List of event dicts
    """
    events = []
    
    # Initial research
    events.append({
        "time": 0.0,
        "source": researcher,
        "target": topic,
        "event_type": "researches",
        "description": f"{researcher} begins research on {topic}",
        "source_type": "researcher",
        "target_type": "topic"
    })
    
    # Research milestones
    for i in range(1, num_events - 2):
        time_offset = i * (duration_days / num_events)
        milestone_type = random.choice(["discovers", "publishes", "collaborates"])
        
        events.append({
            "time": time_offset,
            "source": researcher,
            "target": topic,
            "event_type": milestone_type,
            "description": f"{researcher} {milestone_type} progress on {topic}",
            "source_type": "researcher",
            "target_type": "topic"
        })
    
    # Breakthrough
    events.append({
        "time": duration_days * 0.8,
        "source": researcher,
        "target": topic,
        "event_type": "discovers",
        "description": f"{researcher} makes breakthrough discovery",
        "source_type": "researcher",
        "target_type": "topic"
    })
    
    # Publication
    events.append({
        "time": duration_days * 0.9,
        "source": researcher,
        "target": "journal",
        "event_type": "publishes",
        "description": f"{researcher} publishes findings",
        "source_type": "researcher",
        "target_type": "publication"
    })
    
    events.sort(key=lambda x: x["time"])
    
    if logger:
        logger.info(f"Generated {len(events)} synthetic research events")
    
    return events


def generate_synthetic_social_evolution(
    community: str = "TechCommunity",
    duration_days: int = 730,
    num_events: int = 25
) -> List[Dict[str, Any]]:
    """Generate synthetic social community evolution narrative.
    
    Args:
        community: Community name
        duration_days: Duration in days
        num_events: Number of events
        
    Returns:
        List of event dicts
    """
    events = []
    members = [f"Member{i}" for i in range(1, 6)]
    
    # Community formation
    events.append({
        "time": 0.0,
        "source": members[0],
        "target": community,
        "event_type": "founds",
        "description": f"{members[0]} founds {community}",
        "source_type": "person",
        "target_type": "community"
    })
    
    # Member interactions
    for i in range(1, num_events):
        time_offset = i * (duration_days / num_events)
        source = random.choice(members)
        target = random.choice([m for m in members if m != source] + [community])
        interaction = random.choice(["joins", "contributes", "leads", "supports"])
        
        events.append({
            "time": time_offset,
            "source": source,
            "target": target,
            "event_type": interaction,
            "description": f"{source} {interaction} {target}",
            "source_type": "person",
            "target_type": "person" if target in members else "community"
        })
    
    events.sort(key=lambda x: x["time"])
    
    if logger:
        logger.info(f"Generated {len(events)} synthetic social events")
    
    return events


def create_sample_story_themes() -> List[Dict[str, Any]]:
    """Create sample story themes for testing.
    
    Returns:
        List of theme dicts
    """
    return [
        {
            "theme_id": "merger_story",
            "theme": "Corporate merger between Company A and Company B",
            "narrative_type": "explanation"
        },
        {
            "theme_id": "research_story",
            "theme": "Scientific discovery and publication timeline",
            "narrative_type": "prediction"
        },
        {
            "theme_id": "community_story",
            "theme": "Social community evolution and growth",
            "narrative_type": "anomaly_detection"
        }
    ]

