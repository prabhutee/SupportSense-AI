"""
MongoDB Aggregation Pipelines for SupportSense.

Demonstrates real MongoDB aggregation operators ($match, $group, $project, $sort,
$limit, $facet, $cond, $avg, $sum, $multiply, $divide, $dateToString) to power
analytics and executive dashboards directly in the database engine.
"""

from typing import Dict, Any, List
from pymongo.database import Database
from config.db_config import COLLECTION_TICKETS


def get_intent_volume_and_confidence(db: Database, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Aggregation Pipeline 1: Top Intents Breakdown & SLA Tracking.
    Groups by predicted_intent, computes average confidence and open/resolved counts.
    """
    pipeline = [
        {
            "$group": {
                "_id": "$predicted_intent",
                "total_tickets": {"$sum": 1},
                "avg_confidence": {"$avg": "$intent_confidence"},
                "open_tickets": {
                    "$sum": {
                        "$cond": [{"$in": ["$status", ["Open", "In Progress"]]}, 1, 0]
                    }
                },
                "resolved_tickets": {
                    "$sum": {
                        "$cond": [{"$eq": ["$status", "Resolved"]}, 1, 0]
                    }
                }
            }
        },
        {
            "$project": {
                "intent": "$_id",
                "_id": 0,
                "total_tickets": 1,
                "avg_confidence": 1,
                "open_tickets": 1,
                "resolved_tickets": 1,
                "resolution_rate_pct": {
                    "$cond": [
                        {"$gt": ["$total_tickets", 0]},
                        {"$multiply": [{"$divide": ["$resolved_tickets", "$total_tickets"]}, 100]},
                        0.0
                    ]
                }
            }
        },
        {"$sort": {"total_tickets": -1}},
        {"$limit": limit}
    ]
    raw = list(db[COLLECTION_TICKETS].aggregate(pipeline))
    for item in raw:
        if item.get("avg_confidence") is not None:
            item["avg_confidence"] = round(float(item["avg_confidence"]), 4)
        if item.get("resolution_rate_pct") is not None:
            item["resolution_rate_pct"] = round(float(item["resolution_rate_pct"]), 1)
    return raw


def get_priority_sla_breakdown(db: Database) -> List[Dict[str, Any]]:
    """
    Aggregation Pipeline 2: Operational Priority & Resolution Analysis.
    Groups by priority tier and calculates resolution percentages.
    """
    pipeline = [
        {
            "$group": {
                "_id": "$priority",
                "total": {"$sum": 1},
                "open_count": {
                    "$sum": {"$cond": [{"$eq": ["$status", "Open"]}, 1, 0]}
                },
                "in_progress_count": {
                    "$sum": {"$cond": [{"$eq": ["$status", "In Progress"]}, 1, 0]}
                },
                "resolved_count": {
                    "$sum": {"$cond": [{"$eq": ["$status", "Resolved"]}, 1, 0]}
                },
                "manual_review_count": {
                    "$sum": {"$cond": ["$needs_manual_review", 1, 0]}
                }
            }
        },
        {
            "$project": {
                "priority": "$_id",
                "_id": 0,
                "total": 1,
                "open_count": 1,
                "in_progress_count": 1,
                "resolved_count": 1,
                "manual_review_count": 1,
                "resolution_rate_pct": {
                    "$cond": [
                        {"$gt": ["$total", 0]},
                        {"$multiply": [{"$divide": ["$resolved_count", "$total"]}, 100]},
                        0.0
                    ]
                }
            }
        },
        # Sort Critical first, then High, Medium, Low
        {
            "$addFields": {
                "sort_rank": {
                    "$switch": {
                        "branches": [
                            {"case": {"$eq": ["$priority", "Critical"]}, "then": 1},
                            {"case": {"$eq": ["$priority", "High"]}, "then": 2},
                            {"case": {"$eq": ["$priority", "Medium"]}, "then": 3},
                            {"case": {"$eq": ["$priority", "Low"]}, "then": 4}
                        ],
                        "default": 5
                    }
                }
            }
        },
        {"$sort": {"sort_rank": 1}},
        {"$project": {"sort_rank": 0}}
    ]
    raw = list(db[COLLECTION_TICKETS].aggregate(pipeline))
    for item in raw:
        if item.get("resolution_rate_pct") is not None:
            item["resolution_rate_pct"] = round(float(item["resolution_rate_pct"]), 1)
    return raw


def get_model_performance_in_production(db: Database) -> List[Dict[str, Any]]:
    """
    Aggregation Pipeline 3: Comparative Classifier Tracking in Live Queue.
    Groups by model_used, comparing ticket throughput, confidence, and manual review flags.
    """
    pipeline = [
        {
            "$group": {
                "_id": "$model_used",
                "ticket_count": {"$sum": 1},
                "mean_confidence": {"$avg": "$intent_confidence"},
                "manual_review_rate_pct": {
                    "$avg": {"$cond": ["$needs_manual_review", 100.0, 0.0]}
                }
            }
        },
        {
            "$project": {
                "model_name": "$_id",
                "_id": 0,
                "ticket_count": 1,
                "mean_confidence": 1,
                "manual_review_rate_pct": 1
            }
        },
        {"$sort": {"ticket_count": -1}}
    ]
    raw = list(db[COLLECTION_TICKETS].aggregate(pipeline))
    for item in raw:
        if item.get("mean_confidence") is not None:
            item["mean_confidence"] = round(float(item["mean_confidence"]), 4)
        if item.get("manual_review_rate_pct") is not None:
            item["manual_review_rate_pct"] = round(float(item["manual_review_rate_pct"]), 2)
    return raw


def get_executive_kpis(db: Database) -> Dict[str, Any]:
    """
    Aggregation Pipeline 4: High-Performance Multi-Faceted Executive KPI Calculation.
    Uses $facet to compute total volume, active queue, resolution rates, and critical issues
    in a single database round-trip.
    """
    pipeline = [
        {
            "$facet": {
                "overview": [
                    {
                        "$group": {
                            "_id": None,
                            "total_tickets": {"$sum": 1},
                            "avg_confidence": {"$avg": "$intent_confidence"},
                            "open_tickets": {
                                "$sum": {"$cond": [{"$in": ["$status", ["Open", "In Progress"]]}, 1, 0]}
                            },
                            "resolved_tickets": {
                                "$sum": {"$cond": [{"$eq": ["$status", "Resolved"]}, 1, 0]}
                            },
                            "manual_review_tickets": {
                                "$sum": {"$cond": ["$needs_manual_review", 1, 0]}
                            }
                        }
                    }
                ],
                "critical_count": [
                    {"$match": {"priority": "Critical"}},
                    {"$count": "count"}
                ]
            }
        }
    ]

    result = list(db[COLLECTION_TICKETS].aggregate(pipeline))
    if not result:
        return {}

    facet = result[0]
    overview = facet["overview"][0] if facet["overview"] else {
        "total_tickets": 0, "avg_confidence": 0.0, "open_tickets": 0,
        "resolved_tickets": 0, "manual_review_tickets": 0
    }
    crit_count = facet["critical_count"][0]["count"] if facet["critical_count"] else 0

    total = overview["total_tickets"]
    resolved = overview["resolved_tickets"]
    res_rate = round((resolved / total * 100), 1) if total > 0 else 0.0

    return {
        "total_tickets": total,
        "open_tickets": overview["open_tickets"],
        "resolved_tickets": resolved,
        "resolution_rate_pct": res_rate,
        "avg_confidence": round(float(overview.get("avg_confidence") or 0.0), 4),
        "critical_tickets": crit_count,
        "manual_review_tickets": overview["manual_review_tickets"]
    }


def get_daily_volume_trend(db: Database) -> List[Dict[str, Any]]:
    """
    Aggregation Pipeline 5: Date-Truncated Volume Trends.
    Uses $dateToString to group tickets by calendar day and priority.
    """
    pipeline = [
        {
            "$project": {
                "day_str": {
                    "$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}
                },
                "priority": 1,
                "status": 1
            }
        },
        {
            "$group": {
                "_id": {"date": "$day_str", "priority": "$priority"},
                "count": {"$sum": 1}
            }
        },
        {
            "$project": {
                "date": "$_id.date",
                "priority": "$_id.priority",
                "count": 1,
                "_id": 0
            }
        },
        {"$sort": {"date": 1, "priority": 1}}
    ]
    return list(db[COLLECTION_TICKETS].aggregate(pipeline))
