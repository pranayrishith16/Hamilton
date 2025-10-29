# tier_config.py
"""
Subscription tier configuration for the application.
Defines features, limits, and pricing for each tier.
"""

TIERS = {
    "free": {
        "name": "Free",
        "description": "Free tier with limited queries",
        "daily_query_limit": 5,
        "monthly_query_limit": 100,
        "price": 0,
        "features": [
            "Basic legal document analysis",
            "Up to 5 queries per day",
            "Community support"
        ],
        "permissions": ["query", "view"]
    },
    "basic": {
        "name": "Basic",
        "description": "Basic tier with medium queries",
        "daily_query_limit": 50,
        "monthly_query_limit": 1000,
        "price": 29,
        "features": [
            "Advanced legal document analysis",
            "Up to 50 queries per day",
            "Email support",
            "Case file management"
        ],
        "permissions": ["query", "view", "upload", "edit_own"]
    },
    "pro": {
        "name": "Pro",
        "description": "Pro tier with high queries",
        "daily_query_limit": 200,
        "monthly_query_limit": 5000,
        "price": 99,
        "features": [
            "Premium legal document analysis",
            "Up to 200 queries per day",
            "Priority email & phone support",
            "Advanced case management",
            "API access",
            "Team collaboration (up to 5 users)"
        ],
        "permissions": ["query", "view", "upload", "edit", "delete_own", "api_access"]
    },
    "enterprise": {
        "name": "Enterprise",
        "description": "Enterprise tier with unlimited queries",
        "daily_query_limit": 1000,
        "monthly_query_limit": None,  # Unlimited
        "price": None,  # Custom pricing
        "features": [
            "Unlimited legal document analysis",
            "Unlimited queries",
            "24/7 dedicated support",
            "Custom integrations",
            "Advanced security",
            "SLA guarantee",
            "Unlimited team members"
        ],
        "permissions": ["*"]  # All permissions
    }
}

def get_tier_config(tier_name: str):
    """Get tier configuration by name"""
    return TIERS.get(tier_name.lower(), TIERS["free"])

def validate_tier(tier_name: str) -> bool:
    """Validate if tier exists"""
    return tier_name.lower() in TIERS

def get_daily_limit(tier_name: str) -> int:
    """Get daily query limit for tier"""
    tier = get_tier_config(tier_name)
    return tier.get("daily_query_limit", 5)

def get_tier_features(tier_name: str) -> list:
    """Get features for tier"""
    tier = get_tier_config(tier_name)
    return tier.get("features", [])

def get_tier_permissions(tier_name: str) -> list:
    """Get permissions for tier"""
    tier = get_tier_config(tier_name)
    return tier.get("permissions", [])
