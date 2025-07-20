# filepath: app/core/database.py
"""
Mock in-memory user database for customer context.
"""

USER_DATABASE = {
    "39cea0288bd148b181128f5f21c67728": {
        "business_type": "e-commerce",
        "goals": "increase repeat purchases and customer loyalty",
        "industry": "Retail",
        "company_name": "ShopSmart",
        "contact_email": "owner@shopsmart.com",
    },
    "cdc12004ba104c2fb3c54f17287bfc51": {
        "business_type": "SaaS",
        "goals": "boost trial-to-paid conversion",
        "industry": "Software",
        "company_name": "CloudSuite",
        "contact_email": "ceo@cloudsuite.com",
    },
    "b048a2ff1bb64cc6a04f1656030db4e1": {
        "business_type": "consulting",
        "goals": "generate more qualified leads",
        "industry": "Professional Services",
        "company_name": "GrowthAdvisors",
        "contact_email": "info@growthadvisors.com",
    },
}


def get_customer_context(user_id: str):
    """Retrieve customer context from the mock user database."""
    if not user_id:
        return None
    return USER_DATABASE.get(user_id)
