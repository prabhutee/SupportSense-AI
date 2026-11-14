"""
Priority Rules Engine for SupportSense.

DISCLAIMER & DESIGN NOTE:
The BANKING77 benchmark dataset contains 77 fine-grained customer intent labels
('category'), but does NOT include ticket priority or severity annotations.
This module implements a deterministic, configurable business rule engine to assign
operational ticket priorities (Critical, High, Medium, Low).

These mappings and urgency keyword overrides represent domain-driven business rules
for SLA routing and triage, NOT ground-truth supervised labels from the dataset.
No machine learning classifier is trained for priority prediction.
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Default config path
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "config",
    "priority_rules.json"
)

# Priority rank hierarchy (higher value = higher urgency)
PRIORITY_RANKS: Dict[str, int] = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Critical": 4
}


class PriorityEngine:
    """
    Evaluates customer queries and intent classifications to assign operational priorities
    based on explicit intent-to-tier mappings and configurable urgency keyword overrides.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Priority Engine with rules from a JSON config file.
        Falls back to default built-in mappings if the file cannot be loaded.
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.intent_to_priority: Dict[str, str] = {}
        self.tier_descriptions: Dict[str, str] = {}
        self.keyword_overrides: Dict[str, List[str]] = {}
        self._load_configuration()

    def _load_configuration(self) -> None:
        """Loads configuration from JSON or falls back to internal defaults."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Load tiers & build reverse lookup intent -> priority
                tiers = data.get("priority_tiers", {})
                for tier_name, tier_info in tiers.items():
                    self.tier_descriptions[tier_name] = tier_info.get("description", "")
                    for intent in tier_info.get("intents", []):
                        self.intent_to_priority[intent] = tier_name
                
                # Load keyword overrides
                self.keyword_overrides = data.get("keyword_overrides", {})
                logger.info(
                    "PriorityEngine loaded %d mapped intents and %d override tiers from %s",
                    len(self.intent_to_priority),
                    len(self.keyword_overrides),
                    self.config_path
                )
                return
            except Exception as e:
                logger.warning(
                    "Failed to load priority configuration from %s (%s). Using fallback defaults.",
                    self.config_path, e
                )

        self._load_fallback_defaults()

    def _load_fallback_defaults(self) -> None:
        """Fallback hardcoded mappings ensuring all 77 intents are covered."""
        # 8 Critical
        critical = [
            "lost_or_stolen_card", "compromised_card", "card_payment_not_recognised",
            "cash_withdrawal_not_recognised", "direct_debit_payment_not_recognised",
            "lost_or_stolen_phone", "card_swallowed", "terminate_account"
        ]
        # 18 High
        high = [
            "pin_blocked", "passcode_forgotten", "unable_to_verify_identity",
            "failed_transfer", "declined_card_payment", "declined_transfer",
            "declined_cash_withdrawal", "top_up_failed", "top_up_reverted",
            "transfer_not_received_by_recipient", "balance_not_updated_after_bank_transfer",
            "balance_not_updated_after_cheque_or_cash_deposit", "transaction_charged_twice",
            "extra_charge_on_statement", "wrong_amount_of_cash_received",
            "card_not_working", "Refund_not_showing_up", "request_refund"
        ]
        # 28 Medium
        medium = [
            "pending_card_payment", "pending_transfer", "pending_top_up",
            "pending_cash_withdrawal", "cancel_transfer", "reverted_card_payment?",
            "transfer_fee_charged", "card_payment_fee_charged", "cash_withdrawal_charge",
            "top_up_by_card_charge", "top_up_by_bank_transfer_charge", "exchange_charge",
            "card_payment_wrong_exchange_rate", "wrong_exchange_rate_for_cash_withdrawal",
            "verify_top_up", "verify_source_of_funds", "verify_my_identity",
            "why_verify_identity", "beneficiary_not_allowed", "card_arrival",
            "card_delivery_estimate", "card_about_to_expire", "contactless_not_working",
            "virtual_card_not_working", "automatic_top_up", "top_up_limits",
            "disposable_card_limits", "edit_personal_details"
        ]
        # 23 Low
        low = [
            "activate_my_card", "order_physical_card", "get_physical_card",
            "getting_virtual_card", "get_disposable_virtual_card", "getting_spare_card",
            "card_linking", "card_acceptance", "change_pin", "exchange_rate",
            "exchange_via_app", "fiat_currency_support", "supported_cards_and_currencies",
            "apple_pay_or_google_pay", "atm_support", "visa_or_mastercard",
            "topping_up_by_card", "top_up_by_cash_or_cheque", "transfer_timing",
            "transfer_into_account", "receiving_money", "country_support", "age_limit"
        ]

        for intent in critical:
            self.intent_to_priority[intent] = "Critical"
        for intent in high:
            self.intent_to_priority[intent] = "High"
        for intent in medium:
            self.intent_to_priority[intent] = "Medium"
        for intent in low:
            self.intent_to_priority[intent] = "Low"

        self.keyword_overrides = {
            "Critical": ["stolen", "theft", "fraud", "hacked", "police", "unauthorized", "compromised", "scam", "emergency"],
            "High": ["urgent", "urgently", "asap", "immediately", "locked out", "stranded", "critical"]
        }

    def _check_keyword_overrides(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Scans customer text for configured urgency keywords.
        Checks higher tiers first (Critical before High).
        Returns (highest_matching_tier, matched_keyword).
        """
        if not text:
            return None, None

        clean_lower = text.lower()
        # Scan in order of priority severity (Critical, then High)
        for tier in ["Critical", "High"]:
            keywords = self.keyword_overrides.get(tier, [])
            for kw in keywords:
                # Use word-boundary matching to prevent false positives (e.g., 'stand' matching 'stranded')
                pattern = r'\b' + re.escape(kw.lower()) + r'\b'
                if re.search(pattern, clean_lower):
                    return tier, kw

        return None, None

    def evaluate(
        self,
        query_text: str,
        predicted_intent: str,
        confidence_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a query and its predicted intent, applying configurable business logic:
        1. Identifies the base priority tier from the intent mapping.
        2. Scans the query text for urgency keywords.
        3. If an urgency keyword has higher priority than the base intent, escalates.
        4. Compiles a clear, human-readable reason string.

        Args:
            query_text: Raw or preprocessed query string.
            predicted_intent: The predicted BANKING77 intent label.
            confidence_score: Optional classifier confidence score (0.0 to 1.0).

        Returns:
            Dict containing priority, priority_reason, rule_triggered, base_priority,
            and escalation status.
        """
        # 1. Base Priority Lookup
        base_priority = self.intent_to_priority.get(predicted_intent, "Medium")
        base_rank = PRIORITY_RANKS.get(base_priority, 2)

        # 2. Urgency Keyword Scan
        override_tier, matched_keyword = self._check_keyword_overrides(query_text)
        override_rank = PRIORITY_RANKS.get(override_tier, 0) if override_tier else 0

        # 3. Escalation Decision Logic
        if override_tier and override_rank > base_rank:
            final_priority = override_tier
            rule_triggered = "keyword_override"
            priority_reason = (
                f"Urgency Override: Query contains high-urgency keyword '{matched_keyword}', "
                f"escalating priority from {base_priority} to {final_priority}."
            )
            is_escalated = True
        else:
            final_priority = base_priority
            rule_triggered = "intent_mapping"
            if predicted_intent in self.intent_to_priority:
                priority_reason = (
                    f"Business Rule: Intent '{predicted_intent}' is classified in the "
                    f"{final_priority} priority tier."
                )
            else:
                priority_reason = (
                    f"Default Fallback: Unrecognized intent '{predicted_intent}', "
                    f"defaulted to {final_priority} priority."
                )
            is_escalated = False

        # Optional confidence annotation
        if confidence_score is not None and confidence_score < 0.40:
            priority_reason += f" (Note: Low model confidence of {confidence_score:.2f}; flagged for review)"

        return {
            "priority": final_priority,
            "priority_reason": priority_reason,
            "rule_triggered": rule_triggered,
            "base_priority": base_priority,
            "is_escalated": is_escalated,
            "matched_keyword": matched_keyword
        }

    def get_all_mapped_intents(self) -> Dict[str, str]:
        """Returns the full dictionary of 77 intents to their assigned priority tier."""
        return dict(self.intent_to_priority)
