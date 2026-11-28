"""What a deck costs in credits (4.2). Anchor: AnkiX `AiPricingTiers.cs`.

Credits are the product's price, not the provider's bill (that's the ledger, in dollars). They're charged for
every provider, the fake included, so the wallet can be exercised without a key. Everything is charged when the
cards are generated, because the page shows them in full: charging later, on save, would let a user copy the
preview and never pay. Planning is free here (AnkiX charges 2 credits for it; one charge site keeps this simple).
"""

import math

CREDITS_PER_500_WORDS = 2  # the ingest fee: per started 500 words of text the model is actually sent
CREDITS_PER_CARD = 1


def ingest_credits(sent_text: str) -> int:
    """2 credits per started 500 words of `sent_text`, at least 2. Priced on what was sent, so a document cut at
    `max_source_chars` pays for the part the model read, not the part it never saw."""
    words = len(sent_text.split())
    return max(1, math.ceil(words / 500)) * CREDITS_PER_500_WORDS


def generation_credits(sent_text: str, cards: int) -> int:
    return ingest_credits(sent_text) + cards * CREDITS_PER_CARD
