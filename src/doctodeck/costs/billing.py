"""Generate, then charge (4.2). Anchor: AnkiX `DocumentsAiController.cs`, the generate action.

    pre-flight check  →  LLM call  →  settle: hold back extras the balance can't cover, then the atomic charge

The pre-flight is advisory: it turns away a request that clearly can't pay before any money is spent on the
provider. It can't promise anything, because other requests may spend the balance while this one waits on the
model. The atomic charge is the only real check; if it fails, the cards aren't returned (402).

A failed LLM call charges nothing (the exception skips the settle). A cancelled one (the client went away
mid-call) charges nothing either. But once the cards exist, we've paid the provider, so the charge must happen
even if the request is cancelled now: `asyncio.shield` runs the settle step as its own task, which a
cancellation of the request doesn't reach. (AnkiX passes `CancellationToken.None` to the deduction.)
"""

import asyncio

from doctodeck.costs.credits import CREDITS_PER_CARD, ingest_credits
from doctodeck.costs.ledger import current_deck
from doctodeck.costs.wallet import Wallet
from doctodeck.models import ApiModel
from doctodeck.pipeline.generator import CardGenerator, source_text, target_cards
from doctodeck.pipeline.models import GenerateRequest, GenerateResult, PipelineOptions

INSUFFICIENT_CREDITS = "INSUFFICIENT_CREDITS"


class InsufficientCredits(Exception):
    """Becomes a 402 INSUFFICIENT_CREDITS."""

    def __init__(self, required: int, balance: int) -> None:
        super().__init__(f"This needs {required} credits and the wallet has {balance}.")
        self.required = required
        self.balance = balance


class GenerationCharge(ApiModel):
    ingest_credits: int
    card_credits: int
    total_credits: int
    cards_held_back: int  # cards generated but not returned, because the balance couldn't cover them
    balance_after: int


class ChargedGenerateResult(GenerateResult):
    """What POST /api/generate returns: the generation, plus what it cost in credits."""

    credits: GenerationCharge


# shield() runs the settle step as a task, and the event loop keeps only weak references to tasks: one nobody
# else holds can be garbage-collected mid-run (the asyncio docs warn about exactly this). Hold them here.
_settling: set[asyncio.Task] = set()


async def generate_charged(
    generator: CardGenerator, wallet: Wallet, request: GenerateRequest, options: PipelineOptions
) -> ChargedGenerateResult:
    _, sent = source_text(request, options)
    ingest = ingest_credits(sent)
    target = target_cards(request, options)

    # Pre-flight: the minimum this request will cost, the ingest fee plus the cards asked for.
    required = ingest + target * CREDITS_PER_CARD
    balance = await wallet.balance()
    if balance < required:
        raise InsufficientCredits(required, balance)

    result = await generator.generate(request)  # LlmError or a cancellation here: nothing is charged

    # The task copies the current context, so the deck id (a ContextVar) is still set inside it.
    task = asyncio.create_task(_settle(wallet, result, ingest, target))
    _settling.add(task)
    task.add_done_callback(_settling.discard)
    return await asyncio.shield(task)


async def _settle(wallet: Wallet, result: GenerateResult, ingest: int, target: int) -> ChargedGenerateResult:
    # The model may return more cards than asked for. Every card returned is paid for; extras the balance can't
    # cover are held back, but never below the number asked for, which the pre-flight said was affordable.
    cards = result.cards
    affordable = max(0, (await wallet.balance() - ingest) // CREDITS_PER_CARD)
    keep = len(cards)
    if len(cards) > target and len(cards) > affordable:
        keep = max(target, affordable)

    total = ingest + keep * CREDITS_PER_CARD
    balance_after = await wallet.charge(total, "generate", reference=None, deck_id=current_deck.get())
    if balance_after is None:
        # Other requests spent the balance while the model was working. The cards are not returned.
        raise InsufficientCredits(total, await wallet.balance())

    return ChargedGenerateResult(
        **dict(result) | {"cards": cards[:keep]},
        credits=GenerationCharge(
            ingest_credits=ingest,
            card_credits=keep * CREDITS_PER_CARD,
            total_credits=total,
            cards_held_back=len(cards) - keep,
            balance_after=balance_after,
        ),
    )
