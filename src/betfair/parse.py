"""Low-level parser for Betfair Historical Data ("Basic" tier) market files.

Pure module: takes a filesystem path, returns dataclasses. No config import,
no global paths, no I/O beyond the file it is handed. Everything that needs
to know where the archive lives belongs in :mod:`betfair.manifest`.

Archive shape (verified against the real download)
--------------------------------------------------
``data/raw/betfair/BASIC/<YEAR>/<Mon>/<D>/<eventId>/`` holds

* one ``{eventId}.bz2`` **summary** file -- the concatenation of every
  ``marketDefinition`` update seen across *all* markets of that event. It
  carries no ``rc`` price ticks, so it is far cheaper to scan than opening
  every market file, and it is what the manifest scan reads.
* one ``1.{marketId}.bz2`` per market -- the full bz2-compressed,
  newline-delimited-JSON Stream-API ``{"op":"mcm"}`` message log.

**The folder date is not the market date.** A market's ``marketTime`` can
differ from its archive folder by days (rescheduled fixtures, timezone
rollover, late-added markets). Always date a market from ``marketTime``.

Message shapes
--------------
Market-definition message (first message of a market file, repeated whenever
the definition changes -- runners added/removed, status transitions,
settlement)::

    {"op":"mcm","clk":"...","pt":1711915800000,"mc":[{
        "id":"1.226774974",
        "marketDefinition":{
            "eventId":"33144763",
            "eventName":"Mumbai Indians v Rajasthan Royals",
            "marketType":"1ST_INNINGS_RUNS",
            "bettingType":"ASIAN_HANDICAP_SINGLE_LINE",
            "marketTime":"2024-04-01T14:00:00.000Z",
            "countryCode":"GB",
            "numberOfWinners":0,
            "status":"OPEN",              # OPEN | SUSPENDED | CLOSED
            "runners":[{"id":266950,"hc":-30.0,"name":"30 Runs or more",
                        "status":"WINNER"}, ...]}}]}

Runner-change (price) message -- the bulk of a market file::

    {"op":"mcm","clk":"...","pt":1711978102880,
     "mc":[{"id":"1.226774974",
            "rc":[{"ltp":5.03,"id":269570,"hc":-225.0},
                  {"ltp":2.9,"id":266618,"hc":-200.0}]}]}

``hc`` IS LOAD-BEARING AND MUST NOT BE DROPPED
----------------------------------------------
The previous build's parser read only ``id`` and ``ltp`` from ``rc`` and
silently discarded ``hc``. On a handicap ladder the ``hc`` field *is* the
content of the bet: runner 269570 is not "some runner", it is the binary
"225 Runs or more" rung. Dropping it throws away the market. This parser
keeps ``hc`` on both the runner definition and every individual tick.

Betting types and what ``ltp`` means (verified)
-----------------------------------------------
* ``LINE`` -- ``1_INNING_{6,10,15,20}_OVR_LINE`` and friends. Exactly ONE
  runner, named ``"Total Runs"``, ``id == 15316``, ``numberOfWinners == 0``,
  no ``hc``. **``ltp`` is the traded LINE IN RUNS, not odds** (53.5 at six
  overs, 188.5 at twenty overs on the same match). Never feed a LINE ``ltp``
  to anything in :mod:`betfair.odds`.
* ``ASIAN_HANDICAP_SINGLE_LINE`` -- ``1ST_INNINGS_RUNS``, up to 55 runners,
  each a binary rung ``"X Runs or more"`` with ``hc == -X``, settling
  ``WINNER``/``LOSER`` individually. Here ``ltp`` IS genuine decimal odds,
  and the rung ladder is the market's implied CDF of the first-innings total.
* ``ODDS`` -- ``MATCH_ODDS`` etc. ``ltp`` is decimal odds.

Ladder LTPs are **non-monotone** across rungs -- a verified example has the
"190+" rung last traded at 90.0 while "195+" traded at 11.54 -- because each
rung carries its own asynchronous last-traded price from its own moment in
time. That is not corruption and this layer does NOT repair it. Store raw;
any monotonicity enforcement belongs in a later modelling stage where the
choice of repair is explicit.

Price semantics
---------------
Basic tier is **last-traded-price only**: no back/lay quotes, no sizes, no
traded volume. A price backed by a single 2-pound trade is indistinguishable
from one backed by 50k. If a runner never traded, there is no price for it
and this module reports ``None`` -- it never substitutes an opening price, a
neighbouring runner's price, or any other synthetic fallback.
"""

from __future__ import annotations

import bz2
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

# ``{eventId}.bz2`` -- the per-event summary file, as opposed to a
# ``1.{marketId}.bz2`` per-market file.
SUMMARY_FILENAME_RE = re.compile(r"^\d+\.bz2$")


def parse_betfair_timestamp(ts: str) -> datetime:
    """Parse a Betfair ISO-8601 string (``"2024-04-01T14:00:00.000Z"``) to UTC."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def pt_to_datetime(pt_ms: int) -> datetime:
    """Convert a stream ``pt`` field (epoch milliseconds, UTC) to a UTC datetime."""
    return datetime.fromtimestamp(pt_ms / 1000.0, tz=timezone.utc)


@dataclass(frozen=True)
class Runner:
    """One runner as declared in a ``marketDefinition``.

    ``hc`` is the handicap. It is ``None`` on ODDS and LINE markets and the
    negated run threshold on a handicap ladder (``-30.0`` for the
    "30 Runs or more" rung).
    """

    id: int
    name: str | None = None
    hc: float | None = None
    status: str | None = None  # ACTIVE | REMOVED | WINNER | LOSER
    sort_priority: int | None = None


@dataclass
class MarketDefinition:
    """Last-seen ``marketDefinition`` for one market."""

    market_id: str
    event_id: str | None = None
    event_name: str | None = None
    market_type: str | None = None
    betting_type: str | None = None
    market_name: str | None = None
    market_time: datetime | None = None
    country_code: str | None = None
    number_of_winners: int | None = None
    status: str | None = None
    # Runners as of the LAST definition update seen, in sortPriority order.
    # A market's runner list can grow mid-file (e.g. once an XI is named), so
    # this is the union of every runner ever declared, carrying each one's
    # most recent name / hc / status.
    runners: list[Runner] = field(default_factory=list)
    # True if marketTime was revised part-way through the file.
    market_time_changed: bool = False

    @property
    def n_runners(self) -> int:
        return len(self.runners)

    def runner_by_id(self, runner_id: int) -> Runner | None:
        for r in self.runners:
            if r.id == runner_id:
                return r
        return None


@dataclass(frozen=True)
class Tick:
    """One runner-change price update.

    ``ts`` comes from the enclosing message's ``pt`` (epoch ms, UTC). ``ltp``
    is the last traded price -- decimal odds on ODDS / handicap markets, and
    the traded line IN RUNS on ``LINE`` markets. ``hc`` is carried through
    verbatim and is ``None`` when the message did not supply one.
    """

    ts: datetime
    runner_id: int
    ltp: float
    hc: float | None = None


@dataclass
class ParsedMarket:
    """A fully parsed per-market file: its definition plus every tick."""

    definition: MarketDefinition
    ticks: list[Tick] = field(default_factory=list)

    def ticks_for(self, runner_id: int) -> list[Tick]:
        """Every tick naming ``runner_id``, in file order (chronological)."""
        return [t for t in self.ticks if t.runner_id == runner_id]


def read_mcm_lines(path):
    """Yield each decoded newline-delimited JSON message from a ``.bz2`` file.

    Undecodable lines are skipped rather than aborting the file: the archive
    is third-party data and a single bad line should not cost a whole market.
    """
    with bz2.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _apply_definition(md: dict, out: MarketDefinition) -> None:
    """Fold one ``marketDefinition`` payload into an accumulating definition."""
    out.event_id = md.get("eventId", out.event_id)
    out.event_name = md.get("eventName", out.event_name)
    out.market_type = md.get("marketType", out.market_type)
    out.betting_type = md.get("bettingType", out.betting_type)
    out.market_name = md.get("name", out.market_name)
    out.country_code = md.get("countryCode", out.country_code)
    out.number_of_winners = md.get("numberOfWinners", out.number_of_winners)
    out.status = md.get("status", out.status)

    mt_raw = md.get("marketTime")
    if mt_raw:
        parsed = parse_betfair_timestamp(mt_raw)
        if out.market_time is not None and parsed != out.market_time:
            out.market_time_changed = True
        out.market_time = parsed

    by_id = {r.id: r for r in out.runners}
    order = [r.id for r in out.runners]
    for raw in md.get("runners") or []:
        rid = raw.get("id")
        if rid is None:
            continue
        prev = by_id.get(rid)
        merged = Runner(
            id=rid,
            name=raw.get("name", prev.name if prev else None),
            # hc is the whole content of a ladder rung -- carry it, and do not
            # let a later definition update that omits it wipe an earlier one.
            hc=raw.get("hc", prev.hc if prev else None),
            status=raw.get("status", prev.status if prev else None),
            sort_priority=raw.get("sortPriority", prev.sort_priority if prev else None),
        )
        by_id[rid] = merged
        if prev is None:
            order.append(rid)
    out.runners = sorted(
        (by_id[rid] for rid in order),
        key=lambda r: (r.sort_priority if r.sort_priority is not None else 1 << 30, r.id),
    )


def parse_market_file(path) -> ParsedMarket:
    """Parse one per-market ``1.{marketId}.bz2`` file.

    Returns the market's last-seen definition together with the complete
    runner-change tick stream, ``hc`` and message timestamp included, in file
    (chronological) order. Messages naming a market id other than the file's
    own are skipped rather than mixed in.
    """
    market_id = None
    definition = None
    ticks = []

    for msg in read_mcm_lines(path):
        pt_ms = msg.get("pt")
        for mc in msg.get("mc") or []:
            mid = mc.get("id")
            if market_id is None and mid is not None:
                market_id = mid
                definition = MarketDefinition(market_id=mid)
            if mid is not None and market_id is not None and mid != market_id:
                continue
            if definition is None:
                continue

            md = mc.get("marketDefinition")
            if md is not None:
                _apply_definition(md, definition)

            rc = mc.get("rc")
            if not rc or pt_ms is None:
                continue
            ts = pt_to_datetime(pt_ms)
            for r in rc:
                rid = r.get("id")
                ltp = r.get("ltp")
                if rid is None or ltp is None:
                    continue
                ticks.append(Tick(ts=ts, runner_id=rid, ltp=float(ltp), hc=r.get("hc")))

    if definition is None:
        # No usable message at all: recover the id from the filename
        # ("1.226774966.bz2" -> "1.226774966") so the caller still gets a row.
        definition = MarketDefinition(
            market_id=os.path.splitext(os.path.basename(os.fspath(path)))[0]
        )
    return ParsedMarket(definition=definition, ticks=ticks)


def scan_event_summary(path) -> list:
    """Enumerate every market in an event from its ``{eventId}.bz2`` summary.

    The summary concatenates all ``marketDefinition`` updates for the event,
    so definitions are folded per market id and the accumulated (i.e. final)
    state is returned -- reflecting late corrections to ``marketTime``,
    runner lists and settlement status. Carries no price ticks by design.
    """
    acc = {}
    for msg in read_mcm_lines(path):
        for mc in msg.get("mc") or []:
            mid = mc.get("id")
            md = mc.get("marketDefinition")
            if mid is None or md is None:
                continue
            entry = acc.get(mid)
            if entry is None:
                entry = acc[mid] = MarketDefinition(market_id=mid)
            _apply_definition(md, entry)
    return list(acc.values())


def event_summary_path(event_dir) -> str:
    """Path of the ``{eventId}.bz2`` summary inside an event directory."""
    event_dir = os.fspath(event_dir)
    event_id = os.path.basename(os.path.normpath(event_dir))
    return os.path.join(event_dir, f"{event_id}.bz2")


def market_file_name(market_id: str) -> str:
    """File name of a per-market log given its market id (``1.226774966``)."""
    return f"{market_id}.bz2"


def last_before(ticks, cutoff):
    """Last tick with ``ts`` STRICTLY before ``cutoff``, or ``None``.

    Strict: a tick landing exactly on ``cutoff`` (the scheduled off) is NOT
    pre-match. There is deliberately no fallback -- if nothing traded before
    the off, the answer is "no pre-match price", not the first in-play tick
    and not the opening price.
    """
    best = None
    for t in ticks:
        if t.ts < cutoff and (best is None or t.ts >= best.ts):
            best = t
    return best
