# Betting live on Betfair: logistics

Researched 2026-11-14 from Betfair's developer docs and support pages (several
read through search extracts or Wayback Machine copies because the live pages
blocked automated access). UNCONFIRMED marks anything not confirmed from an
official source. Check the starred items yourself before relying on them.

## Costs

- **Live application key: £499 one-off, non-refundable** (current Application
  Keys page, dated Nov 2026; older official pages still say £299). *Confirm
  with Betfair Developer Support.*
- **Delayed key: free.** Real exchange (there is no sandbox), prices 1–180 s
  late, only the top 3 price levels, Stream API capped at 2 connections and
  200 markets. Whether it can place real bets is contradictory across official
  pages: test it.
- **Commission:** charged on net winnings per market, nothing on a losing
  market. Base rate about 5% for cricket (UNCONFIRMED; one source says 3–4%
  for internationals). Each market's Rules tab shows the actual rate.
- **Expert Fee** (replaced the Premium Charge in Jan 2025): only above £25,000
  gross profit over 52 active weeks (20% from £25k–£100k, 40% above). Not
  relevant at small stakes.
- **Transaction charges:** only above 5,000 transactions in an hour. Not
  relevant.
- **Historical data:** Advanced £49/month or £499/year, PRO £150/month or
  £1,499/year (cricket). PRO has the full order book, needed to backtest real
  prices or market making.
- **Minimum stake: £1** (line markets trade at 2.0, so £1 is the floor).

## How the API works

- **Keys:** one delayed and one live key per account, created with
  `createDeveloperAppKeys` (Accounts API demo tool). The live key needs a fully
  verified account and an application at developer.betfair.com ("For My
  Personal Betting").
- **Login:** non-interactive certificate login is the usual choice for a bot.
  Every request sends `X-Authentication` (session token) and `X-Application`
  (app key). UK sessions last 24 hours; extend with the keepAlive endpoint.
  Login is capped at 100 per minute.
- **Operations** (`https://api.betfair.com/exchange/betting/json-rpc/v1`):
  `listMarketCatalogue` to find the 6-over line market (marketType
  `1_INNING_6_OVR_LINE`); `listMarketBook` for best back/lay prices and sizes,
  `totalMatched`, last traded, status and in-play flag; `placeOrders` /
  `cancelOrders`; `listCurrentOrders` / `listClearedOrders` for open and settled
  bets.
- **Stream API:** a push connection (`stream-api.betfair.com:443`) with live
  price and order updates, same key and session.
- **Request weight limit:** 200 points per market-data request (best offers
  costs 5 per market, etc.); over it returns `TOO_MUCH_DATA`.
- **Python:** `betfairlightweight` (v2.24.0, Sep 2026, supports Python
  3.9–3.14, includes streaming); `flumine` is the bot framework built on it.
- **Bots** are allowed for personal betting (the live key is "for personal
  betting purposes only").

## Line market rules (the parts that matter)

- **The price is always 2.0; the line moves.** You choose the run line.
- **In the API, BACK = UNDER and LAY = OVER.** Backing wins if the result is
  below the line; laying wins if it is above. Exactly on the line returns
  stakes. On the website the buttons say Over/Under. Getting this backwards
  bets the opposite side.
- The line goes in the order's `price` field, in steps of the market's
  `lineInterval` within `lineMinUnit`–`lineMaxUnit`; an invalid line returns
  `INVALID_ODDS`. A better available line is given automatically.
- **Unmatched bets are cancelled at the off unless kept.** Use
  `persistenceType=PERSIST` only if you want them to stay in-play; with `LAPSE`
  an unmatched pre-match order disappears at the first ball.
- Settles after the 6th over is completed; no Cash Out.
- **Settlement details for the 6-over line** (wides and no-balls, all out
  before 6 overs, rain, overs reduced before the start): UNCONFIRMED. Read the
  exact text from `listMarketCatalogue` with `MARKET_DESCRIPTION`
  (`description.rules`) for one market before betting. Our backtest settles on
  runs at 6 overs including extras and drops rain-shortened innings.
- In-play bet delay: read `betDelay` from the market (third parties say 1–9 s
  in general). Whether the market suspends on wickets or boundaries:
  UNCONFIRMED.
- **Liquidity:** no published figures for 6-over line markets. Match Odds is
  far bigger (the £8m–£15m totals seen on the event page are Match Odds).
  Check `totalMatched` and the available amounts on the line market itself.

## Data needed at the toss

The model needs, between the toss and the first ball: both confirmed XIs, the
toss decision, the venue, the start time and a weather forecast.

- **Cricsheet** (training history) has no stated update lag; recent checks
  suggest anywhere from 2 to about 11 days. Matches not yet on Cricsheet are
  missing from form and recent-scoring features. Its `people.csv` maps
  Cricsheet player IDs to ESPNcricinfo, Cricbuzz and other site IDs, and
  `names.csv` holds alternative spellings: use these to turn team-sheet names
  into Cricsheet IDs.
- **XIs and toss:** typing them in by hand is free and fine for a few matches
  a day. Paid APIs: Sportmonks €29/month (26 leagues, includes XIs and toss
  fields, 14-day trial that charges unless cancelled); CricketData.org free tier
  (100 calls/day, toss and XI fields UNCONFIRMED); Entity Sport $150–250/month;
  Roanuz about €55–140/month (UNCONFIRMED).
- **Scraping ESPNcricinfo or Cricbuzz:** likely against their terms (Disney's
  terms ban scripts and scraping); avoid.
- **Weather forecast:** Open-Meteo forecast API, free for non-commercial use
  (10,000 calls/day), hourly at any latitude/longitude, credit required.

## Updating every ball (in-play) — researched 2026-11-14

Betfair prices are not the blocker: the same API key streams the line market
live while play is on. The blockers are:

- **Speed of the ball-by-ball feed.** TV and most score feeds run several
  seconds behind the ground (industry figures: 5–8 s for streams). Traders with
  faster data will match against any price we leave up after a boundary or
  wicket. Betfair's in-play bet delay (read `betDelay`; reported as 1–12 s)
  also applies to our orders.
- **No way to backtest yet.** Our Basic Betfair data has in-play prices only
  about once a minute, and Cricsheet has no ball times, so a price can't be
  matched to the ball it followed. PRO data (£150/month) has the full order
  book but still no ball times. In-play would need its own recorded data:
  ball times from a live feed plus the Betfair stream, collected for weeks.
- **The model side is easy.** The simulator can restart from the current
  score, wickets, balls bowled and batters at the crease.

Costs: the £499 live key; a fast live feed (Sportmonks €29/month is cheap
but its update speed is UNCONFIRMED; faster professional feeds cost far more);
PRO data optional. Do in-play only after the pre-match edge holds up in paper
trading.

**XIs from X/Twitter:** X's API is pay-per-use in 2026 (about $0.005 per post
read, no free tier), and team-sheet tweets need parsing into Cricsheet IDs.
Typing two XIs takes about two minutes; not worth automating yet.

## Suggested order

1. Build the live pricing script (XIs, toss, venue, forecast in; mixed-model
   probability and fair line out).
2. Read one market's `description.rules` and check the settlement matches the
   backtest.
3. Paper-trade 50–100 matches, recording the model's price and the best line
   and amount available at the toss (the free delayed key or the website is
   enough).
4. Only then decide on the £499 live key and real stakes.
