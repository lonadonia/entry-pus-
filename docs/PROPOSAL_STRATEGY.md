# Proposal strategy — what the 2026-11-04 competitor study showed

The proposal writer (`bot/project_analyzer.py`) and its deterministic checks
(`bot/proposal_engine.py`) were rebuilt from a study of real bids. This page records
what was measured, what it means, and which rule each finding became, so a later change
can be judged against the evidence rather than against taste.

## The data

- **41 projects** this account bid on (30 Oct – 2 Nov 2026), plus 4 that had been deleted.
- **1,230 competitor messages**: the first 30 unsealed bids per project, in the order of the
  project's Proposals tab, with each bidder's rating, reviews, delay after posting and
  full text.
- **41 proposals this account sent**, with their position in the tab.
- Categories: graphic design 9, websites 7, e-commerce 4, data 4, WordPress 3, mobile 3,
  full-stack 3, photo editing 3, software 2, AI 1, game 1, video 1. Six briefs were in
  Spanish or French.

**There is no outcome data** — no record of which bids got a reply or an award, for this
account or for anyone else. Nothing below is a measured conversion effect. Position in the
Proposals tab is not one either: see the next section.

## Position in the list is reputation and speed, not writing

| Proposals-tab position | median reviews | median minutes after posting | greeting line | years-of-experience claim | bullet list |
|---|---|---|---|---|---|
| 1–10 | 444 | 4 | 61% | 21% | 23% |
| 11–30 | 134 | 15 | 63% | 24% | 25% |
| 31+ | 107 | 7 | 63% | 24% | 26% |

Writing traits are nearly identical across positions. What separates the top of the list
is review count and how fast the bid arrived: 53% of competitor bids came within 10
minutes of posting, 69% within an hour. **This account's bids arrived a median 11.7 hours
after posting** (2.4 h – 26 h) and sat around the middle of the list (median 40% down).
A once-a-day discovery run will keep that delay; moving the discovery time earlier, or
running "Run today's discovery now" during the day, is the only lever on it.

## What a typical competitor bid looks like

Median 133 words in 5 short paragraphs. 62% open with a greeting line, which spends the
one or two lines Freelancer shows as a preview; 50% end with a signature; 24% claim years of
experience; 26% use bullet lists; 32% lean on stock AI vocabulary ("seamless", "robust",
"tailored", "ensure"); 39% include links. Only 40% open with something specific to the
brief. The most common endings were "Looking forward to…" (111 bids), "Choose me…" (64) and
"Feel free to…" (27).

A large cluster is a recognisable paraphrase template: "[N years / my skills] make me well
suited to [the project restated]. I can [the client's deliverables listed back]… I would be
glad to… Choose me for…". These score high on matching the brief's vocabulary and add
nothing to it. **Matching the brief's words is not understanding.**

## What the strongest bids did

Read across every category, independent of their position:

1. **They open with judgment.** A likely root cause ("the memory spikes probably come from
   storing every scan, the lag from sending full-resolution photos synchronously"), a
   constraint the client may not know about (an LMS editor that strips custom CSS), a known
   failure mode (fast pinball flippers letting the ball tunnel through), or the decision that
   shapes the build.
2. **They add one concrete expert detail** that lowers risk — where a value must come from,
   what happens on a retry, what the client will be able to change themselves.
3. **They reduce the client's effort and risk:** a small first piece to approve before the
   rest of the batch, what the client keeps (code, accounts), what the client won't have to do.
4. **They ask one decision question, as two options** ("one shared room, or private
   conversations?"; "keep its place until outbid, or expire?"). Easy to answer, and the answer
   matters.
5. **Credibility, when used, is one fact tied to the problem**, not a résumé.

## This account's 41 bids

Already better than most on the things that matter: no greetings or signatures, 93% open on
a detail specific to the brief (competitors 40%), every non-English brief answered in its own
language, no invented claims or links, a question in every bid.

What made them read as one template (counts out of 41):

| habit | bids | rule now |
|---|---|---|
| a paragraph starting "I'd start by / I'd first / Empezaría / Je commencerais" | 19 | formula opening (style) |
| "rather than / not just / more than" contrast | 18 | at most one per bid (style) |
| "My background includes X, Y, Z" / "My Freelancer profile includes…" | 12 | résumé line (style) |
| a recited checklist (an extreme enumeration in one sentence) | 11 | deliverable dump (style) |
| the identical "4.8/5 rating across 21 client reviews" | 10 | recycled sentence across bids (repetition) |
| "If you reply / Once you reply, I'll first…" | 10 | banned phrase (style) |
| unscoped work offered ("I can also prepare a small proof of concept", "I'll first show you a clickable prototype") | 12 | free-work / preview-promise checks (style) |
| compound questions ("What user volume, sign-on providers, and cloud environment…?") | 8 | one subject per question (style) |
| a portfolio item stretched to fit (a trade-show booth cited for an industrial logo) | several | portfolio fit graded strong / partial / none |
| a duration in the text ("5 y 7 días por lote") | 1 | durations only when echoing the brief |

Run against the corpus, the new checks flag 35 of the 41 sent bids for at least one of
these, while the hand-picked strongest competitor bids trip them only on house policy (a
"Hi," opener, bullets, stating a delivery time).

## How the writer works now

- **Analysis (v4)** separates confirmed requirements from assumptions, lists constraints,
  1–3 *implementation considerations* with why each matters, practical options, a small
  paid **first milestone**, the expected outcome, up to two **decision questions**, the
  client's register (formality, technical level, tú/usted, tu/vous), and grades the
  portfolio match: `strong` only for the same kind of deliverable; sharing a tool is
  `partial` and may not be presented as similar work.
- **Writing** is steered by an *opening move* (diagnosis, constraint, decision, risk,
  outcome) and a *closing move* that rotate per project, a length that follows complexity
  (≈80–130 words for a small task, up to ≈180 for a complex build), and a list of the
  four-word phrases this account has been overusing in its recent bids — computed from the
  bids themselves, so the list moves as the habits do.
- **Previews** are offered only when the analysis judges one useful and the budget is at
  least $100, and only as an option ("If it would help you decide, I can sketch the header
  in your palette before we settle the rest") — never as free work, never as something
  already made, never started unasked.
- **Every rule about truth is unchanged**: claims come only from the real Freelancer profile;
  an unsupported claim after the revisions holds the bid for a person.

Before/after examples on eight of the study's real briefs are kept, with the brief excerpts,
in `tests/fixtures/proposal_pairs.json` and checked by `tests/test_proposal_quality.py`.
The "after" texts were written by hand to these rules to show the target; they are not
output of the production model.

## Measuring whether it works

Subjective review scores are not evidence of more replies. To build a baseline, record for
each submitted bid whether the client replied (and whether it was awarded) for a few weeks
before and after this change, and compare reply rates per category. Until such data exists,
treat every quality score as a check on the writing, not on the result.
