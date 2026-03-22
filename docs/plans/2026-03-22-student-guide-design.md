# Prajna Student Guide — Full Subject-Wise Analysis Design
**Date:** 2026-03-22
**Feature:** Expandable "See Full Subject-Wise Guide" below the existing 5-item Priority Focus panel
**File affected:** `docs/student-dashboard.html`

---

## Problem

The current "Prajna-Powered Priority Focus Areas" card shows only 5 pre-computed `slm_focus` items. Students cannot see the full picture of their 59 chapters, cannot understand which subjects need the most attention, and get no actionable per-chapter revision guidance.

---

## Goal

Replace the hard limit with an interactive, subject-grouped guide covering **all chapters** for the selected student. Acts as a personal study advisor: tells the student *what* to study, *how long*, *in what order*, and *why*.

---

## Data Sources (all client-side, no new backend)

| Field | Source | Notes |
|---|---|---|
| All chapters (59) | `s.chapters` dict | `{name: [accuracy, level_code, slm_importance]}` |
| Subject mapping | `NEET_MAP` / `JEE_MAP` (already in page) | Maps chapter name → subject |
| Micro-topics | `/api/v1/data/topic-deep-dive?topic=X` | **Lazy** — only on chapter expand |
| Question count | Same API response | Used for "See N questions →" link |

### Computed Fields (browser-side)

```js
priority_score  = (100 - accuracy) * (slm_importance / 100)
study_hours     = { C: 6, W: 4, D: 2.5, S: 1, M: 0.5 }[level]
gap_to_next     = next_level_threshold - accuracy
subject_urgency = avg(priority_score of all chapters in subject)
```

---

## Layout

```
[existing 5-item slm_focus panel]

▼ See Full Subject-Wise Guide (59 chapters)   ← expand toggle button

┌─ Subject Panel (sorted by subject_urgency desc) ────────────────┐
│ 🔴 Chemistry  avg 52%  ·  7 weak  ·  3 critical  ·  Urgent     │  ← clickable header
│   #1  Polymers                 10%  ████░░░  imp 48%  [C]       │
│   #2  Aldehydes Ketones...     35%  ████░░░  imp 78%  [W]  ▶    │  ← click to expand
│        ┌─ Chapter Detail ────────────────────────────────────┐   │
│        │ ⏱ 6h study  │  📈 Rising  │  +25% to next level    │   │
│        │ Micro-topics: [lazy loaded via API]                  │   │
│        │ 📝 Strategy: template-driven revision advice         │   │
│        │ [See 47 exam questions →]                            │   │
│        └──────────────────────────────────────────────────────┘   │
│   #3  S Block Elements         30%  ...                          │
└──────────────────────────────────────────────────────────────────┘
┌─ Physics panel ─────────────────────────────────────────────────┐
│ ...                                                              │
└──────────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1. Expand Button
- Appended below the existing `buildSLMFocus` card
- Text: `▼ See Full Subject-Wise Guide (N chapters)`
- Toggles visibility of the full guide section
- Scrolls smoothly to the guide on open

### 2. Subject Panel Header
Colour-coded by urgency:
- 🔴 Red — avg priority_score > 35 (urgent)
- 🟡 Amber — avg priority_score 20–35 (moderate)
- 🟢 Green — avg priority_score < 20 (on track)

Shows: subject name · avg accuracy · weak count · critical count · status label

### 3. Chapter Row (collapsed)
- Rank badge (#1, #2…)
- Chapter name (truncated with title tooltip)
- Dual inline bars: accuracy (level colour) + Prajna importance (purple)
- Level badge: `[C]` `[W]` `[D]` `[S]` `[M]` with matching colour
- Expand chevron ▶ / ▼

### 4. Chapter Detail Card (expanded)
**Top row (3 columns):**
- ⏱ Study hours estimate
- 📈 Trend + short label (Rising / Stable / Declining)
- 🎯 Gap to next level ("Need +25% to reach Weak")

**Middle row:**
- Micro-topics (top 3, lazy-loaded from `/api/v1/data/topic-deep-dive`)
- Loading skeleton while fetching

**Bottom row (full-width):**
- 📝 Revision strategy (template-driven, data-aware):
  - `C` + rising → "Highest priority. Start NCERT basics immediately. Heavy PYQ practice."
  - `W` + rising → "Study theory first, then solve last 5 years PYQs. High ROI."
  - `D` + stable → "Regular practice. 2–3 mock tests to solidify."
  - `S` + any → "Maintenance only. Weekly revision, 1 test set."
  - `M` + any → "Well mastered. Quick revision before exam is sufficient."
- `[See N exam questions →]` button (count from API; href to Streamlit deep dive)

---

## Revision Strategy Templates

```js
const STRATEGY = {
  C: {
    rising:   "🚨 Highest priority. Start with NCERT basics, then solve last 5-yr PYQs daily. This topic is gaining importance.",
    stable:   "🚨 Critical gap. Dedicate first study block daily. Focus on fundamentals before attempting numericals.",
    declining:"⚠️ Critical but declining. Secure basic marks with NCERT; don't over-invest.",
  },
  W: {
    rising:   "🔴 High ROI. Study theory first, then 30+ PYQs. Rising importance makes this a priority pick.",
    stable:   "🔴 Needs work. Structured 4h block: 1.5h theory → 1h formula drill → 1.5h PYQ practice.",
    declining:"🟠 Weak but declining. Aim for passing marks; redistribute time if overwhelmed.",
  },
  D: {
    rising:   "🟡 Good potential. Consistent practice will push you to Strong. Focus on high-frequency micro-topics.",
    stable:   "🟡 Developing. 2–3 timed mock sets per week will consolidate this chapter.",
    declining:"🟡 Stable enough. Maintain with weekly revision; redirect extra time to weaker chapters.",
  },
  S: {
    any:      "🟢 Strong. Weekly maintenance revision and 1 mock test set is sufficient.",
  },
  M: {
    any:      "✅ Mastered. Quick 30-min revision before exam. No major investment needed.",
  },
};
```

---

## Interaction Flow

1. Student opens dashboard → sees existing 5-item focus panel as before
2. Clicks **"▼ See Full Subject-Wise Guide"** → guide expands, subjects appear
3. Clicks a **subject header** → panel opens (all others close, accordion style)
4. Clicks a **chapter row** → detail card expands, API call fires for micro-topics
5. Clicks **"See N questions →"** → opens `http://localhost:8501` deep dive pre-filtered

---

## Performance

- Guide HTML is built once on `pick(student)` and cached; re-render only on student switch
- Micro-topic API calls are lazy (one per chapter expand, cached in `Map`)
- Subject panels use CSS `max-height` transitions (no JS animation loop)
- Total additional JS: ~200 lines

---

## Files Changed

| File | Change |
|---|---|
| `docs/student-dashboard.html` | Add `buildFullGuide(s)`, expand button in `buildSLMFocus`, CSS for guide components |

No backend changes needed. No new data files.
