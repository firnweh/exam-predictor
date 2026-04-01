# PRAJNA Portal — Unified Next.js Dashboard

**Date:** 2026-04-01
**Status:** Approved
**Repo:** `firnweh/prajna-portal` (new, separate from `prajna-slm`)

## Problem

PRAJNA currently has 4 separate entry points:
- Node.js backend serving static HTML dashboards (student + org views)
- Python FastAPI Intelligence API (prediction engine)
- Streamlit dashboard (11 analysis tabs)
- Static Vercel site (landing page)

Each uses its own tech stack, auth flow, and deployment. Users must navigate between different URLs. There is no unified experience.

## Solution

A single Next.js App Router application that:
- Provides role-based routing (student / center / central)
- Rebuilds key views (student dashboard, org dashboard, predictions) as React components
- Embeds Streamlit via iframe for heavy analysis tabs (backtest, paper generator, question explorer)
- Consumes both existing backends (Node.js + Python) without changes to them

## Tech Stack

- **Framework:** Next.js 14 (App Router)
- **Styling:** Tailwind CSS + shadcn/ui
- **Charts:** Recharts
- **State:** Zustand (user, exam, year, cached predictions)
- **Data fetching:** TanStack Query (5-min stale time)
- **Auth:** JWT from existing backend, stored in httpOnly cookie via Next.js API route

## Page Structure

```
app/
  layout.tsx              Root layout (dark theme, Inter font)
  login/page.tsx          Login page
  student/
    layout.tsx            Student nav shell (sidebar)
    page.tsx              Main dashboard (hero, metrics, subject cards)
    [subject]/page.tsx    Subject deep-dive (KPIs, priorities, chapters, micro-topics)
  org/
    layout.tsx            Org nav shell (sidebar)
    page.tsx              KPIs, branch cards, PRAJNA intel, subject matrix, leaderboard
    [branch]/page.tsx     Branch drill-down
  predictions/
    page.tsx              Top predictions, hot/cold, lesson plan
    [topic]/page.tsx      Topic deep-dive
  analysis/
    page.tsx              Streamlit iframe (backtest, paper gen, question explorer)
```

## Middleware Auth

`middleware.ts` runs on every request:
- No token -> `/login`
- `student` role -> can access `/student`, `/predictions`, `/analysis`
- `center`/`central` role -> can access `/org`, `/predictions`, `/analysis`
- Role mismatch -> redirect to correct home page

## Sidebar Navigation

**Student sidebar:**
- My Dashboard
- Physics / Chemistry / Biology (or Mathematics) -> subject deep-dive pages
- Predictions
- Lesson Plan
- Mistake Analysis
- Deep Analysis (Streamlit iframe)
- Context footer: exam type, year, student name, sign out

**Org sidebar:**
- Organisation overview
- Branches
- Student List
- Predictions
- Lesson Plan
- Mistake Analysis
- Deep Analysis (Streamlit iframe)
- Context footer: exam type, year, branch, role, sign out

Subject links are dynamic — derived from student data (NEET = Phy/Chem/Botany/Zoology, JEE = Phy/Chem/Math).

## Component Library

```
components/
  layout/
    Sidebar.tsx           Role-aware sidebar
    Header.tsx            Breadcrumb + exam toggle
    StreamlitEmbed.tsx    iframe wrapper for Streamlit
  dashboard/
    KpiCard.tsx           Metric card (value, label, trend, color)
    KpiStrip.tsx          Row of KpiCards
    SubjectCard.tsx       Clickable subject entry point
    ChapterRow.tsx        Collapsible chapter with micro-topic table
    MicroTopicTable.tsx   Table: name | student % | PRAJNA % | ROI badge
    ZoneBadge.tsx         M/S/D/W/C level badge
    RoiBadge.tsx          CRITICAL/FOCUS/REVIEW/OK badge
  charts/
    RadarChart.tsx        Subject strengths
    TrajectoryChart.tsx   Score over 10 exams
    BarRank.tsx           Horizontal bar comparison
    DonutChart.tsx        Zone distribution
  org/
    BranchCard.tsx        Branch KPI card
    SubjectMatrix.tsx     Branch x Subject accuracy grid
    Leaderboard.tsx       Sortable student table
    PrajnaIntel.tsx       Predictions x student gaps
  predictions/
    PredictionCard.tsx    Single prediction with prob bar, trend, Qs
    HotColdGrid.tsx       Hot/cold/cyclical topics
    SignalBreakdown.tsx   Signal bar visualization
```

## API Integration

Two fetch helpers in `lib/api.ts`:

```typescript
backend(path)       -> NEXT_PUBLIC_BACKEND_URL + path
intelligence(path)  -> NEXT_PUBLIC_INTEL_URL + path
```

Both attach JWT Bearer token from cookie.

**Environment variables:**
- `NEXT_PUBLIC_BACKEND_URL` = Railway Node.js backend
- `NEXT_PUBLIC_INTEL_URL` = Railway Python Intelligence API
- `NEXT_PUBLIC_STREAMLIT_URL` = Railway Streamlit dashboard

**Page data sources:**

| Page | Backend (Node.js) | Intelligence (Python) |
|------|-------------------|----------------------|
| `/student` | `/api/students?exam=` | `/api/v1/data/predict?level=micro&top_n=200` |
| `/student/[subject]` | Same (from Zustand cache) | Same (from Zustand cache) |
| `/org` | `/api/students`, `/api/branches` | `/api/v1/data/predict?level=chapter&top_n=15` |
| `/predictions` | None | `/api/v1/data/predict`, `/api/v1/data/hot-cold-topics`, `/api/v1/data/lesson-plan` |
| `/analysis` | None | None (Streamlit iframe) |

## Visual Design

Dark theme matching existing PRAJNA aesthetic:

```
--bg:       #0f0f1a    --accent:   #6366f1
--surface:  #131320    --accent2:  #00d4aa
--card:     #1a1d2e    --warn:     #ff6b6b
--border:   #1e1e3a    --gold:     #ffd166
--text:     #e2e8f0    --muted:    #64748b
```

Subject colors: Physics=#f59e0b, Chemistry=#6366f1, Biology=#22c55e, Mathematics=#a855f7

shadcn/ui components with dark mode override. Inter font. Desktop-first responsive (sidebar collapses at 768px).

## Deployment

- Vercel (replaces current static site)
- Environment variables configured in Vercel dashboard
- Auto-deploy from `firnweh/prajna-portal` main branch

## What Stays the Same

- Node.js backend (Railway) — no changes
- Python Intelligence API (Railway) — no changes
- Streamlit dashboard (Railway) — no changes, embedded via iframe
- SQLite exam.db — unchanged
- PostgreSQL student data — unchanged
- JWT auth mechanism — same tokens, same expiry

## Success Criteria

- Single URL for all users (student, center, central)
- Role-based routing works correctly
- Student dashboard shows same data as current HTML version
- Org dashboard shows same data as current HTML version
- Predictions page shows live PRAJNA predictions
- Subject deep-dive has 4 zones (KPIs, top priorities, chapter breakdown, exam history)
- Streamlit iframe loads correctly for deep analysis tabs
- Login flow works end-to-end
- Mobile-responsive at 768px and 375px breakpoints
