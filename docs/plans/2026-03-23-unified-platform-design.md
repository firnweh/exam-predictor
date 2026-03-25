# PRAJNA Unified Platform — Design Document
Date: 2026-03-23

## Summary
Combine student dashboard, org dashboard, and center dashboard into a single
role-gated platform with login, backed by Node.js + Express on Railway.app
with PostgreSQL.

## Tech Stack
| Layer | Choice |
|-------|--------|
| Backend | Node.js + Express |
| Database | PostgreSQL (Railway managed) |
| Auth | JWT (jsonwebtoken) + bcrypt |
| Frontend | Existing static HTML — minimal changes |
| Hosting | Railway.app (backend + DB) + static files served by Express |

## Architecture
```
BROWSER (static HTML/JS)
  login.html → portal.html → [student-dashboard | org-dashboard]
       │
       │ HTTPS REST  (Authorization: Bearer <JWT>)
       ▼
RAILWAY — Express API
  POST /api/auth/login
  GET  /api/auth/me
  GET  /api/students          (role-filtered)
  GET  /api/students/:id      (role-gated)
  GET  /api/branches          (center / central only)
       │
       ▼
RAILWAY — PostgreSQL
  users · students
```

## Roles & Access
| Role     | Login              | Access |
|----------|--------------------|--------|
| central  | admin@prajna.ai    | All students, all branches, full org dashboard |
| center   | kota@pw.in etc.    | Only their branch students, org dashboard pre-filtered |
| student  | STU001@prajna.ai   | Only their own student dashboard |

JWT payload: { userId, role, branch?, studentId?, exam?, iat, exp }
Server enforces rules — front-end never receives data it shouldn't see.

## Database Schema
```sql
users(id SERIAL PK, email TEXT UNIQUE, password_hash TEXT,
      role TEXT, branch_name TEXT, student_id TEXT, exam_type TEXT, created_at)

students(id SERIAL PK, student_id TEXT, exam_type TEXT,
         name TEXT, city TEXT, coaching TEXT, target TEXT,
         metrics JSONB, subjects JSONB, chapters JSONB,
         slm_focus JSONB, strengths JSONB)
```

## Seeding
seed.js reads student_summary_neet.json + student_summary_jee.json:
- Inserts 400 students into students table
- Creates 400 student user accounts (STU001@prajna.ai, default pw: prajna@2025)
- Creates 12 center accounts (kota@pw.in … bhopal@pw.in, default pw: pw@2025)
- Creates 1 central account (admin@prajna.ai, default pw: admin@2025)

## Front-end Changes
| File | Change |
|------|--------|
| login.html | NEW — email+password form, stores JWT in localStorage |
| portal.html | NEW — reads role from JWT, redirects to correct dashboard |
| auth.js | NEW — shared guard, redirects to login if no valid JWT |
| student-dashboard.html | Replace JSON fetch → /api/students/:id |
| org-dashboard.html | Replace JSON fetch → /api/students + /api/branches |

## File Structure
```
exam-predictor/
  backend/
    server.js          # Express app entry
    routes/
      auth.js          # POST /api/auth/login, GET /api/auth/me
      students.js      # GET /api/students, GET /api/students/:id
      branches.js      # GET /api/branches
    middleware/
      authenticate.js  # JWT verification
      authorize.js     # Role enforcement
    db/
      pool.js          # pg Pool
      seed.js          # One-time seed from JSON
    package.json
    railway.toml
  docs/
    login.html         # NEW
    portal.html        # NEW
    auth.js            # NEW shared guard
    student-dashboard.html  # updated fetch
    org-dashboard.html      # updated fetch
```

## Security Notes
- Passwords hashed with bcrypt (rounds=12)
- JWT expires in 8 hours
- CORS restricted to same origin in production
- All role checks server-side — JWT only carries identity
