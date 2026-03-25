# PRAJNA Unified Platform — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a role-gated unified platform combining student, center, and org dashboards behind a JWT login system backed by Express + PostgreSQL on Railway.

**Architecture:** Express REST API serves both static HTML and JSON data; PostgreSQL stores users and student records seeded from existing JSON files; front-end guards read JWT from localStorage and redirect to login if missing.

**Tech Stack:** Node.js 20, Express 4, pg (node-postgres), jsonwebtoken, bcryptjs, Railway.app (Node + Postgres), vanilla JS front-end.

---

## Task 1: Backend scaffold

**Files:**
- Create: `backend/package.json`
- Create: `backend/server.js`
- Create: `backend/.env.example`
- Create: `backend/railway.toml`

**Step 1: Initialise backend folder**

```bash
mkdir -p /Users/aman/exam-predictor/backend
cd /Users/aman/exam-predictor/backend
npm init -y
npm install express pg jsonwebtoken bcryptjs cors dotenv
npm install --save-dev nodemon
```

**Step 2: Create `backend/.env.example`**

```
DATABASE_URL=postgresql://user:pass@localhost:5432/prajna
JWT_SECRET=change_me_to_a_random_32_char_string
PORT=4000
FRONTEND_ORIGIN=http://localhost:8080
```

Copy to `.env` and fill in local PostgreSQL credentials for development.

**Step 3: Create `backend/server.js`**

```js
require('dotenv').config();
const express = require('express');
const cors    = require('cors');
const path    = require('path');

const app = express();
app.use(cors({ origin: process.env.FRONTEND_ORIGIN || '*' }));
app.use(express.json());

// Serve static front-end files
app.use(express.static(path.join(__dirname, '../docs')));

// Routes (added in later tasks)
app.use('/api/auth',     require('./routes/auth'));
app.use('/api/students', require('./routes/students'));
app.use('/api/branches', require('./routes/branches'));

// Catch-all → login
app.get('*', (_req, res) =>
  res.sendFile(path.join(__dirname, '../docs/login.html')));

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => console.log(`PRAJNA API on :${PORT}`));
```

**Step 4: Create `backend/railway.toml`**

```toml
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "node server.js"
healthcheckPath = "/api/auth/health"
```

**Step 5: Add scripts to `backend/package.json`**

Add inside `"scripts"`:
```json
"start": "node server.js",
"dev":   "nodemon server.js"
```

**Step 6: Commit**

```bash
cd /Users/aman/exam-predictor
git add backend/
git commit -m "feat: add express backend scaffold"
```

---

## Task 2: Database pool + schema

**Files:**
- Create: `backend/db/pool.js`
- Create: `backend/db/schema.sql`

**Step 1: Create `backend/db/pool.js`**

```js
const { Pool } = require('pg');
const pool = new Pool({ connectionString: process.env.DATABASE_URL });
module.exports = pool;
```

**Step 2: Create `backend/db/schema.sql`**

```sql
CREATE TABLE IF NOT EXISTS users (
  id           SERIAL PRIMARY KEY,
  email        TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  role         TEXT NOT NULL CHECK (role IN ('central','center','student')),
  branch_name  TEXT,
  student_id   TEXT,
  exam_type    TEXT,
  created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS students (
  id          SERIAL PRIMARY KEY,
  student_id  TEXT NOT NULL,
  exam_type   TEXT NOT NULL,
  name        TEXT,
  city        TEXT,
  coaching    TEXT,
  target      TEXT,
  abilities   JSONB,
  metrics     JSONB,
  subjects    JSONB,
  chapters    JSONB,
  slm_focus   JSONB,
  strengths   JSONB,
  UNIQUE(student_id, exam_type)
);

CREATE INDEX IF NOT EXISTS idx_students_coaching  ON students(coaching);
CREATE INDEX IF NOT EXISTS idx_students_exam_type ON students(exam_type);
CREATE INDEX IF NOT EXISTS idx_users_email        ON users(email);
```

**Step 3: Apply schema to local dev DB**

```bash
psql $DATABASE_URL -f backend/db/schema.sql
```

Expected: `CREATE TABLE`, `CREATE INDEX` lines — no errors.

**Step 4: Commit**

```bash
git add backend/db/
git commit -m "feat: add db pool and schema"
```

---

## Task 3: Seed script

**Files:**
- Create: `backend/db/seed.js`

**Step 1: Create `backend/db/seed.js`**

```js
require('dotenv').config();
const bcrypt = require('bcryptjs');
const pool   = require('./pool');
const fs     = require('fs');
const path   = require('path');

const HASH_ROUNDS = 12;
const PW_BRANCHES = [
  'PW Kota','PW Delhi','PW Patna','PW Lucknow','PW Jaipur','PW Mumbai',
  'PW Hyderabad','PW Kolkata','PW Chennai','PW Pune','PW Ahmedabad','PW Bhopal'
];

async function seed() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // 1. Central admin
    await client.query(
      `INSERT INTO users(email,password_hash,role) VALUES($1,$2,'central')
       ON CONFLICT(email) DO NOTHING`,
      ['admin@prajna.ai', await bcrypt.hash('admin@2025', HASH_ROUNDS)]
    );

    // 2. Center accounts
    for (const branch of PW_BRANCHES) {
      const slug  = branch.toLowerCase().replace('pw ','');
      const email = slug + '@pw.in';
      await client.query(
        `INSERT INTO users(email,password_hash,role,branch_name)
         VALUES($1,$2,'center',$3) ON CONFLICT(email) DO NOTHING`,
        [email, await bcrypt.hash('pw@2025', HASH_ROUNDS), branch]
      );
    }

    // 3. Students from both JSON files
    for (const exam of ['neet','jee']) {
      const filePath = path.join(__dirname,'../../docs/student_summary_'+exam+'.json');
      const data     = JSON.parse(fs.readFileSync(filePath));
      for (const s of data.students) {
        // Insert student record
        await client.query(
          `INSERT INTO students(student_id,exam_type,name,city,coaching,target,
            abilities,metrics,subjects,chapters,slm_focus,strengths)
           VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
           ON CONFLICT(student_id,exam_type) DO UPDATE SET
             metrics=$8, subjects=$9, chapters=$10, slm_focus=$11, strengths=$12`,
          [s.id, exam, s.name, s.city, s.coaching, s.target,
           JSON.stringify(s.abilities||{}),
           JSON.stringify(s.metrics||{}),
           JSON.stringify(s.subjects||{}),
           JSON.stringify(s.chapters||{}),
           JSON.stringify(s.slm_focus||[]),
           JSON.stringify(s.strengths||[])]
        );
        // Insert student user account
        const email = s.id.toLowerCase() + '@prajna.ai';
        await client.query(
          `INSERT INTO users(email,password_hash,role,student_id,exam_type,branch_name)
           VALUES($1,$2,'student',$3,$4,$5) ON CONFLICT(email) DO NOTHING`,
          [email, await bcrypt.hash('prajna@2025', HASH_ROUNDS), s.id, exam, s.coaching]
        );
      }
      console.log(`Seeded ${data.students.length} ${exam.toUpperCase()} students`);
    }

    await client.query('COMMIT');
    console.log('Seed complete.');
  } catch (e) {
    await client.query('ROLLBACK');
    console.error(e);
  } finally {
    client.release();
    pool.end();
  }
}

seed();
```

**Step 2: Run seed against local dev DB**

```bash
cd backend && node db/seed.js
```

Expected output:
```
Seeded 200 NEET students
Seeded 200 JEE students
Seed complete.
```

**Step 3: Verify**

```bash
psql $DATABASE_URL -c "SELECT role, COUNT(*) FROM users GROUP BY role;"
```

Expected:
```
  role   | count
---------+-------
 central |     1
 center  |    12
 student |   400
```

**Step 4: Commit**

```bash
git add backend/db/seed.js
git commit -m "feat: add seed script for users and students"
```

---

## Task 4: Auth middleware

**Files:**
- Create: `backend/middleware/authenticate.js`
- Create: `backend/middleware/authorize.js`

**Step 1: Create `backend/middleware/authenticate.js`**

```js
const jwt = require('jsonwebtoken');

module.exports = function authenticate(req, res, next) {
  const header = req.headers.authorization || '';
  const token  = header.startsWith('Bearer ') ? header.slice(7) : null;
  if (!token) return res.status(401).json({ error: 'No token' });
  try {
    req.user = jwt.verify(token, process.env.JWT_SECRET);
    next();
  } catch {
    res.status(401).json({ error: 'Invalid or expired token' });
  }
};
```

**Step 2: Create `backend/middleware/authorize.js`**

```js
// Usage: authorize('central', 'center')
module.exports = (...roles) => (req, res, next) => {
  if (!roles.includes(req.user.role))
    return res.status(403).json({ error: 'Forbidden' });
  next();
};
```

**Step 3: Commit**

```bash
git add backend/middleware/
git commit -m "feat: add JWT authenticate and authorize middleware"
```

---

## Task 5: Auth routes

**Files:**
- Create: `backend/routes/auth.js`

**Step 1: Create `backend/routes/auth.js`**

```js
const router  = require('express').Router();
const bcrypt  = require('bcryptjs');
const jwt     = require('jsonwebtoken');
const pool    = require('../db/pool');
const authenticate = require('../middleware/authenticate');

// Health check (used by Railway)
router.get('/health', (_req, res) => res.json({ ok: true }));

// POST /api/auth/login
router.post('/login', async (req, res) => {
  const { email, password } = req.body;
  if (!email || !password)
    return res.status(400).json({ error: 'Email and password required' });

  const { rows } = await pool.query(
    'SELECT * FROM users WHERE email = $1', [email.toLowerCase().trim()]
  );
  const user = rows[0];
  if (!user || !(await bcrypt.compare(password, user.password_hash)))
    return res.status(401).json({ error: 'Invalid credentials' });

  const payload = {
    userId:    user.id,
    role:      user.role,
    branch:    user.branch_name  || null,
    studentId: user.student_id   || null,
    exam:      user.exam_type    || null,
  };
  const token = jwt.sign(payload, process.env.JWT_SECRET, { expiresIn: '8h' });
  res.json({ token, user: payload });
});

// GET /api/auth/me
router.get('/me', authenticate, (req, res) => res.json(req.user));

module.exports = router;
```

**Step 2: Test login manually**

```bash
# Start server
cd backend && node server.js &

# Test login
curl -s -X POST http://localhost:4000/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"admin@prajna.ai","password":"admin@2025"}' | jq .
```

Expected: `{ "token": "eyJ...", "user": { "role": "central", ... } }`

**Step 3: Commit**

```bash
git add backend/routes/auth.js
git commit -m "feat: add login and /me auth routes"
```

---

## Task 6: Students API route

**Files:**
- Create: `backend/routes/students.js`

**Step 1: Create `backend/routes/students.js`**

```js
const router       = require('express').Router();
const pool         = require('../db/pool');
const authenticate = require('../middleware/authenticate');
const authorize    = require('../middleware/authorize');

// Apply auth to all routes in this file
router.use(authenticate);

// GET /api/students?exam=neet
router.get('/', async (req, res) => {
  const { role, branch, studentId } = req.user;
  const exam = req.query.exam || 'neet';

  let query, params;

  if (role === 'central') {
    query  = 'SELECT * FROM students WHERE exam_type=$1 ORDER BY student_id';
    params = [exam];
  } else if (role === 'center') {
    query  = 'SELECT * FROM students WHERE exam_type=$1 AND coaching=$2 ORDER BY student_id';
    params = [exam, branch];
  } else {
    // student — return only their own record
    query  = 'SELECT * FROM students WHERE exam_type=$1 AND student_id=$2';
    params = [exam, studentId];
  }

  const { rows } = await pool.query(query, params);
  res.json({ exam_type: exam, students: rows });
});

// GET /api/students/:id
router.get('/:id', async (req, res) => {
  const { role, branch, studentId } = req.user;
  const { id } = req.params;
  const exam   = req.query.exam || 'neet';

  // Students can only fetch themselves
  if (role === 'student' && studentId !== id)
    return res.status(403).json({ error: 'Forbidden' });

  const { rows } = await pool.query(
    'SELECT * FROM students WHERE student_id=$1 AND exam_type=$2', [id, exam]
  );
  if (!rows.length) return res.status(404).json({ error: 'Not found' });

  const student = rows[0];

  // Center can only see their own branch
  if (role === 'center' && student.coaching !== branch)
    return res.status(403).json({ error: 'Forbidden' });

  res.json(student);
});

module.exports = router;
```

**Step 2: Test as central**

```bash
TOKEN=$(curl -s -X POST http://localhost:4000/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"admin@prajna.ai","password":"admin@2025"}' | jq -r .token)

curl -s "http://localhost:4000/api/students?exam=neet" \
  -H "Authorization: Bearer $TOKEN" | jq '.students | length'
```

Expected: `200`

**Step 3: Test as center (should only see their branch)**

```bash
CENTER_TOKEN=$(curl -s -X POST http://localhost:4000/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"kota@pw.in","password":"pw@2025"}' | jq -r .token)

curl -s "http://localhost:4000/api/students?exam=neet" \
  -H "Authorization: Bearer $CENTER_TOKEN" | jq '.students | length'
```

Expected: `~17` (only PW Kota students)

**Step 4: Commit**

```bash
git add backend/routes/students.js
git commit -m "feat: add role-gated students API route"
```

---

## Task 7: Branches API route

**Files:**
- Create: `backend/routes/branches.js`

**Step 1: Create `backend/routes/branches.js`**

```js
const router       = require('express').Router();
const pool         = require('../db/pool');
const authenticate = require('../middleware/authenticate');
const authorize    = require('../middleware/authorize');

router.use(authenticate);
router.use(authorize('central', 'center'));

// GET /api/branches?exam=neet
router.get('/', async (req, res) => {
  const { role, branch } = req.user;
  const exam = req.query.exam || 'neet';

  const where  = role === 'center' ? 'AND coaching=$2' : '';
  const params = role === 'center' ? [exam, branch] : [exam];

  const { rows } = await pool.query(
    `SELECT coaching AS branch, COUNT(*) AS count,
            ROUND(AVG((metrics->>'avg_percentage')::numeric),1) AS avg_score,
            ROUND(AVG((metrics->>'improvement')::numeric),1)    AS avg_improvement,
            SUM(CASE WHEN (metrics->>'avg_percentage')::numeric < 25 THEN 1 ELSE 0 END) AS at_risk
     FROM students WHERE exam_type=$1 ${where}
     GROUP BY coaching ORDER BY coaching`, params
  );
  res.json({ branches: rows });
});

module.exports = router;
```

**Step 2: Commit**

```bash
git add backend/routes/branches.js
git commit -m "feat: add branches summary API route"
```

---

## Task 8: Front-end auth guard + login page

**Files:**
- Create: `docs/auth.js`
- Create: `docs/login.html`
- Create: `docs/portal.html`

**Step 1: Create `docs/auth.js`**

```js
// Shared auth guard — include in every dashboard page
const AUTH = (() => {
  const KEY = 'prajna_token';

  function getToken()  { return localStorage.getItem(KEY); }
  function setToken(t) { localStorage.setItem(KEY, t); }
  function clearToken(){ localStorage.removeItem(KEY); }

  function parseJWT(token) {
    try {
      return JSON.parse(atob(token.split('.')[1]));
    } catch { return null; }
  }

  function getUser() {
    const t = getToken();
    if (!t) return null;
    const payload = parseJWT(t);
    if (!payload || payload.exp * 1000 < Date.now()) { clearToken(); return null; }
    return payload;
  }

  function requireAuth(allowedRoles) {
    const user = getUser();
    if (!user) { window.location.href = '/login.html'; return null; }
    if (allowedRoles && !allowedRoles.includes(user.role)) {
      window.location.href = '/portal.html';
      return null;
    }
    return user;
  }

  async function apiFetch(url, opts = {}) {
    const token = getToken();
    const res = await fetch(url, {
      ...opts,
      headers: { ...(opts.headers||{}), 'Authorization': 'Bearer ' + token,
                 'Content-Type': 'application/json' }
    });
    if (res.status === 401) { clearToken(); window.location.href = '/login.html'; }
    return res;
  }

  function logout() { clearToken(); window.location.href = '/login.html'; }

  return { getToken, setToken, clearToken, getUser, requireAuth, apiFetch, logout };
})();
```

**Step 2: Create `docs/login.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PRAJNA · Login</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0f1117;color:#e8eaf6;font-family:'Segoe UI',system-ui,sans-serif;
  min-height:100vh;display:flex;align-items:center;justify-content:center}
.card{background:#1a1d27;border:1px solid #2a2e45;border-radius:16px;
  padding:2.5rem 2rem;width:100%;max-width:400px}
.logo{font-size:1.4rem;font-weight:800;color:#6c63ff;margin-bottom:.25rem}
.logo span{color:#00d4aa}
.sub{color:#7880a4;font-size:.82rem;margin-bottom:2rem}
label{display:block;font-size:.75rem;font-weight:600;color:#7880a4;
  text-transform:uppercase;letter-spacing:.05em;margin-bottom:.4rem}
input{width:100%;background:#21253a;border:1px solid #2a2e45;color:#e8eaf6;
  padding:.65rem .9rem;border-radius:8px;font-size:.9rem;outline:none;margin-bottom:1.1rem}
input:focus{border-color:#6c63ff}
button{width:100%;background:#6c63ff;color:#fff;border:none;padding:.75rem;
  border-radius:8px;font-size:.95rem;font-weight:700;cursor:pointer;transition:.2s}
button:hover{background:#7c74ff}
.error{color:#ff6b6b;font-size:.8rem;margin-top:.75rem;text-align:center;min-height:1.2em}
.hint{margin-top:1.5rem;padding:1rem;background:#21253a;border-radius:8px;font-size:.72rem;color:#7880a4}
.hint strong{color:#a0a8cc;display:block;margin-bottom:.4rem}
</style>
</head>
<body>
<div class="card">
  <div class="logo">PRAJNA <span>× PhysicsWallah</span></div>
  <div class="sub">Student Intelligence Platform</div>
  <form id="form">
    <label>Email</label>
    <input id="email" type="email" placeholder="you@prajna.ai" autocomplete="username" required>
    <label>Password</label>
    <input id="pass" type="password" placeholder="••••••••" autocomplete="current-password" required>
    <button type="submit" id="btn">Sign In</button>
    <div class="error" id="err"></div>
  </form>
  <div class="hint">
    <strong>Demo credentials</strong>
    Central: admin@prajna.ai / admin@2025<br>
    Center:  kota@pw.in / pw@2025<br>
    Student: stu001@prajna.ai / prajna@2025
  </div>
</div>
<script>
// If already logged in, skip to portal
const t = localStorage.getItem('prajna_token');
if (t) {
  try {
    const p = JSON.parse(atob(t.split('.')[1]));
    if (p.exp * 1000 > Date.now()) window.location.href = '/portal.html';
  } catch {}
}

document.getElementById('form').addEventListener('submit', async e => {
  e.preventDefault();
  const btn = document.getElementById('btn');
  const err = document.getElementById('err');
  btn.textContent = 'Signing in…'; btn.disabled = true; err.textContent = '';
  try {
    const res  = await fetch('/api/auth/login', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ email: document.getElementById('email').value,
                             password: document.getElementById('pass').value })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Login failed');
    localStorage.setItem('prajna_token', data.token);
    window.location.href = '/portal.html';
  } catch(ex) {
    err.textContent = ex.message;
    btn.textContent = 'Sign In'; btn.disabled = false;
  }
});
</script>
</body>
</html>
```

**Step 3: Create `docs/portal.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PRAJNA · Redirecting…</title>
<script src="/auth.js"></script>
<script>
const user = AUTH.requireAuth();
if (user) {
  if (user.role === 'student')
    window.location.href = '/student-dashboard.html?id=' + user.studentId + '&exam=' + (user.exam || 'neet');
  else
    window.location.href = '/org-dashboard.html';
}
</script>
</head>
<body style="background:#0f1117;color:#7880a4;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh">
  Redirecting…
</body>
</html>
```

**Step 4: Commit**

```bash
git add docs/auth.js docs/login.html docs/portal.html
git commit -m "feat: add login page, portal router, and auth guard"
```

---

## Task 9: Wire auth guard into existing dashboards

**Files:**
- Modify: `docs/student-dashboard.html`
- Modify: `docs/org-dashboard.html`

**Step 1: Add `<script src="/auth.js"></script>` to both dashboards**

In `student-dashboard.html`, just before the closing `</head>` tag, add:
```html
<script src="/auth.js"></script>
```

In `org-dashboard.html`, same location, add:
```html
<script src="/auth.js"></script>
```

**Step 2: Add auth guard + logout button to student-dashboard.html**

At the top of the main `<script>` block (before any other code), add:
```js
const _user = AUTH.requireAuth(['student','center','central']);
```

In the nav, replace the existing "← Org Dashboard" link with:
```html
<a href="org-dashboard.html" class="nav-back" id="nav-org" style="margin-left:0">← Org Dashboard</a>
<button onclick="AUTH.logout()" class="nav-back" style="margin-left:0;border:none;cursor:pointer;background:none;color:inherit">Sign Out</button>
```

After `const _user = AUTH.requireAuth(...)`, add:
```js
// Students can't navigate to org dashboard
if (_user && _user.role === 'student')
  document.getElementById('nav-org').style.display = 'none';
```

**Step 3: Add auth guard to org-dashboard.html**

At the top of the main `<script>` block, add:
```js
const _user = AUTH.requireAuth(['center','central']);
```

In the nav, add logout button:
```html
<button onclick="AUTH.logout()" class="nav-link" style="background:none;border:none;cursor:pointer;color:var(--muted)">Sign Out</button>
```

**Step 4: Replace JSON fetch with API fetch in student-dashboard.html**

Find:
```js
async function loadData(exam){
  const r=await fetch('student_summary_'+exam+'.json');
```

Replace with:
```js
async function loadData(exam){
  const r=await AUTH.apiFetch('/api/students?exam='+exam);
```

**Step 5: Replace JSON fetch with API fetch in org-dashboard.html**

Find:
```js
async function loadExam(exam){
  const r = await fetch('student_summary_'+exam+'.json');
  return (await r.json()).students;
}
```

Replace with:
```js
async function loadExam(exam){
  const r = await AUTH.apiFetch('/api/students?exam='+exam);
  return (await r.json()).students;
}
```

**Step 6: Commit**

```bash
git add docs/student-dashboard.html docs/org-dashboard.html
git commit -m "feat: wire auth guard and API fetch into dashboards"
```

---

## Task 10: Deploy to Railway

**Step 1: Push to GitHub (if not already)**

```bash
cd /Users/aman/exam-predictor
git remote add origin https://github.com/<your-username>/prajna.git  # if needed
git push -u origin main
```

**Step 2: Create Railway project**

1. Go to https://railway.app → New Project → Deploy from GitHub repo
2. Select the `exam-predictor` repo
3. Railway auto-detects Node — set **Root Directory** to `backend`
4. Add a **PostgreSQL** plugin to the same project (Railway dashboard → + New → Database → PostgreSQL)

**Step 3: Set environment variables in Railway**

In Railway project → Variables, add:
```
DATABASE_URL   = (auto-filled by Railway Postgres plugin)
JWT_SECRET     = (generate: node -e "console.log(require('crypto').randomBytes(32).toString('hex'))")
FRONTEND_ORIGIN = https://<your-railway-domain>.railway.app
PORT           = 4000
```

**Step 4: Run schema + seed on Railway**

```bash
# Get Railway DB URL
railway variables --service prajna-db | grep DATABASE_URL

# Apply schema
DATABASE_URL=<railway_url> node -e "
  require('dotenv').config();
  const {Pool}=require('pg');
  const fs=require('fs');
  const pool=new Pool({connectionString:process.env.DATABASE_URL,ssl:{rejectUnauthorized:false}});
  pool.query(fs.readFileSync('db/schema.sql','utf8')).then(()=>{console.log('done');pool.end()});
"

# Seed
DATABASE_URL=<railway_url> node db/seed.js
```

**Step 5: Verify deployment**

```bash
curl https://<your-domain>.railway.app/api/auth/health
```

Expected: `{"ok":true}`

**Step 6: Final commit**

```bash
git add .
git commit -m "chore: production-ready unified platform"
git push
```

---

## Credential Reference

| Role | Email | Password |
|------|-------|----------|
| Central admin | admin@prajna.ai | admin@2025 |
| Center (per branch) | kota@pw.in, delhi@pw.in … bhopal@pw.in | pw@2025 |
| Student | stu001@prajna.ai … stu200@prajna.ai | prajna@2025 |
