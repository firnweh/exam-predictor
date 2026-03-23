const router       = require('express').Router();
const bcrypt       = require('bcryptjs');
const jwt          = require('jsonwebtoken');
const pool         = require('../db/pool');
const authenticate = require('../middleware/authenticate');

// Health check (used by Railway)
router.get('/health', (_req, res) => res.json({ ok: true }));

// POST /api/auth/login
router.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password)
      return res.status(400).json({ error: 'Email and password required' });

    const { rows } = await pool.query(
      'SELECT * FROM users WHERE email = $1',
      [email.toLowerCase().trim()]
    );
    const user = rows[0];
    if (!user || !(await bcrypt.compare(password, user.password_hash)))
      return res.status(401).json({ error: 'Invalid credentials' });

    const payload = {
      userId:    user.id,
      role:      user.role,
      branch:    user.branch_name || null,
      studentId: user.student_id  || null,
      exam:      user.exam_type   || null,
    };
    const token = jwt.sign(payload, process.env.JWT_SECRET, { expiresIn: '8h' });
    res.json({ token, user: payload });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
});

// GET /api/auth/me
router.get('/me', authenticate, (req, res) => res.json(req.user));

module.exports = router;
