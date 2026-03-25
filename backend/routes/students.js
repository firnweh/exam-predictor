const router       = require('express').Router();
const path         = require('path');
const fs           = require('fs');
const pool         = require('../db/pool');
const authenticate = require('../middleware/authenticate');

// Apply auth to all routes in this file
router.use(authenticate);

// Helper — load JSON summary file (same files the old dashboard used)
function loadJSON(exam) {
  const file = path.join(__dirname, '../../docs', `student_summary_${exam}.json`);
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (e) {
    return null;
  }
}

// GET /api/students?exam=neet
router.get('/', (req, res) => {
  const { role, branch, studentId } = req.user;
  const exam = (req.query.exam || 'neet').toLowerCase();

  const data = loadJSON(exam);
  if (!data) return res.status(404).json({ error: 'No data file for exam: ' + exam });

  let students = data.students || [];

  if (role === 'center') {
    // centre login — filter to their branch only
    students = students.filter(s => s.coaching === branch);
  } else if (role === 'student') {
    // student login — only their own record
    students = students.filter(s => s.id === studentId);
  }
  // central — return all

  res.json({ exam_type: exam, students });
});

// GET /api/students/:id?exam=neet
router.get('/:id', (req, res) => {
  const { role, branch, studentId } = req.user;
  const { id }  = req.params;
  const exam    = (req.query.exam || 'neet').toLowerCase();

  // Students can only fetch themselves
  if (role === 'student' && studentId !== id)
    return res.status(403).json({ error: 'Forbidden' });

  const data = loadJSON(exam);
  if (!data) return res.status(404).json({ error: 'No data file for exam: ' + exam });

  const student = (data.students || []).find(s => s.id === id);
  if (!student) return res.status(404).json({ error: 'Student not found' });

  // Centre can only see their own branch
  if (role === 'center' && student.coaching !== branch)
    return res.status(403).json({ error: 'Forbidden' });

  res.json(student);
});

module.exports = router;
