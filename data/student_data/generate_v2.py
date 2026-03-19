"""
PRAJNA – Student Data Generator v2
====================================
Generates 200 students × 10 mock exams at the 3-level hierarchy:
  Subject → Chapter → Sub-chapter → Micro-topic

Output CSVs (in same directory):
  students_v2.csv
  neet_results_v2.csv     (200 × 10 × 59 chapters = 118,000 rows)
  neet_summary_v2.csv     (200 × 10 = 2,000 rows)
  jee_results_v2.csv      (200 × 10 × 60 sub-chapters = 120,000 rows)
  jee_summary_v2.csv      (200 × 10 = 2,000 rows)
"""

import csv
import datetime
import math
import random
from pathlib import Path

SEED = 77
random.seed(SEED)
OUT_DIR = Path(__file__).parent

# ── NEET 3-level syllabus (Subject → Chapter → micro_topics) ────────────────
# micro_topics are grouped into sub_chapters automatically (first half = Fundamentals, second = Advanced)

NEET_SYLLABUS = {
    "Physics": {
        "Physical World And Measurement": [
            "Units of measurement", "Dimensional analysis", "Errors in measurement",
            "Significant figures", "Physical laws and nature",
        ],
        "Kinematics": [
            "Frame of reference", "Motion in a straight line",
            "Uniformly accelerated motion", "Motion in a plane",
            "Projectile motion", "Uniform circular motion",
        ],
        "Laws Of Motion": [
            "Newton's first law", "Momentum and impulse", "Newton's second law",
            "Newton's third law", "Friction", "Circular motion and centripetal force",
        ],
        "Work Energy And Power": [
            "Work done by constant force", "Work-energy theorem",
            "Potential energy", "Conservation of energy",
            "Power", "Collisions elastic and inelastic",
        ],
        "Rotational Motion": [
            "Centre of mass", "Moment of inertia", "Torque",
            "Angular momentum", "Conservation of angular momentum", "Rolling motion",
        ],
        "Gravitation": [
            "Universal law of gravitation", "Acceleration due to gravity",
            "Gravitational potential energy", "Escape velocity",
            "Orbital velocity", "Kepler's laws",
        ],
        "Properties Of Solids And Liquids": [
            "Elastic behaviour", "Young's modulus", "Viscosity",
            "Surface tension", "Bernoulli's theorem", "Pascal's law",
        ],
        "Thermodynamics": [
            "Thermal equilibrium", "Zeroth law", "Specific heat capacity",
            "Calorimetry", "Latent heat", "Newton's law of cooling",
        ],
        "Kinetic Theory Of Gases": [
            "Equation of state", "RMS speed", "Degrees of freedom",
            "Law of equipartition", "Mean free path",
        ],
        "Thermodynamics Laws": [
            "First law of thermodynamics", "Second law", "Carnot engine",
            "Isothermal and adiabatic processes",
        ],
        "Oscillations And Waves": [
            "Simple harmonic motion", "Spring mass system", "Simple pendulum",
            "Damped oscillations", "Wave motion", "Speed of sound",
            "Doppler effect", "Superposition and beats",
        ],
        "Electrostatics": [
            "Coulomb's law", "Electric field", "Electric dipole",
            "Gauss's theorem", "Electric potential", "Capacitors",
        ],
        "Current Electricity": [
            "Ohm's law", "Drift velocity and resistivity",
            "Kirchhoff's laws", "Wheatstone bridge", "Potentiometer",
        ],
        "Magnetic Effects Of Current": [
            "Biot-Savart law", "Ampere's law", "Force on moving charge",
            "Torque on current loop", "Moving coil galvanometer",
        ],
        "Electromagnetic Induction": [
            "Faraday's law", "Lenz's law", "Self and mutual inductance", "AC generator",
        ],
        "Alternating Current": [
            "AC voltage and current", "LCR circuit", "Resonance", "Transformer",
        ],
        "Electromagnetic Waves": [
            "Displacement current", "Electromagnetic spectrum", "Properties of EM waves",
        ],
        "Optics": [
            "Reflection and refraction", "Total internal reflection",
            "Lenses and thin lens formula", "Microscope and telescope",
            "Wave optics", "Interference", "Diffraction",
            "Young's double slit experiment",
        ],
        "Dual Nature Of Matter": [
            "Photoelectric effect", "Einstein's equation",
            "De Broglie wavelength", "Davisson-Germer experiment",
        ],
        "Atoms And Nuclei": [
            "Bohr model", "Hydrogen spectrum", "Radioactivity",
            "Nuclear fission and fusion", "Mass-energy relation",
        ],
        "Electronic Devices": [
            "p-n junction diode", "Zener diode", "Transistors", "Logic gates",
        ],
    },
    "Chemistry": {
        "Some Basic Concepts Of Chemistry": [
            "Mole concept", "Stoichiometry", "Atomic and molecular masses",
            "Empirical and molecular formula",
        ],
        "Structure Of Atom": [
            "Bohr's model", "Quantum numbers", "Electronic configuration",
            "Shapes of orbitals", "Aufbau and Hund's rule",
        ],
        "Classification Of Elements": [
            "Modern periodic table", "Periodic trends",
            "Ionization enthalpy", "Electronegativity",
        ],
        "Chemical Bonding": [
            "Ionic bond", "Covalent bond", "VSEPR theory",
            "Hybridization", "Molecular orbital theory", "Hydrogen bonding",
        ],
        "States Of Matter": [
            "Gas laws", "Ideal gas equation", "Real gases", "Van der Waals equation",
        ],
        "Chemical Thermodynamics": [
            "Enthalpy", "Hess's law", "Entropy", "Gibbs free energy", "Bond enthalpy",
        ],
        "Equilibrium": [
            "Chemical equilibrium", "Le Chatelier's principle",
            "Ionic equilibrium", "pH scale", "Buffer solutions",
        ],
        "Redox Reactions": [
            "Oxidation and reduction", "Balancing redox", "Electrode potential",
        ],
        "Hydrogen": [
            "Position in periodic table", "Isotopes", "Properties of water",
        ],
        "S Block Elements": [
            "Alkali metals", "Alkaline earth metals", "Anomalous properties",
        ],
        "P Block Elements": [
            "Boron family", "Carbon family", "Nitrogen family",
            "Halogen family", "Noble gases",
        ],
        "Organic Chemistry Basic Principles": [
            "IUPAC nomenclature", "Isomerism", "Electronic effects",
            "Reaction intermediates", "Types of organic reactions",
        ],
        "Hydrocarbons": [
            "Alkanes", "Alkenes", "Alkynes", "Aromatic hydrocarbons", "Benzene structure",
        ],
        "Environmental Chemistry": [
            "Air and water pollution", "Ozone depletion", "Greenhouse effect",
        ],
        "Solid State": [
            "Crystal lattices", "Unit cell", "Packing efficiency", "Defects in solids",
        ],
        "Solutions": [
            "Concentration terms", "Colligative properties",
            "Raoult's law", "Osmotic pressure", "Van't Hoff factor",
        ],
        "Electrochemistry": [
            "Galvanic cells", "Nernst equation", "Conductance",
            "Faraday's laws", "Batteries and corrosion",
        ],
        "Chemical Kinetics": [
            "Rate of reaction", "Rate law and order",
            "Activation energy", "Arrhenius equation",
        ],
        "Surface Chemistry": [
            "Adsorption", "Catalysis", "Colloids", "Emulsions",
        ],
        "D And F Block Elements": [
            "Transition elements", "KMnO4 and K2Cr2O7", "Lanthanoids",
        ],
        "Coordination Compounds": [
            "Werner's theory", "IUPAC nomenclature", "Crystal field theory",
        ],
        "Haloalkanes And Haloarenes": [
            "SN1 and SN2 reactions", "Elimination reactions", "Grignard reagent",
        ],
        "Alcohols Phenols Ethers": [
            "Preparation of alcohols", "Phenol acidity", "Ethers synthesis",
        ],
        "Aldehydes Ketones Carboxylic Acids": [
            "Nucleophilic addition", "Aldol condensation", "Carboxylic acid reactions",
        ],
        "Amines": [
            "Classification", "Basicity of amines", "Diazonium salts",
        ],
        "Biomolecules": [
            "Carbohydrates", "Proteins", "Nucleic acids", "Vitamins and enzymes",
        ],
        "Polymers": [
            "Types of polymerization", "Rubber", "Synthetic polymers",
        ],
        "Chemistry In Everyday Life": [
            "Drugs and medicines", "Cleansing agents", "Food chemistry",
        ],
    },
    "Biology": {
        "Diversity In Living World": [
            "Biological classification", "Five kingdom classification",
            "Plant kingdom", "Animal kingdom", "Viruses and viroids",
        ],
        "Structural Organisation In Plants And Animals": [
            "Morphology of flowering plants", "Anatomy of flowering plants",
            "Animal tissues", "Organ systems",
        ],
        "Cell Structure And Function": [
            "Cell theory", "Prokaryotic and eukaryotic cells",
            "Cell organelles", "Cell membrane", "Mitosis and meiosis",
        ],
        "Plant Physiology": [
            "Transport in plants", "Mineral nutrition",
            "Photosynthesis", "Respiration in plants",
            "Plant growth and development",
        ],
        "Human Physiology": [
            "Digestion and absorption", "Breathing and gas exchange",
            "Body fluids and circulation", "Excretory products",
            "Locomotion and movement", "Neural control", "Endocrine system",
        ],
        "Reproduction": [
            "Reproduction in organisms", "Sexual reproduction in plants",
            "Human reproduction", "Reproductive health",
        ],
        "Genetics And Evolution": [
            "Mendel's laws", "Chromosomal theory", "Sex determination",
            "Molecular basis of inheritance", "DNA replication and transcription",
            "Gene expression", "Evolution", "Hardy-Weinberg principle",
        ],
        "Biology And Human Welfare": [
            "Immunity and health", "AIDS and cancer", "Microbes in welfare",
        ],
        "Biotechnology": [
            "Recombinant DNA technology", "PCR and gel electrophoresis",
            "Biotechnology applications", "GM organisms", "Gene therapy",
        ],
        "Ecology And Environment": [
            "Organisms and populations", "Ecosystem",
            "Biodiversity", "Environmental issues", "Ecological pyramids",
        ],
    },
}

# ── JEE 3-level syllabus (Subject → Chapter → Sub-chapters as rows) ─────────
# For JEE, chapter = broad unit (Mechanics), sub_chapter = specific unit (Kinematics)
# micro_topic = the actual concept tested

JEE_SYLLABUS = {
    "Physics": {
        "Mechanics": {
            "Kinematics": ["1D motion", "2D motion", "Relative velocity", "Projectile"],
            "Newton's Laws": ["Free body diagrams", "Friction", "Pulley systems", "Pseudo force"],
            "Work Power Energy": ["Work theorem", "Conservative forces", "Collisions"],
            "Centre Of Mass": ["COM motion", "Momentum conservation", "Impulse"],
            "Rotational Mechanics": ["Moment of inertia", "Torque", "Rolling", "Angular momentum"],
            "Gravitation": ["Kepler's laws", "Escape velocity", "Orbital motion"],
            "Simple Harmonic Motion": ["SHM equations", "Spring systems", "Pendulum", "Energy in SHM"],
            "Fluid Mechanics": ["Bernoulli", "Viscosity", "Surface tension", "Buoyancy"],
        },
        "Waves And Thermodynamics": {
            "Wave Motion": ["Transverse and longitudinal", "Speed of wave", "Superposition"],
            "Sound Waves": ["Doppler effect", "Standing waves", "Beats", "Resonance"],
            "Heat And Temperature": ["Thermal expansion", "Calorimetry", "Heat transfer"],
            "KTG And Thermodynamics": ["Ideal gas", "First and second law", "Carnot", "Entropy"],
        },
        "Electromagnetism": {
            "Electrostatics": ["Coulomb's law", "Electric field", "Gauss law", "Potential"],
            "Capacitance": ["Capacitor combinations", "Energy stored", "Dielectrics"],
            "Current Electricity": ["Ohm's law", "Kirchhoff", "Wheatstone", "RC circuits"],
            "Magnetic Effect": ["Biot-Savart", "Ampere's law", "Lorentz force", "Cyclotron"],
            "Electromagnetic Induction": ["Faraday", "Lenz", "Inductance", "Eddy currents"],
            "Alternating Current": ["RMS values", "LCR circuit", "Resonance", "Transformer"],
        },
        "Optics": {
            "Geometrical Optics": ["Mirrors", "Refraction", "TIR", "Lenses", "Optical instruments"],
            "Wave Optics": ["YDSE", "Interference", "Diffraction", "Polarization"],
        },
        "Modern Physics": {
            "Photoelectric Effect": ["Einstein's equation", "Work function", "Stopping potential"],
            "Atomic Models": ["Bohr model", "Hydrogen spectrum", "De Broglie"],
            "Nuclear Physics": ["Radioactivity", "Nuclear reactions", "Binding energy"],
            "Semiconductors": ["p-n junction", "Transistors", "Logic gates"],
        },
    },
    "Chemistry": {
        "Physical Chemistry": {
            "Mole Concept": ["Stoichiometry", "Limiting reagent", "Percentage composition"],
            "Atomic Structure": ["Quantum numbers", "Electronic configuration", "Orbitals"],
            "Chemical Bonding": ["VSEPR", "Hybridization", "MOT", "Hydrogen bonding"],
            "Gaseous State": ["Gas laws", "Real gases", "KTG"],
            "Chemical Thermodynamics": ["Enthalpy", "Hess's law", "Gibbs energy", "Entropy"],
            "Chemical Equilibrium": ["Le Chatelier", "Kc Kp", "Reaction quotient"],
            "Ionic Equilibrium": ["Acids and bases", "pH", "Buffer", "Solubility product"],
            "Chemical Kinetics": ["Rate law", "Order", "Activation energy", "Arrhenius"],
            "Electrochemistry": ["Cell potential", "Nernst", "Faraday", "Conductance"],
            "Solutions": ["Colligative properties", "Raoult's law", "Osmotic pressure"],
            "Surface Chemistry": ["Adsorption", "Colloids", "Catalysis"],
            "Solid State": ["Crystal structure", "Unit cell", "Defects"],
        },
        "Inorganic Chemistry": {
            "Periodic Table": ["Trends", "Ionization energy", "Electronegativity"],
            "s-Block Elements": ["Alkali metals", "Alkaline earth metals"],
            "p-Block Elements": ["Groups 13-18", "Important compounds"],
            "d-Block Elements": ["Transition metals", "KMnO4", "K2Cr2O7"],
            "Coordination Chemistry": ["Werner's theory", "CFT", "IUPAC naming"],
            "Metallurgy": ["Extraction of metals", "Alloys"],
            "Qualitative Analysis": ["Cation tests", "Anion tests"],
            "Hydrogen": ["Isotopes", "Hydrides", "Water chemistry"],
        },
        "Organic Chemistry": {
            "General Organic Chemistry": ["Isomerism", "Electronic effects", "Intermediates"],
            "Hydrocarbons": ["Alkane reactions", "Alkene additions", "Aromatic substitution"],
            "Halides": ["SN1 SN2", "Elimination", "Grignard"],
            "Oxygen Compounds": ["Alcohols", "Phenols", "Ethers", "Aldehydes ketones"],
            "Nitrogen Compounds": ["Amines", "Diazonium salts", "Aromatic amines"],
            "Biomolecules": ["Carbohydrates", "Proteins", "Nucleic acids"],
            "Polymers": ["Addition polymers", "Condensation polymers"],
            "Practical Organic": ["Mechanisms", "Named reactions", "Multi-step synthesis"],
        },
    },
    "Mathematics": {
        "Algebra": {
            "Quadratic Equations": ["Roots and discriminant", "Nature of roots", "Quadratic inequalities"],
            "Complex Numbers": ["Modulus argument", "De Moivre", "Roots of unity"],
            "Sequences And Series": ["AP GP HP", "Sum formulas", "Special sequences"],
            "Permutation Combination": ["Counting principles", "Arrangements", "Selections"],
            "Binomial Theorem": ["General term", "Coefficient finding", "Middle term"],
            "Matrices Determinants": ["Operations", "Inverse", "Cramer's rule", "Rank"],
            "Mathematical Induction": ["Proof by induction", "Divisibility proofs"],
        },
        "Trigonometry": {
            "Trigonometric Functions": ["Identities", "Graphs", "Transformations"],
            "Trigonometric Equations": ["General solutions", "Principal values"],
            "Inverse Trigonometry": ["Domain range", "Identities", "Equations"],
            "Properties Of Triangles": ["Sine rule", "Cosine rule", "Area formulas"],
        },
        "Coordinate Geometry": {
            "Straight Lines": ["Slope forms", "Distance formulas", "Angle bisectors"],
            "Circles": ["Standard forms", "Chord of contact", "Family of circles"],
            "Parabola": ["Standard forms", "Tangents and normals", "Chords"],
            "Ellipse": ["Standard form", "Focal properties", "Tangents"],
            "Hyperbola": ["Standard form", "Asymptotes", "Conjugate hyperbola"],
        },
        "Calculus": {
            "Limits": ["L'Hopital", "Standard limits", "Sandwich theorem"],
            "Continuity And Differentiability": ["Types of discontinuity", "MVT"],
            "Differentiation": ["Chain rule", "Implicit", "Parametric", "Higher order"],
            "Application Of Derivatives": ["Maxima minima", "Tangents normals", "Rate of change"],
            "Indefinite Integrals": ["Standard forms", "Substitution", "Partial fractions"],
            "Definite Integrals": ["Properties", "Newton-Leibniz", "Gamma function"],
            "Area Under Curves": ["Region bounded", "Curve tracing"],
            "Differential Equations": ["Variable separable", "Linear", "Homogeneous"],
        },
        "Vectors And 3D": {
            "Vectors": ["Dot cross product", "Scalar triple product", "Vector equations"],
            "3D Geometry": ["Direction cosines", "Line in 3D", "Plane", "Distance formulas"],
        },
        "Probability And Statistics": {
            "Probability": ["Classical", "Conditional probability", "Bayes theorem"],
            "Distributions": ["Binomial", "Poisson", "Normal distribution"],
            "Statistics": ["Mean median mode", "Standard deviation", "Correlation"],
        },
    },
}

# ── NEET questions per chapter (realistic distribution, ~180 total: 45+45+90) ─
NEET_QS_PER_CHAPTER = {
    "Physics": {
        "Physical World And Measurement": 1,
        "Kinematics": 3,
        "Laws Of Motion": 3,
        "Work Energy And Power": 2,
        "Rotational Motion": 2,
        "Gravitation": 2,
        "Properties Of Solids And Liquids": 2,
        "Thermodynamics": 2,
        "Kinetic Theory Of Gases": 1,
        "Thermodynamics Laws": 1,
        "Oscillations And Waves": 3,
        "Electrostatics": 4,
        "Current Electricity": 3,
        "Magnetic Effects Of Current": 3,
        "Electromagnetic Induction": 2,
        "Alternating Current": 2,
        "Electromagnetic Waves": 1,
        "Optics": 4,
        "Dual Nature Of Matter": 2,
        "Atoms And Nuclei": 3,
        "Electronic Devices": 2,
    },
    "Chemistry": {
        "Some Basic Concepts Of Chemistry": 2,
        "Structure Of Atom": 2,
        "Classification Of Elements": 1,
        "Chemical Bonding": 3,
        "States Of Matter": 1,
        "Chemical Thermodynamics": 2,
        "Equilibrium": 3,
        "Redox Reactions": 1,
        "Hydrogen": 1,
        "S Block Elements": 1,
        "P Block Elements": 2,
        "Organic Chemistry Basic Principles": 3,
        "Hydrocarbons": 2,
        "Environmental Chemistry": 1,
        "Solid State": 2,
        "Solutions": 2,
        "Electrochemistry": 2,
        "Chemical Kinetics": 2,
        "Surface Chemistry": 1,
        "D And F Block Elements": 2,
        "Coordination Compounds": 2,
        "Haloalkanes And Haloarenes": 2,
        "Alcohols Phenols Ethers": 2,
        "Aldehydes Ketones Carboxylic Acids": 2,
        "Amines": 1,
        "Biomolecules": 2,
        "Polymers": 1,
        "Chemistry In Everyday Life": 1,
    },
    "Biology": {
        "Diversity In Living World": 9,
        "Structural Organisation In Plants And Animals": 8,
        "Cell Structure And Function": 10,
        "Plant Physiology": 10,
        "Human Physiology": 15,
        "Reproduction": 10,
        "Genetics And Evolution": 13,
        "Biology And Human Welfare": 7,
        "Biotechnology": 8,
        "Ecology And Environment": 10,
    },
}

# ── JEE questions per sub-chapter (realistic) ───────────────────────────────
JEE_QS_PER_SUBCHAPTER = 3  # ~3 questions per sub-chapter on average

# ── Student profiles ─────────────────────────────────────────────────────────
FIRST_NAMES = [
    "Aarav","Vivaan","Aditya","Vihaan","Arjun","Sai","Reyansh","Ayaan","Krishna","Ishaan",
    "Shaurya","Atharv","Advik","Pranav","Advait","Dhruv","Kabir","Ritvik","Aarush","Arnav",
    "Anaya","Diya","Saanvi","Aanya","Aadhya","Avni","Riya","Meera","Navya","Kiara","Isha",
    "Priya","Neha","Shreya","Kavya","Anushka","Tanvi","Pooja","Divya","Sneha",
    "Rohan","Raj","Karan","Vikram","Rahul","Siddharth","Nikhil","Tarun","Harsh","Dev",
    "Mihir","Parth","Yash","Varun","Akash","Naman","Shiv","Om","Lakshay","Gaurav",
    "Deepak","Sumit","Rohit","Aman","Ravi","Vijay","Suresh","Ajay","Amit","Pawan",
    "Ritesh","Ashish","Manish","Vishal","Pratik","Lokesh","Niraj","Sachin","Tarun","Kunal",
    "Ritu","Suman","Rekha","Geeta","Sunita","Savita","Lata","Sudha","Meena","Usha",
    "Anita","Seema","Vandana","Nisha","Mamta","Radha","Puja","Pinki","Mona","Reena",
    "Arun","Bhavesh","Chirag","Daksh","Eshan","Faiz","Girish","Hitesh","Ishan","Jayesh",
    "Kartik","Lalit","Mohit","Neeraj","Omkar","Paras","Qasim","Ritesh","Sourav","Tanmay",
    "Ujjwal","Vedant","Waqar","Ximena","Yashasvi","Zaid","Atish","Bharat","Chetan","Dinesh",
]
LAST_NAMES = [
    "Sharma","Verma","Singh","Kumar","Gupta","Mishra","Yadav","Tiwari","Pandey","Dubey",
    "Joshi","Patel","Shah","Mehta","Agarwal","Saxena","Srivastava","Chaturvedi","Shukla","Tripathi",
    "Rao","Reddy","Nair","Pillai","Menon","Iyer","Krishnan","Subramaniam","Bhat","Kaur",
    "Malhotra","Kapoor","Sethi","Grover","Bhatia","Anand","Chandra","Dutta","Ghosh","Bose",
]
CITIES = [
    "Delhi","Mumbai","Bangalore","Hyderabad","Chennai","Kolkata","Jaipur","Lucknow",
    "Pune","Ahmedabad","Kota","Chandigarh","Patna","Bhopal","Indore","Surat","Nagpur",
    "Visakhapatnam","Coimbatore","Vadodara",
]
COACHING = [
    "Allen Kota","FIITJEE","Aakash","Resonance","Narayana","PACE","Vibrant","Self Study",
    "Motion IIT JEE","Bansal Classes","Career Point","Sri Chaitanya",
]

def generate_students(n=200):
    students = []
    used = set()
    for i in range(n):
        while True:
            fn = random.choice(FIRST_NAMES)
            ln = random.choice(LAST_NAMES)
            name = f"{fn} {ln}"
            if name not in used:
                used.add(name)
                break
        # Latent ability per subject, Gaussian with realistic mean/std
        phy = max(0.1, min(0.98, random.gauss(0.60, 0.18)))
        chem = max(0.1, min(0.98, random.gauss(0.62, 0.16)))
        bio = max(0.1, min(0.98, random.gauss(0.65, 0.15)))
        maths = max(0.1, min(0.98, random.gauss(0.55, 0.20)))
        students.append({
            "student_id": f"STU{i+1:03d}",
            "name": name,
            "city": random.choice(CITIES),
            "coaching": random.choice(COACHING),
            "target": random.choice(["NEET", "JEE", "Both"]),
            "ability_physics": round(phy, 4),
            "ability_chemistry": round(chem, 4),
            "ability_biology": round(bio, 4),
            "ability_mathematics": round(maths, 4),
        })
    return students

# ── Helper: derive sub_chapter from micro_topics list position ───────────────
def get_sub_chapter(micro_topics, idx):
    n = len(micro_topics)
    if n <= 3:
        return "Fundamentals"
    elif idx < n // 2:
        return "Fundamentals"
    else:
        return "Advanced Applications"

# ── Simulate single topic performance ────────────────────────────────────────
def simulate(ability, exam_no, chapter_idx, n_chapters, n_qs, marks_correct, marks_wrong):
    growth = 0.07 * math.log1p(exam_no - 1)
    day_factor = random.gauss(0.0, 0.08)
    t_diff = 0.70 + 0.15 * math.sin(2 * math.pi * chapter_idx / max(n_chapters, 1))
    p_correct = max(0.05, min(0.97, ability * t_diff + growth + day_factor))
    p_attempt = max(0.50, min(1.0, 0.70 + 0.25 * ability + 0.04 * (exam_no - 1) / 9))
    attempted = max(0, min(n_qs, int(round(n_qs * p_attempt + random.gauss(0, 0.3)))))
    correct = sum(1 for _ in range(attempted) if random.random() < p_correct)
    correct = min(correct, attempted)
    wrong = max(0, attempted - correct)
    score = correct * marks_correct - wrong * marks_wrong
    accuracy = round(100.0 * correct / attempted, 1) if attempted > 0 else 0.0
    time_min = round(
        correct * random.gauss(1.8, 0.3) + wrong * random.gauss(2.5, 0.5), 1
    )
    return {
        "attempted": attempted, "correct": correct, "wrong": wrong,
        "not_attempted": n_qs - attempted,
        "score": round(score, 1), "max_score": n_qs * marks_correct,
        "accuracy_pct": accuracy, "time_min": max(0.0, time_min),
    }

def exam_dates(n, year=2024, gap=18):
    base = datetime.date(year, 1, 15)
    return [(base + datetime.timedelta(days=i * gap)).isoformat() for i in range(n)]

NEET_DATES = exam_dates(10)
JEE_DATES = exam_dates(10, gap=21)

# ── Generate NEET data ────────────────────────────────────────────────────────
def generate_neet(students):
    results, summary = [], []
    chapters_flat = []  # (subject, chapter, micro_topics_list)
    for subj, chapters in NEET_SYLLABUS.items():
        for chap, micro_list in chapters.items():
            chapters_flat.append((subj, chap, micro_list))
    n_chapters = len(chapters_flat)

    for exam_no in range(1, 11):
        label = f"NEET Mock {exam_no:02d}"
        date = NEET_DATES[exam_no - 1]
        totals = {}

        for stu in students:
            sid, sname = stu["student_id"], stu["name"]
            total_score, total_max = 0.0, 0

            for idx, (subj, chap, micro_list) in enumerate(chapters_flat):
                ability = stu[f"ability_{subj.lower()}"]
                n_qs = NEET_QS_PER_CHAPTER.get(subj, {}).get(chap, 2)
                # Pick which micro-topic was the focus this exam
                mt_idx = (exam_no - 1 + idx) % len(micro_list)
                micro_topic = micro_list[mt_idx]
                sub_chap = get_sub_chapter(micro_list, mt_idx)

                perf = simulate(ability, exam_no, idx, n_chapters, n_qs, 4, 1)
                results.append({
                    "student_id": sid, "name": sname,
                    "exam_no": exam_no, "exam_date": date, "exam_label": label,
                    "subject": subj, "chapter": chap,
                    "sub_chapter": sub_chap, "micro_topic": micro_topic,
                    "total_qs": n_qs,
                    "attempted": perf["attempted"], "correct": perf["correct"],
                    "wrong": perf["wrong"], "not_attempted": perf["not_attempted"],
                    "score": perf["score"], "max_score": perf["max_score"],
                    "accuracy_pct": perf["accuracy_pct"], "time_min": perf["time_min"],
                })
                total_score += perf["score"]
                total_max += perf["max_score"]

            totals[sid] = {
                "student_id": sid, "name": sname,
                "exam_no": exam_no, "exam_date": date, "exam_label": label,
                "total_score": round(total_score, 1), "max_score": total_max,
                "percentage": round(100 * total_score / total_max, 2) if total_max else 0,
            }

        sorted_ids = sorted(totals, key=lambda x: totals[x]["total_score"], reverse=True)
        for rank, sid in enumerate(sorted_ids, 1):
            totals[sid]["rank"] = rank
            totals[sid]["percentile"] = round(100 * (len(students) - rank) / max(len(students) - 1, 1), 2)
            summary.append(totals[sid])

        print(f"  NEET exam {exam_no:02d} — {len(results):,} rows")
    return results, summary

# ── Generate JEE data ─────────────────────────────────────────────────────────
def generate_jee(students):
    results, summary = [], []
    sub_chapters_flat = []  # (subject, chapter, sub_chapter, micro_topics_list)
    for subj, chapters in JEE_SYLLABUS.items():
        for chap, sub_chapters in chapters.items():
            for sub_chap, micro_list in sub_chapters.items():
                sub_chapters_flat.append((subj, chap, sub_chap, micro_list))
    n_subs = len(sub_chapters_flat)

    for exam_no in range(1, 11):
        label = f"JEE Main Mock {exam_no:02d}"
        date = JEE_DATES[exam_no - 1]
        totals = {}

        for stu in students:
            sid, sname = stu["student_id"], stu["name"]
            total_score, total_max = 0.0, 0

            for idx, (subj, chap, sub_chap, micro_list) in enumerate(sub_chapters_flat):
                key = f"ability_{subj.lower()}"
                ability = stu.get(key, stu["ability_physics"])
                n_qs = JEE_QS_PER_SUBCHAPTER
                mt_idx = (exam_no - 1 + idx) % len(micro_list)
                micro_topic = micro_list[mt_idx]

                perf = simulate(ability, exam_no, idx, n_subs, n_qs, 4, 1)
                results.append({
                    "student_id": sid, "name": sname,
                    "exam_no": exam_no, "exam_date": date, "exam_label": label,
                    "subject": subj, "chapter": chap,
                    "sub_chapter": sub_chap, "micro_topic": micro_topic,
                    "total_qs": n_qs,
                    "attempted": perf["attempted"], "correct": perf["correct"],
                    "wrong": perf["wrong"], "not_attempted": perf["not_attempted"],
                    "score": perf["score"], "max_score": perf["max_score"],
                    "accuracy_pct": perf["accuracy_pct"], "time_min": perf["time_min"],
                })
                total_score += perf["score"]
                total_max += perf["max_score"]

            totals[sid] = {
                "student_id": sid, "name": sname,
                "exam_no": exam_no, "exam_date": date, "exam_label": label,
                "total_score": round(total_score, 1), "max_score": total_max,
                "percentage": round(100 * total_score / total_max, 2) if total_max else 0,
            }

        sorted_ids = sorted(totals, key=lambda x: totals[x]["total_score"], reverse=True)
        for rank, sid in enumerate(sorted_ids, 1):
            totals[sid]["rank"] = rank
            totals[sid]["percentile"] = round(100 * (len(students) - rank) / max(len(students) - 1, 1), 2)
            summary.append(totals[sid])

        print(f"  JEE exam  {exam_no:02d} — {len(results):,} rows")
    return results, summary

def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved {len(rows):,} rows → {path.name}")

def main():
    print("PRAJNA Student Data Generator v2")
    print("  Hierarchy: Subject → Chapter → Sub-chapter → Micro-topic")

    students = generate_students(200)
    write_csv(students, OUT_DIR / "students_v2.csv")

    print(f"\nGenerating NEET results ({len(students)} students × 10 exams × 59 chapters) …")
    neet_res, neet_sum = generate_neet(students)
    write_csv(neet_res, OUT_DIR / "neet_results_v2.csv")
    write_csv(neet_sum, OUT_DIR / "neet_summary_v2.csv")

    print(f"\nGenerating JEE Main results ({len(students)} students × 10 exams × ~60 sub-chapters) …")
    jee_res, jee_sum = generate_jee(students)
    write_csv(jee_res, OUT_DIR / "jee_results_v2.csv")
    write_csv(jee_sum, OUT_DIR / "jee_summary_v2.csv")

    print(f"\n✓ Done! NEET: {len(neet_res):,} rows | JEE: {len(jee_res):,} rows")

if __name__ == "__main__":
    main()
