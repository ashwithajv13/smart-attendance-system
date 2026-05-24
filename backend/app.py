"""
Smart Attendance System — Backend
Flask + SQLite + PyJWT + numpy

Face embeddings are generated client-side by face-api.js (FaceNet 128-d vectors,
L2-normalised). Cosine similarity = dot product on normalised vectors, so we only
need numpy — no scikit-learn, no dlib, no opencv on the server.

Deployment size: ~85 MB total — well within free-tier limits.
"""

from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone, timedelta
from geopy.distance import geodesic
import os, io, csv, base64, pickle
import jwt
import qrcode
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── App setup ─────────────────────────────────────────────────────────────────

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(BASE_DIR, "database")
os.makedirs(DATABASE_DIR, exist_ok=True)
DATABASE_PATH = os.path.join(DATABASE_DIR, "attendance.db")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "frontend"),
    static_folder=os.path.join(BASE_DIR, "frontend", "static"),
)
app.config["SQLALCHEMY_DATABASE_URI"]        = f"sqlite:///{DATABASE_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

CORS(app, origins=os.getenv("CORS_ORIGINS", "*"))
db = SQLAlchemy(app)

SECRET_KEY        = os.getenv("SECRET_KEY", "change-me-in-production")
JWT_ALGORITHM     = os.getenv("JWT_ALGORITHM", "HS256")
QR_EXPIRY_MINUTES = int(os.getenv("QR_TOKEN_EXPIRY_MINUTES", "10"))
GPS_RADIUS_M      = float(os.getenv("GPS_RADIUS_METRES", "100"))
# face-api.js FaceNet vectors are L2-normalised → cosine sim = dot product
# >= 0.75 is a reliable match; tune down to 0.65 for looser matching
FACE_THRESHOLD    = float(os.getenv("FACE_THRESHOLD", "0.75"))


# ── Models ───────────────────────────────────────────────────────────────────

class Student(db.Model):
    __tablename__ = "students"

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(100), unique=True, nullable=False, index=True)
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200), unique=True, nullable=False)
    # 128-d face embedding from face-api.js, stored as pickled numpy array
    face_encoding = db.Column(db.LargeBinary, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    attendances = db.relationship("Attendance", back_populates="student", lazy=True)

    def to_dict(self):
        return {
            "id": self.id,
            "student_id": self.student_id,
            "name": self.name,
            "email": self.email,
            "has_face": self.face_encoding is not None,
            "created_at": self.created_at.isoformat(),
        }


class Session(db.Model):
    __tablename__ = "sessions"

    id = db.Column(db.Integer, primary_key=True)
    course = db.Column(db.String(200), nullable=False)
    faculty_id = db.Column(db.String(100), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=True)
    location_lat = db.Column(db.Float, nullable=True)
    location_lng = db.Column(db.Float, nullable=True)
    location_name = db.Column(db.String(200), nullable=True)
    qr_token = db.Column(db.Text, nullable=True)
    qr_image_b64 = db.Column(db.Text, nullable=True)   # PNG as base64
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    attendances = db.relationship("Attendance", back_populates="session", lazy=True)

    def to_dict(self, include_qr=False):
        d = {
            "id": self.id,
            "course": self.course,
            "faculty_id": self.faculty_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "location_lat": self.location_lat,
            "location_lng": self.location_lng,
            "location_name": self.location_name,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
        }
        if include_qr:
            d["qr_image_b64"] = self.qr_image_b64
            d["qr_token"] = self.qr_token
        return d


class Attendance(db.Model):
    __tablename__ = "attendance"

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("students.id"), nullable=False)
    session_id = db.Column(db.Integer, db.ForeignKey("sessions.id"), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    gps_lat = db.Column(db.Float, nullable=True)
    gps_lng = db.Column(db.Float, nullable=True)
    gps_distance_m = db.Column(db.Float, nullable=True)
    gps_status = db.Column(db.String(50), default="unknown")

    face_confidence = db.Column(db.Float, nullable=True)
    face_verified = db.Column(db.Boolean, default=False)
    qr_verified = db.Column(db.Boolean, default=False)

    # present / partial / rejected
    status = db.Column(db.String(50), default="pending")

    student = db.relationship("Student", back_populates="attendances")
    session = db.relationship("Session", back_populates="attendances")

    def to_dict(self):
        return {
            "id": self.id,
            "student_id": self.student_id,
            "student_name": self.student.name if self.student else None,
            "session_id": self.session_id,
            "course": self.session.course if self.session else None,
            "timestamp": self.timestamp.isoformat(),
            "gps_lat": self.gps_lat,
            "gps_lng": self.gps_lng,
            "gps_distance_m": round(self.gps_distance_m, 1) if self.gps_distance_m is not None else None,
            "gps_status": self.gps_status,
            "face_confidence": round(self.face_confidence, 4) if self.face_confidence is not None else None,
            "face_verified": self.face_verified,
            "qr_verified": self.qr_verified,
            "status": self.status,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_qr_token(session_id: int, location_lat: float, location_lng: float) -> str:
    """Create a signed JWT embedded in the QR code."""
    now = datetime.now(timezone.utc)
    payload = {
        "session_id": session_id,
        "location_lat": location_lat,
        "location_lng": location_lng,
        "iat": now,
        "exp": now + timedelta(minutes=QR_EXPIRY_MINUTES),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)


def _decode_qr_token(token: str) -> dict:
    """Decode and verify a QR JWT. Raises jwt.ExpiredSignatureError / jwt.InvalidTokenError."""
    return jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])


def _generate_qr_image(data: str) -> str:
    """Generate a QR code PNG and return it as a base64 string."""
    img = qrcode.make(data)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _haversine_distance(lat1, lng1, lat2, lng2) -> float:
    """Return distance in metres between two GPS coordinates."""
    return geodesic((lat1, lng1), (lat2, lng2)).meters


def _compare_face(stored_encoding: bytes, incoming_descriptor: list) -> tuple:
    """
    Compare stored face encoding against incoming 128-d descriptor from face-api.js.

    face-api.js FaceNet vectors are L2-normalised, so:
        cosine_similarity = dot(a, b)  (no division needed)

    Returns (is_match: bool, similarity: float  0.0–1.0).
    """
    stored   = pickle.loads(stored_encoding).astype(np.float64)
    incoming = np.array(incoming_descriptor, dtype=np.float64)

    # Re-normalise defensively (should already be unit vectors)
    stored   = stored   / (np.linalg.norm(stored)   + 1e-10)
    incoming = incoming / (np.linalg.norm(incoming) + 1e-10)

    sim = float(np.dot(stored, incoming))
    return sim >= FACE_THRESHOLD, sim


# ── Routes — Frontend ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/attend")
def attend_page():
    """Student-facing attendance page (opened after QR scan)."""
    return render_template("index.html")


# ── Routes — Students ─────────────────────────────────────────────────────────

@app.route("/api/students", methods=["GET"])
def get_students():
    students = Student.query.order_by(Student.name).all()
    return jsonify([s.to_dict() for s in students])


@app.route("/api/students/<int:student_id>", methods=["GET"])
def get_student(student_id):
    s = Student.query.get_or_404(student_id)
    return jsonify(s.to_dict())


@app.route("/api/students/register", methods=["POST"])
def register_student():
    """Register a new student (no face yet)."""
    data = request.get_json(force=True)
    required = ("student_id", "name", "email")
    if not all(data.get(k) for k in required):
        return jsonify({"error": "student_id, name and email are required"}), 400

    if Student.query.filter_by(student_id=data["student_id"]).first():
        return jsonify({"error": "student_id already registered"}), 409
    if Student.query.filter_by(email=data["email"]).first():
        return jsonify({"error": "email already registered"}), 409

    student = Student(
        student_id=data["student_id"],
        name=data["name"],
        email=data["email"],
    )
    db.session.add(student)
    db.session.commit()
    return jsonify({"message": "Student registered", "student": student.to_dict()}), 201


@app.route("/api/students/<int:student_id>/enroll-face", methods=["POST"])
def enroll_face(student_id):
    """
    Store a face encoding for a student.
    Expects JSON: { "descriptor": [128 floats from face-api.js] }
    """
    student = Student.query.get_or_404(student_id)
    data = request.get_json(force=True)

    descriptor = data.get("descriptor")
    if not descriptor or len(descriptor) != 128:
        return jsonify({"error": "descriptor must be a 128-element float array"}), 400

    arr = np.array(descriptor, dtype=np.float64)
    student.face_encoding = pickle.dumps(arr)
    db.session.commit()
    return jsonify({"message": "Face enrolled successfully"})


@app.route("/api/students/<int:student_id>", methods=["DELETE"])
def delete_student(student_id):
    student = Student.query.get_or_404(student_id)
    Attendance.query.filter_by(student_id=student.id).delete()
    db.session.delete(student)
    db.session.commit()
    return jsonify({"message": "Student removed", "student_id": student_id})


# ── Routes — Sessions ─────────────────────────────────────────────────────────

@app.route("/api/sessions", methods=["GET"])
def get_sessions():
    sessions = Session.query.order_by(Session.created_at.desc()).all()
    return jsonify([s.to_dict() for s in sessions])


@app.route("/api/sessions/<int:session_id>", methods=["GET"])
def get_session(session_id):
    s = Session.query.get_or_404(session_id)
    return jsonify(s.to_dict(include_qr=True))


@app.route("/api/sessions/create", methods=["POST"])
def create_session():
    """
    Create a new class session and generate a time-limited QR code.
    Body: { course, faculty_id, location_lat, location_lng, location_name? }
    """
    data = request.get_json(force=True)
    required = ("course", "faculty_id")
    if not all(data.get(k) for k in required):
        return jsonify({"error": "course and faculty_id are required"}), 400

    lat = data.get("location_lat")
    lng = data.get("location_lng")

    session = Session(
        course=data["course"],
        faculty_id=data["faculty_id"],
        location_lat=lat,
        location_lng=lng,
        location_name=data.get("location_name", ""),
    )
    db.session.add(session)
    db.session.flush()  # get session.id before commit

    # Build the JWT token and QR image
    token = _make_qr_token(session.id, lat or 0.0, lng or 0.0)
    session.qr_token = token

    # The QR encodes a URL the student's phone will open
    attend_url = f"/attend?token={token}"
    session.qr_image_b64 = _generate_qr_image(attend_url)

    db.session.commit()
    return jsonify({
        "message": "Session created",
        "session": session.to_dict(include_qr=True),
    }), 201


@app.route("/api/sessions/<int:session_id>/refresh-qr", methods=["POST"])
def refresh_qr(session_id):
    """Re-generate a fresh QR token for an existing session."""
    session = Session.query.get_or_404(session_id)
    if not session.is_active:
        return jsonify({"error": "Session is closed"}), 400

    token = _make_qr_token(session.id, session.location_lat or 0.0, session.location_lng or 0.0)
    session.qr_token = token
    attend_url = f"/attend?token={token}"
    session.qr_image_b64 = _generate_qr_image(attend_url)
    db.session.commit()
    return jsonify({"message": "QR refreshed", "session": session.to_dict(include_qr=True)})


@app.route("/api/sessions/<int:session_id>/close", methods=["POST"])
def close_session(session_id):
    session = Session.query.get_or_404(session_id)
    session.is_active = False
    session.end_time = datetime.now(timezone.utc)
    db.session.commit()
    return jsonify({"message": "Session closed", "session": session.to_dict()})


# ── Routes — Attendance ───────────────────────────────────────────────────────

@app.route("/api/attendance/mark", methods=["POST"])
def mark_attendance():
    """
    Unified endpoint — all 3 checks happen here.

    Body:
    {
        "token":       "<JWT from QR>",
        "student_id":  "<student_id string>",
        "descriptor":  [128 floats],          // face-api.js embedding
        "gps_lat":     12.345,
        "gps_lng":     67.890,
        "gps_available": true
    }

    Returns status: present / partial / rejected
    """
    data = request.get_json(force=True)

    # ── 1. Validate QR token ──────────────────────────────────────────────────
    token = data.get("token", "")
    try:
        claims = _decode_qr_token(token)
    except jwt.ExpiredSignatureError:
        return jsonify({"error": "QR code has expired. Ask your lecturer to refresh it."}), 401
    except jwt.InvalidTokenError as e:
        return jsonify({"error": f"Invalid QR token: {e}"}), 401

    session_id = claims["session_id"]
    session = Session.query.get(session_id)
    if not session or not session.is_active:
        return jsonify({"error": "Session not found or already closed"}), 404

    qr_ok = True

    # ── 2. Identify student ───────────────────────────────────────────────────
    student = Student.query.filter_by(student_id=data.get("student_id", "")).first()
    if not student:
        return jsonify({"error": "Student not found. Please register first."}), 404

    # Prevent duplicate attendance for same session
    existing = Attendance.query.filter_by(
        student_id=student.id, session_id=session.id
    ).first()
    if existing:
        return jsonify({
            "error": "Attendance already recorded for this session",
            "attendance": existing.to_dict(),
        }), 409

    # ── 3. Face verification ──────────────────────────────────────────────────
    face_ok = False
    face_sim = None
    descriptor = data.get("descriptor")

    if not student.face_encoding:
        # No face enrolled — flag for manual review but don't block
        face_ok = False
        face_sim = None
    elif descriptor and len(descriptor) == 128:
        face_ok, face_sim = _compare_face(student.face_encoding, descriptor)
    else:
        face_ok = False

    # ── 4. GPS verification ───────────────────────────────────────────────────
    gps_available = data.get("gps_available", False)
    gps_lat = data.get("gps_lat")
    gps_lng = data.get("gps_lng")
    gps_distance = None
    gps_status = "unavailable"

    if gps_available and gps_lat is not None and gps_lng is not None:
        if session.location_lat and session.location_lng:
            gps_distance = _haversine_distance(
                session.location_lat, session.location_lng,
                gps_lat, gps_lng,
            )
            gps_status = "ok" if gps_distance <= GPS_RADIUS_M else "out_of_range"
        else:
            # Session has no location set — skip GPS check
            gps_status = "not_required"
    else:
        gps_status = "unavailable"

    gps_ok = gps_status in ("ok", "not_required")

    # ── 5. Determine overall status ───────────────────────────────────────────
    checks_passed = sum([qr_ok, face_ok, gps_ok])
    if checks_passed == 3:
        status = "present"
    elif checks_passed >= 1:
        status = "partial"   # flagged for manual review
    else:
        status = "rejected"

    # ── 6. Save record ────────────────────────────────────────────────────────
    record = Attendance(
        student_id=student.id,
        session_id=session.id,
        gps_lat=gps_lat,
        gps_lng=gps_lng,
        gps_distance_m=gps_distance,
        gps_status=gps_status,
        face_confidence=face_sim,
        face_verified=face_ok,
        qr_verified=qr_ok,
        status=status,
    )
    db.session.add(record)
    db.session.commit()

    return jsonify({
        "message": f"Attendance recorded — {status}",
        "attendance": record.to_dict(),
        "checks": {
            "qr": qr_ok,
            "face": face_ok,
            "gps": gps_ok,
            "gps_distance_m": round(gps_distance, 1) if gps_distance is not None else None,
            "face_similarity": round(face_sim, 4) if face_sim is not None else None,
        },
    }), 201


@app.route("/api/attendance/session/<int:session_id>", methods=["GET"])
def get_session_attendance(session_id):
    records = (
        Attendance.query
        .filter_by(session_id=session_id)
        .order_by(Attendance.timestamp)
        .all()
    )
    return jsonify([r.to_dict() for r in records])


@app.route("/api/attendance/student/<int:student_id>", methods=["GET"])
def get_student_attendance(student_id):
    records = (
        Attendance.query
        .filter_by(student_id=student_id)
        .order_by(Attendance.timestamp.desc())
        .all()
    )
    return jsonify([r.to_dict() for r in records])


@app.route("/api/attendance/today", methods=["GET"])
def get_today_attendance():
    today = datetime.now(timezone.utc).date()
    records = Attendance.query.filter(
        db.func.date(Attendance.timestamp) == today
    ).order_by(Attendance.timestamp.desc()).all()
    return jsonify([r.to_dict() for r in records])


@app.route("/api/attendance/export/csv", methods=["GET"])
def export_csv():
    """Export all attendance as CSV."""
    import csv, io as _io
    session_id = request.args.get("session_id", type=int)
    query = Attendance.query.order_by(Attendance.timestamp.desc())
    if session_id:
        query = query.filter_by(session_id=session_id)
    records = query.all()

    output = _io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID", "Student ID", "Student Name", "Course", "Session ID",
        "Timestamp", "GPS Lat", "GPS Lng", "GPS Distance (m)", "GPS Status",
        "Face Similarity", "Face Verified", "QR Verified", "Status",
    ])
    for r in records:
        writer.writerow([
            r.id,
            r.student.student_id if r.student else "",
            r.student.name if r.student else "",
            r.session.course if r.session else "",
            r.session_id,
            r.timestamp.isoformat(),
            r.gps_lat, r.gps_lng,
            round(r.gps_distance_m, 1) if r.gps_distance_m is not None else "",
            r.gps_status,
            round(r.face_confidence, 4) if r.face_confidence is not None else "",
            r.face_verified, r.qr_verified, r.status,
        ])

    from flask import Response
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=attendance.csv"},
    )


# ── Routes — Stats ────────────────────────────────────────────────────────────

@app.route("/api/stats", methods=["GET"])
def get_stats():
    today = datetime.now(timezone.utc).date()
    total_students = Student.query.count()
    total_sessions = Session.query.count()
    active_sessions = Session.query.filter_by(is_active=True).count()
    today_records = Attendance.query.filter(
        db.func.date(Attendance.timestamp) == today
    ).count()
    present_today = Attendance.query.filter(
        db.func.date(Attendance.timestamp) == today,
        Attendance.status == "present",
    ).count()
    partial_today = Attendance.query.filter(
        db.func.date(Attendance.timestamp) == today,
        Attendance.status == "partial",
    ).count()

    return jsonify({
        "total_students": total_students,
        "total_sessions": total_sessions,
        "active_sessions": active_sessions,
        "today_records": today_records,
        "present_today": present_today,
        "partial_today": partial_today,
        "date": today.isoformat(),
    })


# ── Token validation helper (used by frontend before submitting) ──────────────

@app.route("/api/validate-token", methods=["POST"])
def validate_token():
    """Quick check — is this QR token still valid?"""
    data = request.get_json(force=True)
    token = data.get("token", "")
    try:
        claims = _decode_qr_token(token)
        session = Session.query.get(claims["session_id"])
        if not session or not session.is_active:
            return jsonify({"valid": False, "reason": "Session closed"}), 200
        return jsonify({
            "valid": True,
            "session_id": claims["session_id"],
            "course": session.course,
            "faculty_id": session.faculty_id,
            "location_lat": claims.get("location_lat"),
            "location_lng": claims.get("location_lng"),
        })
    except jwt.ExpiredSignatureError:
        return jsonify({"valid": False, "reason": "QR code expired"}), 200
    except jwt.InvalidTokenError:
        return jsonify({"valid": False, "reason": "Invalid token"}), 200


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(_):
    db.session.rollback()
    return jsonify({"error": "Internal server error"}), 500


# ── Startup ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(
        debug=os.getenv("DEBUG", "True") == "True",
        host=os.getenv("SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("SERVER_PORT", "5000")),
    )
