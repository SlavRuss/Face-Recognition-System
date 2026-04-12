import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from face_model import get_connection, FaceModelError


SESSION_DURATION_DAYS = 7
INACTIVITY_TIMEOUT_MINUTES = 5


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def create_user(username: str, password: str, is_admin: bool = False) -> int:
    username = username.strip()

    if not username:
        raise FaceModelError("Имя пользователя не может быть пустым")

    if not password or len(password) < 4:
        raise FaceModelError("Пароль должен содержать минимум 4 символа")

    password_hash = hash_password(password)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id
                FROM users
                WHERE username = %s
                """,
                (username,)
            )
            existing = cur.fetchone()

            if existing:
                raise FaceModelError("Пользователь с таким именем уже существует")

            cur.execute(
                """
                INSERT INTO users (username, password_hash, is_admin)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (username, password_hash, is_admin)
            )
            user_id = cur.fetchone()[0]

    return user_id


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    username = username.strip()
    password_hash = hash_password(password)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, username, password_hash, is_admin
                FROM users
                WHERE username = %s
                """,
                (username,)
            )
            row = cur.fetchone()

    if row is None:
        return None

    user_id, db_username, db_password_hash, is_admin = row

    if db_password_hash != password_hash:
        return None

    return {
        "id": user_id,
        "username": db_username,
        "is_admin": bool(is_admin),
    }


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, username, is_admin
                FROM users
                WHERE id = %s
                """,
                (user_id,)
            )
            row = cur.fetchone()

    if row is None:
        return None

    return {
        "id": row[0],
        "username": row[1],
        "is_admin": bool(row[2]),
    }


def get_all_users() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, username, is_admin
                FROM users
                ORDER BY id ASC
                """
            )
            rows = cur.fetchall()

    return [
        {
            "id": row[0],
            "username": row[1],
            "is_admin": bool(row[2]),
        }
        for row in rows
    ]


def create_session(user_id: int) -> str:
    session_token = secrets.token_urlsafe(48)
    now = datetime.utcnow()
    expires_at = now + timedelta(days=SESSION_DURATION_DAYS)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_sessions (
                    user_id,
                    session_token,
                    created_at,
                    expires_at,
                    last_activity_at
                )
                VALUES (%s, %s, %s, %s, %s)
                """,
                (user_id, session_token, now, expires_at, now)
            )

    return session_token


def update_session_activity(session_token: str) -> None:
    if not session_token:
        return

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE user_sessions
                SET last_activity_at = NOW()
                WHERE session_token = %s
                """,
                (session_token,)
            )


def get_user_by_session(session_token: str) -> Optional[Dict[str, Any]]:
    if not session_token:
        return None

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    u.id,
                    u.username,
                    u.is_admin,
                    s.expires_at,
                    s.last_activity_at
                FROM user_sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.session_token = %s
                """,
                (session_token,)
            )
            row = cur.fetchone()

    if row is None:
        return None

    user_id, username, is_admin, expires_at, last_activity_at = row

    if expires_at is None or expires_at < datetime.utcnow():
        delete_session(session_token)
        return None

    if last_activity_at is None:
        delete_session(session_token)
        return None

    inactive_limit = datetime.utcnow() - timedelta(minutes=INACTIVITY_TIMEOUT_MINUTES)
    if last_activity_at < inactive_limit:
        delete_session(session_token)
        return None

    return {
        "id": user_id,
        "username": username,
        "is_admin": bool(is_admin),
    }


def is_session_expired_by_inactivity(session_token: str) -> bool:
    if not session_token:
        return True

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT last_activity_at
                FROM user_sessions
                WHERE session_token = %s
                """,
                (session_token,)
            )
            row = cur.fetchone()

    if row is None:
        return True

    last_activity_at = row[0]
    if last_activity_at is None:
        return True

    inactive_limit = datetime.utcnow() - timedelta(minutes=INACTIVITY_TIMEOUT_MINUTES)
    return last_activity_at < inactive_limit


def delete_session(session_token: str) -> None:
    if not session_token:
        return

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM user_sessions
                WHERE session_token = %s
                """,
                (session_token,)
            )


def delete_expired_sessions() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM user_sessions
                WHERE expires_at < NOW()
                   OR last_activity_at < NOW() - INTERVAL '5 minutes'
                """
            )