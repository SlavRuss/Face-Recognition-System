import json
import os
from typing import List, Dict, Any, Optional

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from deepface import DeepFace

from db_config import DB_CONFIG

MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "retinaface"
DISTANCE_THRESHOLD = 0.35


class FaceModelError(Exception):
    pass


class FaceNotFoundError(FaceModelError):
    pass


class MultipleFacesError(FaceModelError):
    pass


def get_connection():
    password = DB_CONFIG.get("password", "").strip()

    if not password or password == "ВСТАВЬ_СЮДА_РЕАЛЬНЫЙ_ПАРОЛЬ":
        raise FaceModelError(
            "Не задан пароль БД. Укажи реальный пароль в db_config.py "
            "или через переменную окружения DB_PASSWORD."
        )

    return psycopg2.connect(**DB_CONFIG)


def validate_image_extension(filename: str) -> bool:
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    ext = os.path.splitext(filename.lower())[1]
    return ext in allowed_extensions


def cosine_distance(vec1: List[float], vec2: List[float]) -> float:
    a = np.array(vec1, dtype=np.float32)
    b = np.array(vec2, dtype=np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 1.0

    similarity = float(np.dot(a, b) / (norm_a * norm_b))
    similarity = max(min(similarity, 1.0), -1.0)

    return 1.0 - similarity


def similarity_percent(distance: float) -> float:
    score = max(0.0, 1.0 - distance)
    return round(score * 100, 2)


def ensure_single_face(image_path: str) -> None:
    faces = DeepFace.extract_faces(
        img_path=image_path,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=True,
        align=True,
    )

    if len(faces) == 0:
        raise FaceNotFoundError(f"Лицо не найдено: {image_path}")

    if len(faces) > 1:
        raise MultipleFacesError(f"На фото найдено больше одного лица: {image_path}")


def generate_embedding(image_path: str) -> List[float]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл не найден: {image_path}")

    if not validate_image_extension(image_path):
        raise ValueError(f"Неподдерживаемый формат изображения: {image_path}")

    ensure_single_face(image_path)

    result = DeepFace.represent(
        img_path=image_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=True,
        align=True,
    )

    if not result or "embedding" not in result[0]:
        raise FaceModelError(f"Не удалось получить embedding для: {image_path}")

    embedding = result[0]["embedding"]
    return [float(x) for x in embedding]


def insert_employee_if_not_exists(
    employee_id: int,
    full_name: str,
    position: str,
    photo_value: str
) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO employees (id, full_name, position, photo, embedding_json)
                VALUES (%s, %s, %s, %s, NULL)
                ON CONFLICT (id) DO UPDATE
                SET full_name = EXCLUDED.full_name,
                    position = EXCLUDED.position,
                    photo = EXCLUDED.photo
                """,
                (employee_id, full_name, position, photo_value),
            )


def employee_has_embedding(employee_id: int) -> bool:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT embedding_json
                FROM employees
                WHERE id = %s
                """,
                (employee_id,)
            )
            row = cur.fetchone()

            if row is None:
                return False

            return row[0] is not None


def update_employee_embedding(
    employee_id: int,
    image_path: str,
    photo_value: str,
    full_name: Optional[str] = None,
    position: Optional[str] = None,
) -> None:
    embedding = generate_embedding(image_path)
    embedding_json = json.dumps(embedding, ensure_ascii=False)

    with get_connection() as conn:
        with conn.cursor() as cur:
            if full_name is not None and position is not None:
                cur.execute(
                    """
                    UPDATE employees
                    SET full_name = %s,
                        position = %s,
                        photo = %s,
                        embedding_json = %s
                    WHERE id = %s
                    """,
                    (full_name, position, photo_value, embedding_json, employee_id),
                )
            else:
                cur.execute(
                    """
                    UPDATE employees
                    SET photo = %s,
                        embedding_json = %s
                    WHERE id = %s
                    """,
                    (photo_value, embedding_json, employee_id),
                )

            if cur.rowcount == 0:
                raise FaceModelError(f"Сотрудник с id={employee_id} не найден в БД")


def get_all_employee_embeddings() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, full_name, position, photo, embedding_json
                FROM employees
                WHERE embedding_json IS NOT NULL
                """
            )
            rows = cur.fetchall()

    result = []
    for row in rows:
        raw_embedding = row["embedding_json"]

        try:
            if isinstance(raw_embedding, str):
                embedding = json.loads(raw_embedding)
            else:
                embedding = raw_embedding

            if embedding:
                result.append(
                    {
                        "id": row["id"],
                        "full_name": row["full_name"],
                        "position": row["position"],
                        "photo": row["photo"],
                        "embedding": embedding,
                    }
                )
        except Exception:
            continue

    return result


def search_similar_faces(
    query_image_path: str,
    top_k: int = 5,
    threshold: float = DISTANCE_THRESHOLD,
) -> List[Dict[str, Any]]:
    query_embedding = generate_embedding(query_image_path)
    employees = get_all_employee_embeddings()

    results = []

    for employee in employees:
        distance = cosine_distance(query_embedding, employee["embedding"])

        if distance <= threshold:
            results.append(
                {
                    "employee_id": employee["id"],
                    "full_name": employee["full_name"],
                    "position": employee["position"],
                    "photo": employee["photo"],
                    "distance": round(distance, 6),
                    "confidence": similarity_percent(distance),
                }
            )

    results.sort(key=lambda x: x["distance"])
    return results[:top_k]


def save_search_history(
    user_id: int,
    query_image_name: str,
    query_image_value: str,
    search_results: List[Dict[str, Any]],
) -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO search_history (user_id, query_image_name, query_image, timestamp)
                VALUES (%s, %s, %s, NOW())
                RETURNING id
                """,
                (user_id, query_image_name, query_image_value),
            )
            search_id = cur.fetchone()[0]

            for item in search_results:
                cur.execute(
                    """
                    INSERT INTO search_results (search_id, employee_id, similarity_score)
                    VALUES (%s, %s, %s)
                    """,
                    (search_id, item["employee_id"], item["distance"]),
                )

    return search_id


def get_existing_user_id() -> Optional[int]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id
                FROM users
                ORDER BY id
                LIMIT 1
                """
            )
            row = cur.fetchone()

            if row is None:
                return None

            return row[0]
        

def get_user_search_history(user_id: int):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, user_id, query_image_name, query_image, timestamp
                FROM search_history
                WHERE user_id = %s
                ORDER BY timestamp DESC, id DESC
                """,
                (user_id,)
            )
            rows = cur.fetchall()

    return [dict(row) for row in rows]


def get_search_results_by_search_id(search_id: int):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    sr.id,
                    sr.search_id,
                    sr.employee_id,
                    sr.similarity_score,
                    e.full_name,
                    e.position,
                    e.photo
                FROM search_results sr
                JOIN employees e ON e.id = sr.employee_id
                WHERE sr.search_id = %s
                ORDER BY sr.similarity_score ASC
                """,
                (search_id,)
            )
            rows = cur.fetchall()

    result = []
    for row in rows:
        distance = float(row["similarity_score"])

        result.append({
            "id": row["id"],
            "search_id": row["search_id"],
            "employee_id": row["employee_id"],
            "similarity_score": distance,
            "confidence": similarity_percent(distance),
            "full_name": row["full_name"],
            "position": row["position"],
            "photo": row["photo"],
        })

    return result

def employee_exists(employee_id: int) -> bool:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM employees
                WHERE id = %s
                """,
                (employee_id,)
            )
            return cur.fetchone() is not None


def create_employee_with_embedding(
    employee_id: int,
    full_name: str,
    position: str,
    image_path: str,
    photo_value: str,
) -> None:
    if employee_exists(employee_id):
        raise FaceModelError(f"Сотрудник с id={employee_id} уже существует")

    embedding = generate_embedding(image_path)
    embedding_json = json.dumps(embedding, ensure_ascii=False)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO employees (id, full_name, position, photo, embedding_json)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (employee_id, full_name, position, photo_value, embedding_json)
            )


def get_all_employees_short() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, full_name, position, photo
                FROM employees
                ORDER BY id ASC
                """
            )
            rows = cur.fetchall()

    return [dict(row) for row in rows]