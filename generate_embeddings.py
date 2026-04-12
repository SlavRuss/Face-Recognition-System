import os
import csv
import traceback

from face_model import (
    insert_employee_if_not_exists,
    update_employee_embedding,
    employee_has_embedding,
    FaceModelError
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "employees.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
LIMIT = 5
FORCE_REBUILD = False  # True, если нужно пересчитать embeddings заново


def main():
    success_count = 0
    error_count = 0
    skip_count = 0

    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Файл CSV не найден: {CSV_PATH}")
        return

    if not os.path.exists(IMAGES_DIR):
        print(f"[ERROR] Папка с изображениями не найдена: {IMAGES_DIR}")
        return

    with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        required_columns = {"employee_id", "full_name", "position", "filename"}
        missing_columns = required_columns - set(reader.fieldnames or [])

        if missing_columns:
            print(f"[ERROR] В CSV отсутствуют колонки: {', '.join(missing_columns)}")
            return

        for i, row in enumerate(reader):
            if i >= LIMIT:
                print(f"\n[INFO] Достигнут лимит {LIMIT} записей")
                break

            try:
                employee_id = int(row["employee_id"])
                full_name = row["full_name"].strip()
                position = row["position"].strip()
                filename = row["filename"].strip()

                image_path = os.path.join(IMAGES_DIR, filename)

                if not os.path.exists(image_path):
                    print(f"[SKIP] Файл не найден: {image_path}")
                    error_count += 1
                    continue

                photo_value = filename

                insert_employee_if_not_exists(
                    employee_id=employee_id,
                    full_name=full_name,
                    position=position,
                    photo_value=photo_value
                )

                if not FORCE_REBUILD and employee_has_embedding(employee_id):
                    print(f"[SKIP] Уже есть embedding: {employee_id} - {full_name}")
                    skip_count += 1
                    continue

                update_employee_embedding(
                    employee_id=employee_id,
                    image_path=image_path,
                    photo_value=photo_value,
                    full_name=full_name,
                    position=position
                )

                print(f"[OK] {employee_id} - {full_name}")
                success_count += 1

            except FaceModelError as e:
                print(f"[FACE ERROR] {row.get('filename', 'unknown')} -> {e}")
                error_count += 1

            except Exception as e:
                print(f"[ERROR] {row.get('filename', 'unknown')} -> {e}")
                traceback.print_exc()
                error_count += 1

    print("\n=== ГОТОВО ===")
    print(f"Успешно: {success_count}")
    print(f"Пропущено: {skip_count}")
    print(f"Ошибок:    {error_count}")


if __name__ == "__main__":
    main()