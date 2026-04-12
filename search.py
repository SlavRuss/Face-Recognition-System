import os

from face_model import (
    search_similar_faces,
    save_search_history,
    FaceModelError
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUERY_IMAGE = os.path.join(BASE_DIR, "3.2\человек_справка_373.jpg")

USER_ID = 999  # пока фиксированный пользователь


def main():
    if not os.path.exists(QUERY_IMAGE):
        print(f"[ERROR] Тестовое изображение не найдено: {QUERY_IMAGE}")
        return

    try:
        results = search_similar_faces(
            query_image_path=QUERY_IMAGE,
            top_k=5,
            threshold=0.35
        )

        if not results:
            print("Совпадения не найдены")
            return

        print("\n=== РЕЗУЛЬТАТЫ ПОИСКА ===\n")

        for i, item in enumerate(results, start=1):
            print(
                f"{i}. {item['full_name']} | "
                f"{item['position']} | "
                f"distance={item['distance']} | "
                f"confidence={item['confidence']}%"
            )

        # 🔥 Сохраняем в БД
        search_id = save_search_history(
            user_id=USER_ID,
            query_image_name=os.path.basename(QUERY_IMAGE),
            query_image_value=os.path.basename(QUERY_IMAGE),  # пока просто имя файла
            search_results=results
        )

        print(f"\n[INFO] Поиск сохранён в БД, search_id = {search_id}")

    except FaceModelError as e:
        print(f"[FACE ERROR] {e}")
    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()