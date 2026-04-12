import os
import re
import uuid
import streamlit as st
import extra_streamlit_components as stx
from streamlit_autorefresh import st_autorefresh

from auth_utils import (
    create_user,
    authenticate_user,
    get_all_users,
    create_session,
    get_user_by_session,
    delete_session,
    delete_expired_sessions,
    update_session_activity,
    is_session_expired_by_inactivity,
)
from face_model import (
    search_similar_faces,
    save_search_history,
    get_user_search_history,
    get_search_results_by_search_id,
    create_employee_with_embedding,
    get_all_employees_short,
    FaceModelError
)



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🔍",
    layout="wide"
)


def init_session_state():
    if "user" not in st.session_state:
        st.session_state.user = None
    if "page" not in st.session_state:
        st.session_state.page = "search"
    if "logout_message" not in st.session_state:
        st.session_state.logout_message = None
    if "_cookie_manager" not in st.session_state:
        st.session_state["_cookie_manager"] = None
    if "_last_watchdog_count" not in st.session_state:
        st.session_state["_last_watchdog_count"] = 0


def get_cookie_manager():
    if st.session_state["_cookie_manager"] is None:
        st.session_state["_cookie_manager"] = stx.CookieManager()
    return st.session_state["_cookie_manager"]


def get_session_token():
    cookie_manager = get_cookie_manager()
    try:
        return cookie_manager.get("session_token")
    except Exception:
        return None


def is_watchdog_rerun(current_watchdog_count: int) -> bool:
    last_count = st.session_state.get("_last_watchdog_count", 0)
    is_watchdog = current_watchdog_count != last_count
    st.session_state["_last_watchdog_count"] = current_watchdog_count
    return is_watchdog


def restore_user_from_cookie():
    cookie_manager = get_cookie_manager()

    try:
        cookies = cookie_manager.get_all()
    except Exception:
        cookies = {}

    session_token = cookies.get("session_token")

    if not session_token:
        return

    user = get_user_by_session(session_token)

    if user is not None:
        st.session_state.user = user
    else:
        try:
            cookie_manager.delete("session_token")
        except Exception:
            pass


def force_logout_due_to_timeout():
    cookie_manager = get_cookie_manager()
    session_token = get_session_token()

    if session_token:
        delete_session(session_token)
        try:
            cookie_manager.delete("session_token")
        except Exception:
            pass

    st.session_state.user = None
    st.session_state.page = "search"
    st.session_state.logout_message = "Сессия завершена: вы были неактивны более 5 минут."
    st.rerun()


def enforce_inactivity_timeout():
    if st.session_state.user is None:
        return

    session_token = get_session_token()

    if not session_token:
        st.session_state.user = None
        st.session_state.page = "search"
        return

    if is_session_expired_by_inactivity(session_token):
        force_logout_due_to_timeout()


def touch_session_activity():
    if st.session_state.user is None:
        return

    session_token = get_session_token()
    if session_token:
        update_session_activity(session_token)


def logout():
    cookie_manager = get_cookie_manager()
    session_token = get_session_token()

    if session_token:
        delete_session(session_token)
        try:
            cookie_manager.delete("session_token")
        except Exception:
            pass

    st.session_state.user = None
    st.session_state.page = "search"
    st.session_state.logout_message = None
    st.rerun()


def save_temp_uploaded_file(uploaded_file) -> str:
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    unique_name = f"{uuid.uuid4().hex}{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def sanitize_filename(filename: str) -> str:
    name, ext = os.path.splitext(filename)
    name = re.sub(r"[^a-zA-Zа-яА-Я0-9_-]+", "_", name).strip("_")
    if not name:
        name = "employee"
    return f"{name}{ext.lower()}"


def save_employee_photo(uploaded_file, employee_id: int):
    original_name = sanitize_filename(uploaded_file.name)
    _, ext = os.path.splitext(original_name)
    final_filename = f"{employee_id}_{uuid.uuid4().hex[:8]}{ext}"
    final_path = os.path.join(IMAGES_DIR, final_filename)

    with open(final_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return final_path, final_filename


def get_employee_photo_path(photo_filename: str):
    if not photo_filename:
        return None

    photo_path = os.path.join(IMAGES_DIR, photo_filename)

    if os.path.exists(photo_path):
        return photo_path

    return None


def render_auth_page():
    cookie_manager = get_cookie_manager()

    st.title("Система поиска сотрудников по лицу")
    st.subheader("Авторизация")

    if st.session_state.logout_message:
        st.warning(st.session_state.logout_message)
        st.session_state.logout_message = None

    tab1, tab2 = st.tabs(["Вход", "Регистрация"])

    with tab1:
        with st.form("login_form"):
            username = st.text_input("Логин")
            password = st.text_input("Пароль", type="password")
            submitted = st.form_submit_button("Войти")

            if submitted:
                try:
                    user = authenticate_user(username, password)

                    if user is None:
                        st.error("Неверный логин или пароль")
                    else:
                        session_token = create_session(user["id"])

                        try:
                            cookie_manager.set(
                                "session_token",
                                session_token,
                                max_age=60 * 60 * 24 * 7
                            )
                        except Exception:
                            pass

                        st.session_state.user = user
                        st.session_state.page = "search"
                        st.success("Вход выполнен")
                        st.rerun()

                except Exception as e:
                    st.error(f"Ошибка входа: {e}")

    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("Новый логин")
            new_password = st.text_input("Новый пароль", type="password")
            repeat_password = st.text_input("Повторите пароль", type="password")
            submitted = st.form_submit_button("Зарегистрироваться")

            if submitted:
                try:
                    if new_password != repeat_password:
                        st.error("Пароли не совпадают")
                    else:
                        user_id = create_user(new_username, new_password, is_admin=False)
                        st.success(f"Пользователь создан. id = {user_id}")
                except Exception as e:
                    st.error(f"Ошибка регистрации: {e}")


def render_sidebar():
    user = st.session_state.user

    with st.sidebar:
        st.title("Меню")
        st.write(f"Пользователь: **{user['username']}**")
        st.write(f"Роль: **{'Админ' if user.get('is_admin') else 'Пользователь'}**")

        if st.button("Поиск"):
            touch_session_activity()
            st.session_state.page = "search"
            st.rerun()

        if st.button("История поисков"):
            touch_session_activity()
            st.session_state.page = "history"
            st.rerun()

        if user.get("is_admin"):
            if st.button("Админ-панель"):
                touch_session_activity()
                st.session_state.page = "admin"
                st.rerun()

        st.markdown("---")

        if st.button("Выйти"):
            logout()


def render_search_page():
    user = st.session_state.user

    st.title("Поиск сотрудников по лицу")
    st.write(f"Вы вошли как: **{user['username']}**")

    uploaded_file = st.file_uploader(
        "Загрузите фото для поиска",
        type=["jpg", "jpeg", "png"]
    )

    col1, col2 = st.columns(2)

    with col1:
        threshold = st.slider(
            "Порог расстояния",
            min_value=0.10,
            max_value=1.00,
            value=0.35,
            step=0.01
        )

    with col2:
        top_k = st.slider(
            "Количество результатов",
            min_value=1,
            max_value=10,
            value=5,
            step=1
        )

    if uploaded_file is not None:
        st.subheader("Предпросмотр загруженного фото")
        preview_col1, preview_col2, preview_col3 = st.columns([1, 2, 1])
        with preview_col2:
            st.image(
                uploaded_file,
                caption="Загруженное изображение",
                width=260
            )

        if st.button("Найти похожих сотрудников", type="primary"):
            touch_session_activity()
            temp_file_path = None

            try:
                with st.spinner("Выполняется поиск..."):
                    temp_file_path = save_temp_uploaded_file(uploaded_file)

                    results = search_similar_faces(
                        query_image_path=temp_file_path,
                        top_k=top_k,
                        threshold=threshold
                    )

                    if not results:
                        st.warning("Совпадения не найдены")
                    else:
                        search_id = save_search_history(
                            user_id=user["id"],
                            query_image_name=uploaded_file.name,
                            query_image_value=uploaded_file.name,
                            search_results=results
                        )

                        st.success(f"Поиск завершён. search_id = {search_id}")

                        st.subheader("Результаты поиска")
                        for i, item in enumerate(results, start=1):
                            with st.container(border=True):
                                card_col1, card_col2 = st.columns([1, 2])

                                with card_col1:
                                    employee_photo_path = get_employee_photo_path(item["photo"])
                                    if employee_photo_path:
                                        st.image(
                                            employee_photo_path,
                                            caption=item["full_name"],
                                            width=180
                                        )
                                    else:
                                        st.info("Фото сотрудника не найдено")

                                with card_col2:
                                    st.write(f"**#{i} {item['full_name']}**")
                                    st.write(f"Должность: {item['position']}")
                                    st.write(f"Employee ID: {item['employee_id']}")
                                    st.write(f"Distance: {item['distance']}")
                                    st.write(f"Confidence: {item['confidence']}%")
                                    st.write(f"Photo: {item['photo']}")

            except FaceModelError as e:
                st.error(f"Ошибка модели: {e}")
            except Exception as e:
                st.error(f"Общая ошибка: {e}")
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except Exception:
                        pass


def render_history_page():
    user = st.session_state.user
    touch_session_activity()

    st.title("История поисков")
    st.write(f"История пользователя: **{user['username']}**")

    try:
        history = get_user_search_history(user["id"])

        if not history:
            st.info("История поиска пока пуста")
            return

        for item in history:
            timestamp_value = item.get("timestamp")
            search_id = item["id"]
            query_image_name = item["query_image_name"]

            with st.expander(
                f"Поиск #{search_id} | {query_image_name} | {timestamp_value}",
                expanded=False
            ):
                st.write(f"**Search ID:** {search_id}")
                st.write(f"**Имя файла:** {query_image_name}")
                st.write(f"**Время:** {timestamp_value}")

                search_results = get_search_results_by_search_id(search_id)

                if not search_results:
                    st.warning("Для этого поиска не найдено результатов")
                    continue

                st.subheader("Найденные совпадения")

                for idx, result in enumerate(search_results, start=1):
                    with st.container(border=True):
                        card_col1, card_col2 = st.columns([1, 2])

                        with card_col1:
                            employee_photo_path = get_employee_photo_path(result["photo"])
                            if employee_photo_path:
                                st.image(
                                    employee_photo_path,
                                    caption=result["full_name"],
                                    width=180
                                )
                            else:
                                st.info("Фото сотрудника не найдено")

                        with card_col2:
                            st.write(f"**#{idx} {result['full_name']}**")
                            st.write(f"Должность: {result['position']}")
                            st.write(f"Employee ID: {result['employee_id']}")
                            st.write(f"Distance: {result['similarity_score']}")
                            st.write(f"Confidence: {result['confidence']}%")
                            st.write(f"Photo: {result['photo']}")

    except Exception as e:
        st.error(f"Ошибка загрузки истории: {e}")


def render_admin_page():
    user = st.session_state.user
    touch_session_activity()

    if not user.get("is_admin"):
        st.error("Доступ запрещён. Эта страница доступна только администраторам.")
        return

    st.title("Админ-панель")
    st.write(f"Вы вошли как администратор: **{user['username']}**")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Добавить пользователя",
        "Добавить сотрудника",
        "Список пользователей",
        "Список сотрудников",
    ])

    with tab1:
        st.subheader("Создание пользователя приложения")

        with st.form("admin_create_user_form"):
            username = st.text_input("Логин нового пользователя")
            password = st.text_input("Пароль", type="password")
            is_admin = st.checkbox("Сделать администратором", value=False)
            submitted = st.form_submit_button("Создать пользователя")

            if submitted:
                touch_session_activity()
                try:
                    user_id = create_user(username, password, is_admin=is_admin)
                    st.success(f"Пользователь создан. id = {user_id}")
                except Exception as e:
                    st.error(f"Ошибка создания пользователя: {e}")

    with tab2:
        st.subheader("Добавление сотрудника в базу распознавания")

        employee_photo = st.file_uploader(
            "Фото сотрудника",
            type=["jpg", "jpeg", "png"],
            key="admin_employee_photo"
        )

        if employee_photo is not None:
            st.markdown("**Предпросмотр фото сотрудника**")
            preview_col1, preview_col2, preview_col3 = st.columns([1, 2, 1])
            with preview_col2:
                st.image(
                    employee_photo,
                    caption="Фото сотрудника",
                    width=220
                )

        with st.form("admin_create_employee_form"):
            employee_id = st.number_input("ID сотрудника", min_value=1, step=1)
            full_name = st.text_input("ФИО")
            position = st.text_input("Должность")
            submitted = st.form_submit_button("Добавить сотрудника")

            if submitted:
                touch_session_activity()
                saved_photo_path = None

                try:
                    if not full_name.strip():
                        st.error("ФИО не может быть пустым")
                    elif not position.strip():
                        st.error("Должность не может быть пустой")
                    elif employee_photo is None:
                        st.error("Нужно загрузить фото")
                    else:
                        with st.spinner("Добавление сотрудника и построение эмбеддинга..."):
                            saved_photo_path, saved_photo_name = save_employee_photo(
                                employee_photo,
                                int(employee_id)
                            )

                            create_employee_with_embedding(
                                employee_id=int(employee_id),
                                full_name=full_name.strip(),
                                position=position.strip(),
                                image_path=saved_photo_path,
                                photo_value=saved_photo_name,
                            )

                        st.success("Сотрудник успешно добавлен")

                except Exception as e:
                    if saved_photo_path and os.path.exists(saved_photo_path):
                        try:
                            os.remove(saved_photo_path)
                        except Exception:
                            pass
                    st.error(f"Ошибка добавления сотрудника: {e}")

    with tab3:
        st.subheader("Пользователи приложения")

        try:
            users = get_all_users()
            if not users:
                st.info("Пользователей пока нет")
            else:
                st.dataframe(users, use_container_width=True)
        except Exception as e:
            st.error(f"Ошибка загрузки пользователей: {e}")

    with tab4:
        st.subheader("Сотрудники в базе распознавания")

        try:
            employees = get_all_employees_short()
            if not employees:
                st.info("Сотрудников пока нет")
            else:
                st.dataframe(employees, use_container_width=True)
        except Exception as e:
            st.error(f"Ошибка загрузки сотрудников: {e}")


def main():
    init_session_state()
    delete_expired_sessions()

    watchdog_count = st_autorefresh(interval=60000, key="session_watchdog")
    watchdog_triggered = is_watchdog_rerun(watchdog_count)

    if st.session_state.user is None:
        restore_user_from_cookie()

    if watchdog_triggered:
        enforce_inactivity_timeout()

    if st.session_state.user is None:
        render_auth_page()
        return

    render_sidebar()

    if st.session_state.page == "search":
        render_search_page()
    elif st.session_state.page == "history":
        render_history_page()
    elif st.session_state.page == "admin":
        render_admin_page()
    else:
        render_search_page()


if __name__ == "__main__":
    main()