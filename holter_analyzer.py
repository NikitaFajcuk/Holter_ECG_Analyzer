import sys
import os
import wfdb
import numpy as np
from google import genai
from collections import Counter
import hashlib
import re
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QTextEdit, QStatusBar, QTabWidget,
                             QDialog, QDialogButtonBox, QLineEdit, QMessageBox, QStackedWidget)
from PyQt6.QtCore import Qt, QObject, pyqtSignal, pyqtSlot, QThread
from dotenv import load_dotenv
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtGui import QIcon

BEAT_ANNOTATION_MAP = {
    'N': 'Нормальні (N)', 'L': 'Блокада лівої ніжки (L)', 'R': 'Блокада правої ніжки (R)',
    'V': 'Шлуночкові екстрасистоли (PVC)', 'A': 'Передсердні екстрасистоли (APC)',
    'S': 'Надшлуночкові екстрасистоли (SVE)', 'J': 'Вузлові екстрасистоли (J)',
    'F': 'Злиті шлуночкові (F)', '/': 'Paced beat', 'Q': 'Невизначені (Q)',
}

load_dotenv()


# ==============================================================================
# 1. КЛАС ДЛЯ УПРАВЛІННЯ АВТОРИЗАЦІЄЮ
# ==============================================================================
class AuthManager:
    def __init__(self, user_file='doctors.txt'):
        self.user_file = user_file

    def _hash_data(self, data):
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def _load_users(self):
        users = {}
        try:
            with open(self.user_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(':', 4)
                    if len(parts) == 5:
                        hashed_login, hashed_pass, first, middle, last = parts
                        users[hashed_login] = {
                            'password': hashed_pass,
                            'full_name': f"{last} {first} {middle}"
                        }
        except FileNotFoundError:
            pass
        return users

    def register_user(self, login, password, first_name, middle_name, last_name):
        if not all([login, password, first_name, last_name]):
            return False, "Всі поля, окрім по-батькові, є обов'язковими."
        users = self._load_users()
        hashed_login = self._hash_data(login)
        if hashed_login in users:
            return False, "Користувач з таким логіном вже існує."

        hashed_password = self._hash_data(password)
        user_data_line = f"{hashed_login}:{hashed_password}:{first_name}:{middle_name}:{last_name}\n"
        with open(self.user_file, 'a', encoding='utf-8') as f:
            f.write(user_data_line)
        return True, "Реєстрація успішна! Тепер ви можете увійти."

    def verify_user(self, login, password):
        users = self._load_users()
        hashed_login = self._hash_data(login)
        if hashed_login not in users:
            return False, None

        stored_user_data = users[hashed_login]
        if self._hash_data(password) == stored_user_data['password']:
            return True, stored_user_data['full_name']
        return False, None


# ==============================================================================
# 2. КЛАС ДЛЯ ВІКНА ВХОДУ ТА РЕЄСТРАЦІЇ
# ==============================================================================
class LoginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Авторизація")
        self.auth_manager = AuthManager()
        self.current_user_name = None
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(self._create_login_page())
        self.stacked_widget.addWidget(self._create_register_page())

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.stacked_widget)
        self.setFixedSize(400, 350)

    def _create_login_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<h2>Вхід для лікаря</h2>"))
        self.login_user = QLineEdit()
        self.login_user.setPlaceholderText("Логін")
        self.login_pass = QLineEdit()
        self.login_pass.setPlaceholderText("Пароль")
        self.login_pass.setEchoMode(QLineEdit.EchoMode.Password)

        login_button = QPushButton("Увійти")
        switch_to_reg_button = QPushButton("Зареєструватися")

        layout.addWidget(QLabel("Логін:"))
        layout.addWidget(self.login_user)
        layout.addWidget(QLabel("Пароль:"))
        layout.addWidget(self.login_pass)
        layout.addStretch()
        layout.addWidget(login_button)
        layout.addWidget(switch_to_reg_button)

        login_button.clicked.connect(self._handle_login)
        switch_to_reg_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        return page

    def _create_register_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<h3>Реєстрація нового лікаря</h3>"))

        self.reg_login = QLineEdit()
        self.reg_login.setPlaceholderText("Логін*")
        self.reg_pass = QLineEdit()
        self.reg_pass.setPlaceholderText("Пароль*")
        self.reg_pass.setEchoMode(QLineEdit.EchoMode.Password)
        self.reg_last_name = QLineEdit()
        self.reg_last_name.setPlaceholderText("Прізвище*")
        self.reg_first_name = QLineEdit()
        self.reg_first_name.setPlaceholderText("Ім'я*")
        self.reg_middle_name = QLineEdit()
        self.reg_middle_name.setPlaceholderText("По-батькові")

        register_button = QPushButton("Зареєструватися")
        switch_to_login_button = QPushButton("Вже є акаунт? Увійти")

        layout.addWidget(self.reg_login)
        layout.addWidget(self.reg_pass)
        layout.addWidget(self.reg_last_name)
        layout.addWidget(self.reg_first_name)
        layout.addWidget(self.reg_middle_name)
        layout.addWidget(register_button)
        layout.addWidget(switch_to_login_button)

        register_button.clicked.connect(self._handle_register)
        switch_to_login_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        return page

    def _handle_login(self):
        success, user_name = self.auth_manager.verify_user(self.login_user.text(), self.login_pass.text())
        if success:
            self.current_user_name = user_name
            self.accept()
        else:
            QMessageBox.warning(self, "Помилка входу", "Неправильний логін або пароль.")

    def _handle_register(self):
        success, message = self.auth_manager.register_user(
            self.reg_login.text(), self.reg_pass.text(),
            self.reg_first_name.text(), self.reg_middle_name.text(), self.reg_last_name.text()
        )
        if success:
            QMessageBox.information(self, "Успіх", message)
            self.stacked_widget.setCurrentIndex(0)
        else:
            QMessageBox.warning(self, "Помилка реєстрації", message)


# ==============================================================================
# 3. КЛАСИ АНАЛІЗУ ТА ОСНОВНОГО ВІКНА
# ==============================================================================
class AnalysisWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, folder_path, api_key):
        super().__init__()
        self.folder_path = folder_path
        self.api_key = api_key

    def _get_episodes(self, annotation):
        episodes = []
        start = None
        event_times_sec = [s / annotation.fs for s in annotation.sample]
        for t, sym in zip(event_times_sec, annotation.symbol):
            if sym == '*':
                start = t
            elif sym == '-' and start is not None:
                episodes.append((start, t))
                start = None
        return episodes

    @pyqtSlot()
    def run(self):
        try:
            self.progress.emit("Зчитування файлів...")
            record_name = os.path.basename(self.folder_path)
            path = os.path.join(self.folder_path, record_name)

            record = wfdb.rdrecord(path)
            annotation = wfdb.rdann(path, extension='atr')

            st_episodes_map = {}
            for ext in ['sta', 'stb', 'stc']:
                try:
                    self.progress.emit(f"Читання .{ext}...")
                    st_ann = wfdb.rdann(path, extension=ext)
                    st_episodes_map[ext] = self._get_episodes(st_ann)
                except FileNotFoundError:
                    st_episodes_map[ext] = []

            fs = record.fs
            signal_full = record.p_signal[:, 0]
            time_full = np.arange(len(signal_full)) / fs

            # Розрахунок ЧСС з захистом від ділення на нуль
            qrs_samples = annotation.sample
            rr_intervals_sec = np.diff(qrs_samples) / fs
            rr_intervals_sec = rr_intervals_sec[rr_intervals_sec > 0]  # Видаляємо нульові інтервали

            hr = 60 / rr_intervals_sec
            hrv_sdnn = np.std(rr_intervals_sec) * 1000
            avg_hr_val = np.mean(hr) if len(hr) > 0 else 0

            beat_counts = Counter(annotation.symbol)
            arrhythmia_data = {BEAT_ANNOTATION_MAP.get(s, s): c for s, c in beat_counts.items() if
                               s in BEAT_ANNOTATION_MAP}

            stf_file = os.path.join(self.folder_path, record_name + '.stf')
            st_dev = []
            if os.path.exists(stf_file):
                with open(stf_file, 'r') as f:
                    st_dev = [float(x.split()[2]) for x in f.readlines() if len(x.split()) >= 3]

            num_sta, num_stb, num_stc = len(st_episodes_map.get('sta', [])), len(st_episodes_map.get('stb', [])), len(
                st_episodes_map.get('stc', []))

            text_summary = (
                f"Запис: {record_name}\n\n"
                f"Середня ЧСС: {avg_hr_val:.1f} уд/хв\n"
                f"SDNN (HRV): {hrv_sdnn:.1f} мс\n\n"
                f"Кількість ST-епізодів (.sta/.stb/.stc): {num_sta}/{num_stb}/{num_stc}\n"
                f"(Критерії: .sta >75µV,>30с; .stb >100µV,>30с; .stc >100µV,>60с)\n\n"
                f"Підсумок по типах скорочень:\n"
            )
            for name, count in sorted(arrhythmia_data.items(), key=lambda item: item[1], reverse=True):
                text_summary += f"- {name}: {count}\n"

            self.progress.emit("Генерація висновку...")
            ai_conclusion = self.generate_conclusion_with_gemini(record_name, avg_hr_val, hrv_sdnn, num_sta, num_stc,
                                                                 arrhythmia_data)

            results = {
                "signal_data": (time_full, signal_full),
                "fs": fs,
                "hr_data": hr,
                "st_data": (np.arange(len(st_dev)) / fs if st_dev else np.array([]), st_dev),
                "st_episodes_map": st_episodes_map,
                "avg_hr": avg_hr_val,
                "rr_intervals": rr_intervals_sec,
                "arrhythmia_data": arrhythmia_data,
                "text_summary": text_summary,
                "ai_conclusion": ai_conclusion
            }
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(f"Помилка аналізу: {e}")

    def generate_conclusion_with_gemini(self, record_name, avg_hr, hrv_sdnn, num_episodes_sensitive,
                                        num_episodes_specific, arrhythmia_data):
        if not self.api_key or self.api_key == "":
            return "Помилка: API ключ для Google не вказано."

        arrhythmia_summary = "\n".join([f"- {name}: {count}" for name, count in arrhythmia_data.items()])

        prompt = f"""
        Роль
        Ви — висококваліфікований кардіолог, що спеціалізується на інтерпретації даних Холтерівського моніторування ЕКГ. Ваш висновок має бути професійним, структурованим та клінічно обґрунтованим.

        Вхідні дані пацієнта
        - Назва запису: {record_name}
        - Середня ЧСС: {avg_hr:.1f} уд/хв
        - Варіабельність серцевого ритму (SDNN): {hrv_sdnn:.1f} мс
        - Ішемічні епізоди (чутливі критерії, .sta): {num_episodes_sensitive}
        - Ішемічні епізоди (специфічні критерії, .stc): {num_episodes_specific}
        - Підсумок по типах скорочень:
        {arrhythmia_summary}

        Завдання
        Напишіть детальний медичний висновок. Дотримуйтесь наступної структури:
        1. Загальний висновок: Підсумуйте основні знахідки.
        2. Детальний аналіз:
            - Ритм та ЧСС.
            - Порушення ритму (Аритмії).
            - Аналіз сегмента ST (Ішемія).
            - Кореляція знахідок.

        Обмеження
        - Не ставте остаточних діагнозів. Використовуйте "може свідчити про".
        - Не використовуйте форматування Markdown. Пишіть чистим текстом.
        """

        try:
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            return re.sub(r'\s{2,}', ' ', response.text.strip())
        except Exception as e:
            return f"Помилка під час звернення до ШІ: {e}"


class MainWindow(QMainWindow):
    def __init__(self, current_user_name):
        super().__init__()
        self.setWindowTitle(f"Розширений Аналізатор ЕКГ - Лікар: {current_user_name}")
        self.setGeometry(100, 100, 1200, 900)

        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            QMessageBox.critical(self, "Помилка", "GEMINI_API_KEY не знайдено. Перевірте файл .env")

        self.folder_path = ""
        self.select_folder_button = QPushButton("1. Вибрати папку з записом")
        self.selected_folder_label = QLabel("Папку не вибрано")
        self.analyze_button = QPushButton("2. Почати аналіз")
        self.analyze_button.setEnabled(False)
        self.analyze_button.setStyleSheet("font-weight: bold; font-size: 14px")

        self.tabs_plots = QTabWidget()
        self.ecg_canvas = FigureCanvas(Figure(figsize=(12, 4)))
        self.hr_canvas = FigureCanvas(Figure(figsize=(12, 4)))
        self.st_canvas = FigureCanvas(Figure(figsize=(12, 4)))
        self.arrhythmia_canvas = FigureCanvas(Figure(figsize=(12, 4)))
        self.rr_hist_canvas = FigureCanvas(Figure(figsize=(12, 4)))
        self.poincare_canvas = FigureCanvas(Figure(figsize=(12, 4)))

        self.tabs_plots.addTab(self.ecg_canvas, "Фрагмент ЕКГ")
        self.tabs_plots.addTab(self.hr_canvas, "Динаміка ЧСС")
        self.tabs_plots.addTab(self.st_canvas, "ST-девіація (комбінований)")
        self.tabs_plots.addTab(self.arrhythmia_canvas, "Аналіз аритмій")
        self.tabs_plots.addTab(self.rr_hist_canvas, "Гістограма RR")
        self.tabs_plots.addTab(self.poincare_canvas, "Діаграма Пуанкаре")

        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.select_folder_button)
        top_layout.addWidget(self.selected_folder_label, 1)
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.analyze_button)
        main_layout.addWidget(self.tabs_plots)

        self.summary_text_edit = QTextEdit()
        self.ai_conclusion_edit = QTextEdit()
        self.tabs_results = QTabWidget()
        self.tabs_results.addTab(self.summary_text_edit, "Статистичний підсумок")
        self.tabs_results.addTab(self.ai_conclusion_edit, "Висновок від AI (Gemini)")
        main_layout.addWidget(self.tabs_results)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.setStatusBar(QStatusBar(self))

        self.select_folder_button.clicked.connect(self.select_folder)
        self.analyze_button.clicked.connect(self.start_analysis)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Виберіть папку")
        if folder:
            self.folder_path = folder
            self.selected_folder_label.setText(f"Вибрано: ...{self.folder_path[-50:]}")
            self.analyze_button.setEnabled(True)
            self.statusBar().showMessage(f"Папку {os.path.basename(folder)} вибрано.", 5000)

    def start_analysis(self):
        self.analyze_button.setEnabled(False)
        self.analyze_button.setText("Обробка...")
        self.statusBar().showMessage("Початок аналізу...")

        self.summary_text_edit.clear()
        self.ai_conclusion_edit.clear()

        self.thread = QThread()
        self.worker = AnalysisWorker(self.folder_path, self.api_key)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.update_ui_with_results)
        self.worker.error.connect(self.show_error)
        self.worker.progress.connect(self.update_status)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def update_ui_with_results(self, results):
        self.statusBar().showMessage("Аналіз успішно завершено!", 5000)
        self.plot_st(results["st_data"], results["st_episodes_map"])
        self.plot_ecg(results["signal_data"], results["fs"], results["st_episodes_map"].get('sta', []))
        self.plot_hr(results["hr_data"], results["avg_hr"])
        self.plot_arrhythmias(results["arrhythmia_data"])
        self.plot_rr_histogram(results["rr_intervals"])
        self.plot_poincare(results["rr_intervals"])
        self.summary_text_edit.setText(results["text_summary"])
        self.ai_conclusion_edit.setText(results["ai_conclusion"])
        self.analyze_button.setText("2. Почати аналіз")
        self.analyze_button.setEnabled(True)

    def plot_st(self, st_data, st_episodes_map):
        self.st_canvas.figure.clear()
        ax = self.st_canvas.figure.add_subplot(111)
        st_time, st_dev = st_data

        if len(st_time) > 0:
            ax.plot(st_time, st_dev, label="ST-девіація", color='black', linewidth=0.7)

        colors = {'sta': 'red', 'stb': 'orange', 'stc': 'purple'}
        alphas = {'sta': 0.2, 'stb': 0.3, 'stc': 0.4}
        labels = {'sta': 'Епізоди .sta', 'stb': 'Епізоди .stb', 'stc': 'Епізоди .stc'}
        has_episodes = False

        for ext, episodes in st_episodes_map.items():
            if episodes:
                has_episodes = True
                start, end = episodes[0]
                ax.axvspan(start, end, color=colors.get(ext, 'gray'), alpha=alphas.get(ext, 0.2), label=labels.get(ext))
                for s, e in episodes[1:]:
                    ax.axvspan(s, e, color=colors.get(ext, 'gray'), alpha=alphas.get(ext, 0.2))

        ax.set_title("Комбінований аналіз ST-девіації")
        ax.set_xlabel("Час, с")
        ax.set_ylabel("Девіація (мВ)")
        ax.grid(True, linestyle='--', alpha=0.6)

        if has_episodes:
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Ішемічні епізоди не виявлені', ha='center', va='center', transform=ax.transAxes,
                    fontsize=16, color='gray',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", lw=1, alpha=0.8))
        self.st_canvas.draw()

    def clear_plots(self):
        canvases = [self.ecg_canvas, self.hr_canvas, self.st_canvas, self.arrhythmia_canvas, self.rr_hist_canvas,
                    self.poincare_canvas]
        for canvas in canvases:
            canvas.figure.clear()
            canvas.draw()

    def plot_arrhythmias(self, arrhythmia_data):
        self.arrhythmia_canvas.figure.clear()
        ax = self.arrhythmia_canvas.figure.add_subplot(111)
        sorted_data = sorted(arrhythmia_data.items(), key=lambda item: item[1])
        names, counts = [item[0] for item in sorted_data], [item[1] for item in sorted_data]
        bars = ax.barh(names, counts, color='skyblue')
        ax.set_title("Типи серцевих скорочень")
        ax.set_xlabel("Кількість")
        ax.bar_label(bars, padding=3)
        self.arrhythmia_canvas.figure.tight_layout()
        self.arrhythmia_canvas.draw()

    def plot_rr_histogram(self, rr_intervals):
        self.rr_hist_canvas.figure.clear()
        ax = self.rr_hist_canvas.figure.add_subplot(111)
        ax.hist(rr_intervals * 1000, bins=50, color='lightgreen', edgecolor='black')
        ax.set_title("Гістограма RR-інтервалів")
        ax.set_xlabel("Тривалість RR (мс)")
        ax.set_ylabel("Кількість")
        ax.grid(True, linestyle='--', alpha=0.6)
        self.rr_hist_canvas.draw()

    def plot_poincare(self, rr_intervals):
        self.poincare_canvas.figure.clear()
        ax = self.poincare_canvas.figure.add_subplot(111)
        if len(rr_intervals) > 1:
            rr_n, rr_n1 = rr_intervals[:-1] * 1000, rr_intervals[1:] * 1000
            ax.scatter(rr_n, rr_n1, alpha=0.5, s=10)
            ax.set_aspect('equal', 'box')
            min_val, max_val = min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Лінія ідентичності')
            ax.legend()

        ax.set_title("Діаграма Пуанкаре (HRV)")
        ax.set_xlabel("RR(n) (мс)")
        ax.set_ylabel("RR(n+1) (мс)")
        ax.grid(True, linestyle='--', alpha=0.6)
        self.poincare_canvas.draw()

    def update_status(self, message):
        self.statusBar().showMessage(message)

    def show_error(self, message):
        self.ai_conclusion_edit.setText(f"ПОМИЛКА:\n\n{message}")
        self.statusBar().showMessage("Помилка аналізу!", 5000)
        self.analyze_button.setText("2. Почати аналіз")
        self.analyze_button.setEnabled(True)

    def plot_ecg(self, signal_data, fs, episodes):
        time_full, signal_full = signal_data
        self.ecg_canvas.figure.clear()
        ax = self.ecg_canvas.figure.add_subplot(111)

        if episodes:
            start_time_sec = episodes[0][0]
            start_index = int(start_time_sec * fs)
            end_index = int((start_time_sec + 10) * fs)
            title = "ЕКГ: Початок ішемічного епізоду"
        else:
            start_index = 0
            end_index = int(10 * fs)
            title = "ЕКГ: Перші 10 секунд"

        ax.plot(time_full[start_index:end_index], signal_full[start_index:end_index])
        ax.set_title(title)
        ax.set_xlabel("Час, с")
        ax.set_ylabel("мВ")
        ax.grid(True)
        self.ecg_canvas.draw()

    def plot_hr(self, hr_data, avg_hr):
        self.hr_canvas.figure.clear()
        ax = self.hr_canvas.figure.add_subplot(111)
        ax.plot(hr_data, label="Динаміка ЧСС")
        ax.axhline(y=avg_hr, color='r', linestyle='--', label=f'Середня ЧСС: {avg_hr:.1f}')
        ax.set_title("Частота серцевих скорочень")
        ax.set_xlabel("Номер удару")
        ax.set_ylabel("ЧСС (уд/хв)")
        ax.legend()
        ax.grid(True)
        self.hr_canvas.draw()


class GraphDescriptionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Опис графіків та елементів")
        self.setMinimumSize(700, 550)

        layout = QVBoxLayout(self)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setHtml("""
            <h2>Опис графіків та елементів аналізу</h2>
            <p>Цей посібник пояснює призначення кожного візуального елемента в програмі.</p>

            <h3>1. Фрагмент ЕКГ</h3>
            <p><b>Що це:</b> Короткий відрізок (10 секунд) "сирого" сигналу ЕКГ.</p>
            <p><b>Навіщо потрібен:</b></p>
            <ul>
                <li><b>Для візуальної верифікації:</b> Дозволяє лікарю миттєво оцінити якість запису, наявність шумів та артефактів.</li>
                <li><b>Для аналізу морфології:</b> Це найважливіше. Лікар може побачити реальну форму комплексів QRS та сегмента ST під час патологічних подій.</li>
                <li><b>"Розумний" вибір фрагмента:</b> Якщо програма знаходить ішемічні епізоди, вона показує ЕКГ саме на початку першого епізоду, що є найбільш інформативним.</li>
            </ul>

            <h3>2. Динаміка ЧСС</h3>
            <p><b>Що це:</b> Графік зміни частоти серцевих скорочень протягом усього періоду моніторування.</p>

            <h3>3. ST-девіація (комбінований)</h3>
            <p><b>Що це:</b> Основний графік для виявлення ішемії міокарда. Показує відхилення сегмента ST від ізолінії.</p>
            <p><b>Навіщо потрібен:</b> Кольорові зони показують ішемічні епізоди, виявлені за різними критеріями:</p>
            <ul>
                <li><b>Червоний (.sta):</b> Найчутливіші критерії (найменше відхилення).</li>
                <li><b>Оранжевий (.stb):</b> Більш суворі критерії.</li>
                <li><b>Фіолетовий (.stc):</b> Найбільш специфічні та клінічно значущі критерії.</li>
            </ul>

            <h3>4. Аналіз аритмій</h3>
            <p><b>Що це:</b> Стовпчаста діаграма, що показує загальну кількість кожного типу серцевих скорочень.</p>

            <h3>5. Гістограма RR</h3>
            <p><b>Що це:</b> Графік, що показує розподіл тривалості інтервалів між серцевими скороченнями (RR-інтервалів).</p>

            <h3>6. Діаграма Пуанкаре</h3>
            <p><b>Що це:</b> Потужний візуальний метод аналізу варіабельності серцевого ритму (HRV).</p>
        """)
        layout.addWidget(text_edit)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Закрити")
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)


class WelcomeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ласкаво просимо!")
        self.setMinimumWidth(600)
        layout = QVBoxLayout(self)

        title_label = QLabel("Аналізатор Холтерівського моніторування ЕКГ")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px")
        layout.addWidget(title_label)

        description_label = QLabel(
            "<b>Що робить програма:</b><ul><li>Аналізує дані Холтера.</li><li>Розраховує ключові показники.</li>"
            "<li>Виявляє аритмії та ішемію.</li><li>Візуалізує результати.</li><li>Використовує AI для генерації висновку.</li></ul>"
        )
        description_label.setWordWrap(True)
        layout.addWidget(description_label)

        instruction_label = QLabel(
            "<b>Як користуватися:</b><ol><li>Натисніть <b>'1. Вибрати папку'</b>.</li><li>Виберіть папку з файлами дослідження.</li>"
            "<li>Натисніть <b>'2. Почати аналіз'</b>.</li><li>Переглядайте результати.</li></ol>"
        )
        instruction_label.setWordWrap(True)
        layout.addWidget(instruction_label)

        button_box = QDialogButtonBox()
        ok_button = button_box.addButton("Зрозуміло, почати!", QDialogButtonBox.ButtonRole.AcceptRole)
        desc_button = button_box.addButton("Детальний опис елементів", QDialogButtonBox.ButtonRole.ActionRole)

        button_box.accepted.connect(self.accept)
        desc_button.clicked.connect(self._show_description)
        layout.addWidget(button_box)

    def _show_description(self):
        desc_dialog = GraphDescriptionDialog(self)
        desc_dialog.exec()


# ==============================================================================
# 4. ТОЧКА ВХОДУ В ПРОГРАМУ
# ==============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    icon_path = 'app_icon.png'
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    login_dialog = LoginDialog()
    if login_dialog.exec() == QDialog.DialogCode.Accepted:
        current_user = login_dialog.current_user_name
        welcome_dialog = WelcomeDialog()
        if welcome_dialog.exec() == QDialog.DialogCode.Accepted:
            window = MainWindow(current_user)
            window.show()
            sys.exit(app.exec())
    else:
        sys.exit(0)