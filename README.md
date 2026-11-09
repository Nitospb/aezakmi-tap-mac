# Aezakmi Tap Mac

Эксперименты с распознаванием касаний несенсорного экрана MacBook по акселерометру, гироскопу, точному углу крышки и освещённости.

- `outputs/tap-model/live/` — приложение предсказания, сбор обратной связи и переобучение.
- `outputs/tap-model/live/test-mode/TapTest.swift` — свободный тест: пробел, тап, прогноз, автоматическое ожидание следующего тапа.
- `outputs/tap-collector/` и `outputs/TapCollector.swift` — сбор обучающих данных.
- `outputs/sensor-scope/` — просмотр датчиков.
- `outputs/tap-model/` — обучение, признаки, оценки и эксперименты.
- `outputs/README.txt` и вложенные README — подробности форматов и работы приложений.

## Окружение

macOS, Apple Silicon, Xcode Command Line Tools, Python 3.11. Доступность датчиков зависит от модели Mac.

```sh
python3.11 -m venv work/model-env
work/model-env/bin/pip install -r requirements.txt
```

Приложения используют относительное расположение `outputs/` и `work/model-env/`. Полный снимок с готовыми приложениями, весами моделей, записями датчиков, микрофоном и рабочими скриптами находится в assets приватного релиза `snapshot-2026-11-09`. Python-окружение и кэши сборки исключены; их нужно создать заново.

## Восстановление полного снимка

Скачайте все части `project-snapshot.tar.gz.part*` из одного релиза в отдельную папку:

```sh
cat project-snapshot.tar.gz.part* > project-snapshot.tar.gz
shasum -a 256 -c SHA256SUMS
mkdir restored
 tar -xzf project-snapshot.tar.gz -C restored
```

Архив сохраняет исходные относительные пути; отдельные сохранённые метаданные моделей могут содержать абсолютные пути старого компьютера, их потребуется обновить при переносе. Исходная модель: `outputs/tap-model/light-dynamics/model.joblib`. Приватность репозитория следует сохранять: архив содержит сырые записи, включая аудио.

Исходная лицензия LidTap сохранена в `outputs/LidTap-LICENSE.txt`.
