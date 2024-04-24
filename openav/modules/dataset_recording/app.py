from flask import Flask, request, render_template, jsonify, send_file
import os
import csv
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
global video_path, output_dir
global processing_finished

processing_finished = False
video_path = "temp_dir/video.mp4"
output_dir = "cuted"


def read_questions_from_csv():
    questions = []
    with open("questions.csv", mode="r", newline="", encoding="utf-8") as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            question_number = int(row["QuestionNumber"])
            question_text = row["QuestionText"]
            disable_time = row["Disable_time"]
            questions.append(
                {"QuestionNumber": question_number, "QuestionText": question_text, "Disable_time": disable_time}
            )
    return questions


@app.route("/get_questions", methods=["GET"])
def get_questions():
    """Получение списка вопросов"""
    questions = read_questions_from_csv()
    return jsonify(questions=questions)


# Configure the upload folder for recorded videos
app.config["UPLOAD_FOLDER"] = "static/recorded-video"

if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])


@app.route("/")
def index():
    """Отображение основной страницы записи"""
    global processing_finished
    processing_finished = False
    return render_template("index_home.html")


@app.route("/store_timing_data", methods=["POST"])
def store_timing_data():
    """Сохранение файла временных отметок записи"""
    if request.method == "POST":
        data = request.json
        if os.path.exists("timing_data.txt"):
            os.remove("timing_data.txt")
        with open("timing_data.txt", "w", encoding="utf-8") as file:
            for item in data:
                file.write(f"Question: {item['question']}, Timestamp: {item['timestamp']}\n")

            file.close()
        return "Data stored successfully"


@app.route("/upload", methods=["POST"])
def upload():
    """Сохранение файла записи"""
    global processing_finished
    if "video" in request.files:
        video_file = request.files["video"]
        if video_file.filename != "":
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], video_file.filename) + ".webm"
            print(video_path)
            # video_file.save(video_path)
            if not os.path.exists("temp_dir"):
                os.makedirs("temp_dir")
            video_file.save("temp_dir/video.webm")
            # convert_webm_to_mp4('temp_dir/video.webm', 'temp_dir/video.mp4')
            print("Done")
            processing_finished = True

            return "DONE"

    return "No video data received."


@app.route("/download_processed_video")
def download_processed_video():
    """Выгрузка файла записи"""
    processed_video_path = "temp_dir/video.webm"
    return send_file(processed_video_path, as_attachment=True)


@app.route("/download_timing_data")
def download_timing_data():
    """Выгрузка файла временных отметок записи"""
    timing_data_path = "timing_data.txt"  # Path to the file
    # Set cache-control headers to prevent caching
    response = send_file(timing_data_path, as_attachment=True, mimetype="text/plain")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


if __name__ == "__main__":
    app.run()
