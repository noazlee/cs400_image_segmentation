import io
import os
import random
import sys
import pysqlite3 as sqlite3

from flask import Flask, abort, redirect, render_template, request, send_file, send_from_directory, url_for

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dustification import DEFAULT_PARAMS, apply_martian_dust_storm

EXAMPLES = os.path.join(ROOT, "dust_tuner", "examples")
DB_PATH = os.path.join(ROOT, "dust_tuner", "submissions.db")
NAMES = list(DEFAULT_PARAMS)

app = Flask(__name__)

def init_db():
    cols = ", ".join(f"{n} REAL NOT NULL" for n in NAMES)
    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS submissions ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL,"
            "clear_image TEXT NOT NULL, dusty_image TEXT NOT NULL, seed INTEGER NOT NULL,"
            f"{cols}, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )

@app.route("/")
def index():
    return render_template("index.html",
        clear=random.choice(os.listdir(os.path.join(EXAMPLES, "clear"))),
        dusty=random.choice(os.listdir(os.path.join(EXAMPLES, "dusty"))),
        seed=random.randint(0, 2**31 - 1),
        username=request.args.get("username",""),
    )

@app.route("/examples/<path:path>")
def examples(path):
    return send_from_directory(EXAMPLES, path)

@app.route("/preview")
def preview():
    clear_path = os.path.join(EXAMPLES, "clear", os.path.basename(request.args["clear"]))
    if not os.path.isfile(clear_path):
        abort(404)
    values = {n: float(request.args.get(n, DEFAULT_PARAMS[n])) for n in NAMES}
    img = apply_martian_dust_storm(clear_path, values, int(request.args["seed"]))
    buf = io.BytesIO()
    img.save(buf, "JPEG",quality=90)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")

@app.route("/submit", methods=["POST"])
def submit():
    data = request.form
    username = data.get("username", "").strip()
    if not username:
        abort(400, "you need a username fool!")
    row = [username, data["clear"], data["dusty"], int(data["seed"])] + [float(data[n]) for n in NAMES]
    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            f"INSERT INTO submissions (username, clear_image, dusty_image, seed, {', '.join(NAMES)}) "
            f"VALUES ({', '.join('?' * len(row))})",
            row
        )
    return redirect(url_for("index", username=username))

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=11243)

# python -c "import pysqlite3 as sqlite3; cur = sqlite3.connect('dust_tuner/submissions.db').execute('select * from submissions'); print([c[0] for c in cur.description]); [print(r) for r in cur]"
# ^ print every row in the db