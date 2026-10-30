import os
import uuid

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from ocr import extract_text_from_image
from summarizer import summarize_document


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Maximum upload size: 50 MB
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


# ============================================================
# ALLOWED FILE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    "txt",
    "jpg",
    "jpeg",
    "png"
}


def allowed_file(filename):

    if "." not in filename:
        return False

    extension = filename.rsplit(
        ".",
        1
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# SUMMARIZATION API
# ============================================================

@app.route(
    "/summarize",
    methods=["POST"]
)
def summarize():

    file_path = None

    try:

        # ----------------------------------------------------
        # GET REDUCTION PERCENTAGE
        # ----------------------------------------------------

        reduction_value = request.form.get(
            "reduction",
            "70"
        )

        try:

            reduction = int(
                reduction_value
            )

        except ValueError:

            return jsonify({
                "success": False,
                "error":
                    "Invalid reduction percentage."
            }), 400


        # ----------------------------------------------------
        # VALIDATE REDUCTION
        # ----------------------------------------------------

        if reduction < 10 or reduction > 90:

            return jsonify({
                "success": False,
                "error":
                    "Reduction must be between 10% and 90%."
            }), 400


        # ----------------------------------------------------
        # GET FILE
        # ----------------------------------------------------

        uploaded_file = request.files.get(
            "file"
        )


        # ----------------------------------------------------
        # GET DIRECT TEXT
        # ----------------------------------------------------

        input_text = request.form.get(
            "text",
            ""
        ).strip()


        extracted_text = ""


        # ====================================================
        # CASE 1 — FILE UPLOAD
        # ====================================================

        if (
            uploaded_file
            and uploaded_file.filename
        ):

            original_filename = secure_filename(
                uploaded_file.filename
            )


            if not allowed_file(
                original_filename
            ):

                return jsonify({

                    "success": False,

                    "error":
                        "Unsupported file type. "
                        "Please use TXT, JPG, JPEG or PNG."

                }), 400


            # ------------------------------------------------
            # UNIQUE FILE NAME
            # ------------------------------------------------

            extension = (
                original_filename
                .rsplit(".", 1)[1]
                .lower()
            )


            unique_filename = (
                f"{uuid.uuid4().hex}."
                f"{extension}"
            )


            file_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                unique_filename
            )


            uploaded_file.save(
                file_path
            )


            # ------------------------------------------------
            # TEXT FILE
            # ------------------------------------------------

            if extension == "txt":

                with open(
                    file_path,
                    "r",
                    encoding="utf-8",
                    errors="ignore"
                ) as file:

                    extracted_text = (
                        file.read()
                    )


            # ------------------------------------------------
            # IMAGE FILE
            # ------------------------------------------------

            else:

                extracted_text = (
                    extract_text_from_image(
                        file_path
                    )
                )


        # ====================================================
        # CASE 2 — DIRECT TEXT
        # ====================================================

        elif input_text:

            extracted_text = input_text


        # ====================================================
        # NO INPUT
        # ====================================================

        else:

            return jsonify({

                "success": False,

                "error":
                    "Please enter text or upload a file."

            }), 400


        # ====================================================
        # VALIDATE EXTRACTED TEXT
        # ====================================================

        if not extracted_text:

            return jsonify({

                "success": False,

                "error":
                    "No text was found."

            }), 400


        extracted_text = extracted_text.strip()


        if not extracted_text:

            return jsonify({

                "success": False,

                "error":
                    "No readable text was found."

            }), 400


        # ====================================================
        # WORD COUNT
        # ====================================================

        original_words = len(
            extracted_text.split()
        )


        if original_words < 10:

            return jsonify({

                "success": False,

                "error":
                    "Please provide at least 10 words."

            }), 400


        # ====================================================
        # CALCULATE TARGET WORDS
        # ====================================================

        target_words = max(
            5,
            round(
                original_words *
                (1 - reduction / 100)
            )
        )


        # ====================================================
        # AI SUMMARIZATION
        # ====================================================

        result = summarize_document(

            extracted_text,

            reduction_percentage=reduction

        )


        # ====================================================
        # GET RESULT
        # ====================================================

        summary = result["summary"]

        summary_words = result[
            "summary_words"
        ]

        chunks = result[
            "chunks"
        ]


        # ====================================================
        # ACTUAL COMPRESSION
        # ====================================================

        if original_words > 0:

            actual_compression = (
                (1 -
                 summary_words /
                 original_words)
                * 100
            )

        else:

            actual_compression = 0


        actual_compression = round(
            actual_compression,
            1
        )


        # ====================================================
        # RETURN RESPONSE
        # ====================================================

        return jsonify({

            "success": True,

            "summary": summary,

            "original_words":
                original_words,

            "summary_words":
                summary_words,

            "chunks":
                chunks,

            "requested_reduction":
                reduction,

            "target_words":
                target_words,

            "compression":
                actual_compression

        })


    except Exception as e:

        print(
            "\nERROR:",
            str(e)
        )

        return jsonify({

            "success": False,

            "error":
                f"Processing failed: {str(e)}"

        }), 500


    finally:

        # ----------------------------------------------------
        # DELETE TEMPORARY UPLOAD
        # ----------------------------------------------------

        if file_path:

            try:

                if os.path.exists(
                    file_path
                ):

                    os.remove(
                        file_path
                    )

            except OSError:

                pass


# ============================================================
# FILE TOO LARGE
# ============================================================

@app.errorhandler(413)
def request_entity_too_large(error):

    return jsonify({

        "success": False,

        "error":
            "File is too large. Maximum size is 50 MB."

    }), 413


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print(
        "ADVANCED TEXT SUMMARIZATION SYSTEM"
    )
    print("=" * 60)
    print(
        "Starting Flask server..."
    )
    print(
        "Open: http://127.0.0.1:5000"
    )
    print("=" * 60)
    print()


    app.run(

        host="127.0.0.1",

        port=5001,

        debug=True

    )